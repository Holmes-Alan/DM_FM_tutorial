"""
=============================================================================
  DINOSAUR GENERATIVE MODELS — TESTING & EVALUATION
  test.py
=============================================================================
  Evaluates trained generative models on the Datasaurus dinosaur dataset.
  For each model, sweeps over sampling steps and records quality metrics.

  Models:
    vae           : samples from N(0,I) → decoder  (no steps)
    ddpm          : ancestral sampling  T, T/2, T/4 ... steps
    ddim          : deterministic ODE   1000→1 steps
    score         : annealed Langevin   steps sweep
    consistency   : 1, 2, 4, 8 steps
    flow          : Euler ODE           1→200 steps
    rectified     : Euler ODE           1→200 steps
    meanflow      : 1, 2, 4, 8 steps
    drifting      : 1, 2, 4, 8 steps

  Metrics (all lower is better):
    MMD           : Maximum Mean Discrepancy (rbf kernel) vs. ground truth
    Sinkhorn      : Sinkhorn divergence ≈ Wasserstein distance
    Coverage      : fraction of real points within ε of a generated point

  Usage:
    python test.py --model ddpm
    python test.py --model all          # run all models, produce comparison table
    python test.py --model flow --gif   # save sampling animation as gif

  Outputs (saved to ./results/<model>/):
    metrics.json          : all metrics per step count
    sampling_steps.gif    : animation of sampling process
    final_samples.png     : final sample scatter plot
    comparison_table.png  : (--model all only) full comparison table
=============================================================================
"""

import os, json, time, argparse, copy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── import all model definitions from train.py ───────────────────────────────
from train import (
    load_dinosaur,
    VAE, DDPMModel, DDPMSchedule, ScoreModel,
    ConsistencyModel, FlowModel, RectifiedFlow,
    MeanFlowModel, DriftingModel,
    TrigSchedule, LinearSchedule,
    SinusoidalEmbed, TimeCondMLP, EMA
)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# =============================================================================
#  1. METRICS
# =============================================================================

def rbf_kernel(X, Y, sigma=0.5):
    """RBF kernel matrix between rows of X and Y."""
    XX = (X ** 2).sum(1, keepdim=True)
    YY = (Y ** 2).sum(1, keepdim=True)
    XY = X @ Y.T
    D2 = XX + YY.T - 2 * XY
    return torch.exp(-D2 / (2 * sigma ** 2))


def mmd(X, Y, sigma=0.5):
    """
    Biased MMD² between samples X and Y.

    Uses the biased V-statistic estimator:
        MMD²(X,Y) = mean(Kxx) + mean(Kyy) - 2·mean(Kxy)

    This equals ||μ̂_p - μ̂_q||²_H in the RKHS and is ALWAYS ≥ 0.

    The unbiased U-statistic (diagonal zeroed) is an alternative but
    can return negative values when the two distributions are similar,
    which is misleading as a quality metric.

    Lower = distributions are more similar.  Zero iff X and Y are identical.
    """
    Kxx = rbf_kernel(X, X, sigma)
    Kyy = rbf_kernel(Y, Y, sigma)
    Kxy = rbf_kernel(X, Y, sigma)
    return (Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()).item()


def sinkhorn_div(X, Y, eps=0.05, n_iter=100):
    """
    Sinkhorn divergence (≈ regularised Wasserstein-1).
    Uses log-domain Sinkhorn for numerical stability.
    Lower = distributions are more similar.
    """
    def cost_matrix(A, B):
        return ((A.unsqueeze(1) - B.unsqueeze(0)) ** 2).sum(-1)

    def sinkhorn_loss(C, reg, n_iter):
        n, m = C.shape
        log_a = torch.zeros(n, device=C.device) - np.log(n)
        log_b = torch.zeros(m, device=C.device) - np.log(m)
        log_u = torch.zeros(n, device=C.device)
        log_v = torch.zeros(m, device=C.device)
        M = -C / reg
        for _ in range(n_iter):
            log_u = log_a - torch.logsumexp(M + log_v.unsqueeze(0), dim=1)
            log_v = log_b - torch.logsumexp(M + log_u.unsqueeze(1), dim=0)
        P = torch.exp(M + log_u.unsqueeze(1) + log_v.unsqueeze(0))
        return (P * C).sum().item()

    Cxy = cost_matrix(X, Y)
    Cxx = cost_matrix(X, X)
    Cyy = cost_matrix(Y, Y)
    s   = sinkhorn_loss(Cxy, eps, n_iter) \
        - 0.5 * sinkhorn_loss(Cxx, eps, n_iter) \
        - 0.5 * sinkhorn_loss(Cyy, eps, n_iter)
    return max(s, 0.0)


def coverage(real, generated, eps=0.15):
    """
    Coverage: fraction of real points that have at least one
    generated point within distance eps.
    Higher = better (we report 1 - coverage so lower = better).
    """
    D = ((real.unsqueeze(1) - generated.unsqueeze(0)) ** 2).sum(-1).sqrt()
    covered = (D.min(dim=1).values < eps).float().mean().item()
    return 1.0 - covered   # lower is better (missed fraction)


def evaluate(real_pts, gen_pts, device):
    """Compute all metrics. Returns dict."""
    R = torch.tensor(real_pts, device=device, dtype=torch.float32)
    G = torch.tensor(gen_pts,  device=device, dtype=torch.float32) \
        if not isinstance(gen_pts, torch.Tensor) else gen_pts.to(device)

    # cap sample sizes for speed
    n = min(1000, R.shape[0], G.shape[0])
    R = R[torch.randperm(R.shape[0])[:n]]
    G = G[torch.randperm(G.shape[0])[:n]]

    return {
        'mmd':      round(mmd(R, G),            6),
        'sinkhorn': round(sinkhorn_div(R, G),   6),
        'coverage': round(coverage(R, G),       6),
    }


# =============================================================================
#  2. LOAD MODEL HELPERS
# =============================================================================

def load_model(model_name, ckpt_dir, device):
    """Load trained model weights and config from checkpoint directory."""
    cfg_path = os.path.join(ckpt_dir, 'config.json')
    mdl_path = os.path.join(ckpt_dir, 'model.pt')

    if not os.path.exists(mdl_path):
        raise FileNotFoundError(
            f"No checkpoint found at {mdl_path}. "
            f"Run: python train.py --model {model_name}"
        )

    with open(cfg_path) as f:
        cfg = json.load(f)

    # resolve ddim → ddpm checkpoint
    load_name = 'ddpm' if model_name == 'ddim' else model_name
    mdl_path  = mdl_path.replace(f'/{model_name}/', f'/{load_name}/')

    if model_name == 'vae':
        m = VAE(latent_dim=2, hidden=256)
    elif model_name in ('ddpm', 'ddim'):
        m = DDPMModel(hidden=256, depth=4)
    elif model_name == 'score':
        m = ScoreModel(hidden=256, depth=4, L=cfg.get('score_levels', 10))
    elif model_name == 'consistency':
        m = ConsistencyModel(hidden=256, depth=4, sigma_data=0.5)
    elif model_name == 'flow':
        m = FlowModel(hidden=256, depth=4)
    elif model_name == 'rectified':
        m = RectifiedFlow(hidden=256, depth=4)
    elif model_name == 'meanflow':
        m = MeanFlowModel(hidden=256, depth=4)
    elif model_name == 'drifting':
        m = DriftingModel(noise_dim=32, hidden=256, depth=4, R_list=(0.02, 0.05, 0.2))
    else:
        raise ValueError(f"Unknown model: {model_name}")

    m.load_state_dict(torch.load(mdl_path, map_location=device))
    m.to(device).eval()
    return m, cfg


# =============================================================================
#  3. SAMPLERS  (each returns list of (step_count, samples) pairs)
# =============================================================================

@torch.no_grad()
def sample_vae(model, n, device, **_):
    """VAE: single decoder pass. Returns [(1, samples)]."""
    samples = model.sample(n, device).cpu().numpy()
    return [(1, samples)]


@torch.no_grad()
def sample_ddpm(model, n, device, cfg, step_counts=None, **_):
    """
    DDPM ancestral sampling. Sweeps over step_counts (subsets of T).
    Each step_count uses uniform stride through {1..T}.
    """
    T    = cfg.get('T_ddpm', 1000)
    sched = DDPMSchedule(T)
    sched.abar  = sched.abar.to(device)
    sched.betas = sched.betas.to(device)
    sched.alphas = sched.alphas.to(device)

    if step_counts is None:
        step_counts = [1, 5, 10, 50, 100, 200, 500, T]

    results = []
    for S in step_counts:
        timesteps = torch.linspace(T, 1, S, dtype=torch.long, device=device)
        x = torch.randn(n, 2, device=device)
        for t_val in timesteps:
            t_batch = t_val.expand(n)
            eps_pred = model(x, t_batch, T=T)
            # DDPM reverse step
            a    = sched.alphas[t_val - 1]
            abar = sched.abar[t_val]
            abar_prev = sched.abar[t_val - 1] if t_val > 1 else torch.tensor(1.0, device=device)
            beta = sched.betas[t_val - 1]
            # posterior mean
            mu   = (1 / a.sqrt()) * (x - beta / (1 - abar).sqrt() * eps_pred)
            if t_val > 1:
                var = (1 - abar_prev) / (1 - abar) * beta
                x   = mu + var.sqrt() * torch.randn_like(x)
            else:
                x = mu
        results.append((S, x.cpu().numpy()))
    return results


@torch.no_grad()
def sample_ddim(model, n, device, cfg, step_counts=None, **_):
    """
    DDIM deterministic sampling. Same network as DDPM.

    Uses timesteps in [T-1, 1] — avoids t=T where ᾱ_T≈0 causes
    division by zero in the x0 prediction formula.
    """
    T     = cfg.get('T_ddpm', 1000)
    sched = DDPMSchedule(T)
    sched.abar = sched.abar.to(device)

    if step_counts is None:
        step_counts = [1, 5, 10, 20, 50, 100, 200, T]

    results = []
    for S in step_counts:
        # linspace from T-1 down to 1 — never touch t=T where abar≈0
        timesteps = torch.linspace(T - 1, 1, S, dtype=torch.long, device=device)
        x = torch.randn(n, 2, device=device)
        for i, t_val in enumerate(timesteps):
            t_batch  = t_val.expand(n)
            eps_pred = model(x, t_batch, T=T)
            abar_t   = sched.abar[t_val].clamp(min=1e-8)
            abar_prev = sched.abar[timesteps[i + 1]] if i + 1 < len(timesteps) \
                        else torch.tensor(1.0, device=device)
            # predicted x0, clamped for numerical safety
            x0_pred = (x - (1 - abar_t).sqrt() * eps_pred) / abar_t.sqrt()
            x0_pred = x0_pred.clamp(-10, 10)
            # DDIM update (σ=0, fully deterministic)
            x = abar_prev.sqrt() * x0_pred + (1 - abar_prev).sqrt() * eps_pred
        results.append((S, x.cpu().numpy()))
    return results


@torch.no_grad()
def sample_score(model, n, device, step_counts=None, **_):
    """Annealed Langevin sampling. Sweeps total step budget."""
    if step_counts is None:
        step_counts = [10, 50, 100, 200, 500, 1000]

    results = []
    L = model.L
    for S in step_counts:
        steps_per_level = max(1, S // L)
        step_size = 0.05
        x = torch.randn(n, 2, device=device)
        for i in reversed(range(L)):
            t_idx = torch.full((n,), i, device=device, dtype=torch.long)
            sigma = model.sigmas[i].item()
            lr    = step_size * (sigma ** 2)
            for _ in range(steps_per_level):
                score = model(x, t_idx)
                noise = torch.randn_like(x)
                x     = x + lr * score + (2 * lr) ** 0.5 * noise
        results.append((S, x.cpu().numpy()))
    return results


@torch.no_grad()
def sample_consistency(model, n, device, step_counts=None, **_):
    """
    Consistency model multistep sampling — Song et al. 2023, Algorithm 1 & 2.

    Delegates entirely to model.sample(steps=S), which implements:

      S=1  (Algorithm 1): z ~ N(0,t_max²I), x = f_θ(z, t_max)
      S>1  (Algorithm 2): repeat S times:
               x = f_θ(x, t_i)               ← denoise to x̂₀
               x = x + √(t_{i+1}²-t_min²)·ε ← re-noise to next level

    Time sequence uses the rho=7 EDM schedule (NOT linspace), which
    concentrates steps in the high-noise regime where they matter most:
      t_i = (t_max^(1/7) + i/S · (t_min^(1/7) − t_max^(1/7)))^7

    Each additional step strictly improves quality because the model
    is trained to map any noise level directly to x̂₀ — more steps
    give it more chances to correct errors from earlier steps.
    """
    if step_counts is None:
        step_counts = [1, 2, 4, 8, 16]

    results = []
    for S in step_counts:
        x = model.sample(n, device, steps=S)
        results.append((S, x.cpu().numpy()))
    return results


@torch.no_grad()
def sample_flow(model, n, device, step_counts=None, **_):
    """Flow matching Euler ODE, t: 1→0."""
    if step_counts is None:
        step_counts = [1, 5, 10, 20, 50, 100, 200]

    results = []
    for S in step_counts:
        x  = torch.randn(n, 2, device=device)
        dt = -1.0 / S
        for i in range(S):
            t_val = 1.0 - i / S
            t     = torch.full((n,), t_val, device=device)
            x     = x + model(x, t) * dt
        results.append((S, x.cpu().numpy()))
    return results


@torch.no_grad()
def sample_rectified(model, n, device, step_counts=None, **_):
    """Rectified flow Euler ODE (same as flow matching after reflow)."""
    if step_counts is None:
        step_counts = [1, 5, 10, 20, 50, 100, 200]

    results = []
    for S in step_counts:
        x  = torch.randn(n, 2, device=device)
        dt = -1.0 / S
        for i in range(S):
            t_val = 1.0 - i / S
            t     = torch.full((n,), t_val, device=device)
            x     = x + model(x, t) * dt
        results.append((S, x.cpu().numpy()))
    return results


@torch.no_grad()
def sample_meanflow(model, n, device, step_counts=None, **_):
    """MeanFlow multistep drift chain."""
    schedule = TrigSchedule()
    if step_counts is None:
        step_counts = [1, 2, 4, 8, 16]

    results = []
    for S in step_counts:
        t_seq = torch.linspace(1.0, 0.0, S + 1, device=device)
        x = torch.randn(n, 2, device=device)
        for i in range(S):
            t_curr = t_seq[i]
            t_next = t_seq[i + 1]
            t_b    = t_curr.expand(n)
            u_bar  = model(x, t_b)
            # x0 estimate
            x0_hat = x - t_curr.item() * u_bar
            if t_next.item() > 0:
                eps = torch.randn_like(x)
                a   = schedule.alpha(t_next.unsqueeze(0)).squeeze()
                b   = schedule.beta(t_next.unsqueeze(0)).squeeze()
                x   = a * x0_hat + b * eps
            else:
                x = x0_hat
        results.append((S, x.cpu().numpy()))
    return results


@torch.no_grad()
@torch.no_grad()
def sample_drifting(model, n, device, step_counts=None, **_):
    """
    Drifting model: one forward pass z → x0.
    Per tinydrift.py, multi-step sampling is NOT supported — the model
    is a direct noise→data mapping trained with the mean-shift drifting
    field, not a score/flow model. Only step_count=1 is meaningful.
    We report step_counts=[1] and include a few runs for variance estimates.
    """
    if step_counts is None:
        step_counts = [1]

    results = []
    for S in step_counts:
        # always one-step regardless of S (S is kept for API consistency)
        samples = model.sample(n, device)
        results.append((S, samples.cpu().numpy()))
    return results


# Map model name → sampler function
SAMPLERS = {
    'vae':         sample_vae,
    'ddpm':        sample_ddpm,
    'ddim':        sample_ddim,
    'score':       sample_score,
    'consistency': sample_consistency,
    'flow':        sample_flow,
    'rectified':   sample_rectified,
    'meanflow':    sample_meanflow,
    'drifting':    sample_drifting,
}


# =============================================================================
#  4. GIF ANIMATION
# =============================================================================

def make_gif(model_name, results, real_pts, save_path, fps=4):
    """
    Animate sampling quality improving with more steps.
    Each frame: scatter of generated samples vs. real dinosaur.

    NOTE: To display the animated GIF in Colab use HTML, not IPython.display.Image:
        from IPython.display import HTML, display
        display(HTML(f'<img src="{gif_path}" style="max-width:700px">'))
    IPython.display.Image shows only the first frame (static).
    """
    from PIL import Image as PILImage

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.patch.set_facecolor('#0f0f0f')
    for ax in axes:
        ax.set_facecolor('#1a1a2e')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')

    # compute limits from real data with a small margin
    pad = 0.3
    xlo = real_pts[:, 0].min() - pad;  xhi = real_pts[:, 0].max() + pad
    ylo = real_pts[:, 1].min() - pad;  yhi = real_pts[:, 1].max() + pad

    # call tight_layout ONCE outside the loop so canvas size stays constant
    fig.tight_layout(pad=1.5)

    frames = []
    for step_count, samples in results:
        for ax in axes:
            ax.cla()
            ax.set_facecolor('#1a1a2e')
            ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_edgecolor('#444')

        # left: generated
        axes[0].scatter(samples[:, 0], samples[:, 1],
                        s=4, alpha=0.5, c='#00d4ff', rasterized=True)
        axes[0].set_title(f'{model_name.upper()}  |  steps = {step_count}',
                          color='white', fontsize=11, pad=8)
        axes[0].set_xlabel('x', color='#aaa'); axes[0].set_ylabel('y', color='#aaa')

        # right: real
        axes[1].scatter(real_pts[:, 0], real_pts[:, 1],
                        s=4, alpha=0.5, c='#ff6b6b', rasterized=True)
        axes[1].set_title('Ground Truth (Dinosaur)', color='white', fontsize=11, pad=8)
        axes[1].set_xlabel('x', color='#aaa')

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        try:
            buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            buf = buf.reshape(h, w, 4)[:, :, :3]   # RGBA → RGB
        except AttributeError:
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(h, w, 3)
        frames.append(PILImage.fromarray(buf))

    plt.close(fig)

    if len(frames) == 0:
        print("  No frames to save — skipping GIF.")
        return

    frames[0].save(
        save_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),   # ms per frame
        loop=0,                     # 0 = loop forever
        optimize=False,             # keep exact RGB colours
    )
    print(f"  GIF saved → {save_path}  ({len(frames)} frames @ {fps} fps)")


# =============================================================================
#  5. PLOTTING HELPERS
# =============================================================================

def plot_metrics(model_name, results_with_metrics, save_dir):
    """Plot MMD and Sinkhorn vs. number of sampling steps."""
    steps   = [r['steps']    for r in results_with_metrics]
    mmds    = [r['mmd']      for r in results_with_metrics]
    sinks   = [r['sinkhorn'] for r in results_with_metrics]
    covs    = [r['coverage'] for r in results_with_metrics]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle(f'{model_name.upper()} — Metrics vs. Sampling Steps',
                 fontsize=13, fontweight='bold')

    for ax, vals, name, color in zip(
        axes,
        [mmds, sinks, covs],
        ['MMD ↓', 'Sinkhorn ↓', 'Coverage miss ↓'],
        ['steelblue', 'coral', 'mediumseagreen']
    ):
        ax.plot(steps, vals, 'o-', color=color, lw=2, ms=6)
        ax.set_xlabel('Sampling steps'); ax.set_ylabel(name)
        ax.set_title(name); ax.grid(True, alpha=0.3)
        if len(steps) > 1:
            ax.set_xscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics_vs_steps.png'), dpi=130)
    plt.close()


def plot_final_samples(model_name, gen_pts, real_pts, save_dir):
    """Side-by-side scatter: generated vs. real."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle(f'{model_name.upper()} — Final Samples vs. Ground Truth',
                 fontsize=12, fontweight='bold')

    # compute limits from real data with a small margin
    pad = 0.3
    xlo = real_pts[:, 0].min() - pad;  xhi = real_pts[:, 0].max() + pad
    ylo = real_pts[:, 1].min() - pad;  yhi = real_pts[:, 1].max() + pad

    for ax, pts, title, color in [
        (ax1, gen_pts,  'Generated',      'steelblue'),
        (ax2, real_pts, 'Ground Truth',   'coral'),
    ]:
        ax.scatter(pts[:, 0], pts[:, 1], s=5, alpha=0.5, c=color)
        ax.set_title(title)
        ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
        ax.set_aspect('equal'); ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'final_samples.png'), dpi=130)
    plt.close()


# =============================================================================
#  6. TRAIN LOSS COMPARISON  (--model all)
# =============================================================================

def plot_train_losses(ckpt_dir, save_path):
    """
    Load train_loss.npy for every available model and plot all curves
    in a single figure with a shared x-axis (epochs) and a log-scale y-axis.

    Layout:
      - One subplot per model (3×3 grid) sharing the y-axis, for easy
        per-curve inspection of convergence shape.
      - One overlay subplot at the bottom showing all curves together
        for direct cross-model comparison.
    """
    models_ordered = [
        'vae', 'ddpm', 'score', 'consistency',
        'flow', 'rectified', 'meanflow', 'drifting'
    ]
    model_labels = {
        'vae':         'VAE',
        'ddpm':        'DDPM',
        'score':       'Score Matching',
        'consistency': 'Consistency (CT)',
        'flow':        'Flow Matching',
        'rectified':   'Rectified Flow',
        'meanflow':    'MeanFlow',
        'drifting':    'Drifting Model',
    }
    # colour per model — consistent with sample plots
    colors = {
        'vae':         '#7F77DD',
        'ddpm':        '#D85A30',
        'score':       '#D4537E',
        'consistency': '#378ADD',
        'flow':        '#1D9E75',
        'rectified':   '#BA7517',
        'meanflow':    '#185FA5',
        'drifting':    '#639922',
    }

    # load all available loss curves
    curves = {}
    for m in models_ordered:
        # ddim shares ddpm checkpoint
        load_name = 'ddpm' if m == 'ddim' else m
        loss_path = os.path.join(ckpt_dir, load_name, 'train_loss.npy')
        if os.path.exists(loss_path):
            curves[m] = np.load(loss_path)

    if not curves:
        print("  No train_loss.npy files found — skipping loss plot.")
        return

    n_models = len(curves)

    # ── Figure: 2 sections ────────────────────────────────────────────────
    # Top: 2-row grid of individual subplots (up to 4 per row)
    # Bottom: single overlay panel
    ncols   = 4
    nrows_g = (n_models + ncols - 1) // ncols   # rows for individual plots
    fig     = plt.figure(figsize=(14, 3.5 * nrows_g + 4))

    # gridspec: individual plots on top, overlay at bottom
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(
        nrows_g + 1, ncols,
        figure=fig,
        hspace=0.55, wspace=0.35,
        height_ratios=[1] * nrows_g + [1.4],
    )

    # ── Individual subplots ───────────────────────────────────────────────
    for idx, (m, losses) in enumerate(curves.items()):
        row, col = divmod(idx, ncols)
        ax = fig.add_subplot(gs[row, col])
        epochs = np.arange(1, len(losses) + 1)
        ax.plot(epochs, losses, color=colors.get(m, '#888780'),
                linewidth=1.2, alpha=0.9)
        # smoothed trend line
        if len(losses) >= 20:
            window = max(1, len(losses) // 20)
            smooth = np.convolve(losses,
                                 np.ones(window) / window, mode='valid')
            ep_s = np.arange(window, len(losses) + 1)
            ax.plot(ep_s, smooth, color=colors.get(m, '#444'),
                    linewidth=2.0, alpha=0.6, linestyle='--')
        ax.set_title(model_labels.get(m, m), fontsize=9, fontweight='500', pad=4)
        ax.set_xlabel('epoch', fontsize=7, labelpad=2)
        ax.set_ylabel('loss',  fontsize=7, labelpad=2)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.spines[['top', 'right']].set_visible(False)
        # log scale if range spans > 1 order of magnitude
        pos_losses = losses[losses > 0] if losses.min() > 0 else losses
        if len(pos_losses) > 0 and (pos_losses.max() / (pos_losses.min() + 1e-9)) > 10:
            ax.set_yscale('log')

    # hide unused subplot slots
    for idx in range(len(curves), nrows_g * ncols):
        row, col = divmod(idx, ncols)
        fig.add_subplot(gs[row, col]).set_visible(False)

    # ── Overlay panel ─────────────────────────────────────────────────────
    ax_all = fig.add_subplot(gs[nrows_g, :])
    max_epochs = max(len(v) for v in curves.values())
    min_val, max_val = float('inf'), float('-inf')

    for m, losses in curves.items():
        epochs = np.arange(1, len(losses) + 1)
        # normalise each curve to [0,1] so curves with different loss scales
        # can be compared on a single axis
        lo, hi = losses.min(), losses.max()
        norm   = (losses - lo) / (hi - lo + 1e-9)
        ax_all.plot(epochs, norm,
                    color=colors.get(m, '#888780'),
                    linewidth=1.5, alpha=0.85,
                    label=model_labels.get(m, m))

    ax_all.set_xlim(0, max_epochs)
    ax_all.set_ylim(-0.05, 1.15)
    ax_all.set_xlabel('epoch', fontsize=9)
    ax_all.set_ylabel('normalised loss  (0=min, 1=max)', fontsize=9)
    ax_all.set_title('All models — normalised training loss overlay', fontsize=10)
    ax_all.legend(fontsize=8, ncol=4, loc='upper right',
                  framealpha=0.8, edgecolor='none')
    ax_all.grid(True, alpha=0.25, linewidth=0.5)
    ax_all.spines[['top', 'right']].set_visible(False)

    fig.suptitle('Training convergence — all models', fontsize=12,
                 fontweight='500', y=1.01)
    plt.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  Train loss comparison → {save_path}")


# =============================================================================
#  7. COMPARISON TABLE  (--model all)
# =============================================================================

def plot_comparison_table(all_results, save_path):
    """
    Final table: rows = models, columns = metrics at best step count
    + training stats.
    """
    models_ordered = [
        'vae', 'ddpm', 'ddim', 'score', 'consistency',
        'flow', 'rectified', 'meanflow', 'drifting'
    ]
    model_labels = {
        'vae':         'VAE',
        'ddpm':        'DDPM',
        'ddim':        'DDIM',
        'score':       'Score Matching',
        'consistency': 'Consistency (CT)',
        'flow':        'Flow Matching',
        'rectified':   'Rectified Flow',
        'meanflow':    'MeanFlow',
        'drifting':    'Drifting Model',
    }
    col_headers = [
        'Model', 'Min Steps\n(good quality)',
        'Best MMD ↓', 'Best Sinkhorn ↓', 'Best Coverage ↓',
        'Train Time\n(s)', 'Sampler\nType'
    ]
    sampler_type = {
        'vae':         'Decoder',
        'ddpm':        'SDE',
        'ddim':        'ODE',
        'score':       'Langevin',
        'consistency': 'Jump',
        'flow':        'ODE',
        'rectified':   'ODE',
        'meanflow':    'Mean vel.',
        'drifting':    'Direct (1-step)',
    }

    rows = []
    for m in models_ordered:
        if m not in all_results:
            continue
        res  = all_results[m]
        mets = res.get('metrics', [])
        if not mets:
            continue
        best_mmd  = min(r['mmd']      for r in mets)
        best_sink = min(r['sinkhorn'] for r in mets)
        best_cov  = min(r['coverage'] for r in mets)
        # min steps to reach mmd < 2x best
        thresh    = best_mmd * 2.0
        min_steps = next((r['steps'] for r in sorted(mets, key=lambda x: x['steps'])
                          if r['mmd'] <= thresh), mets[-1]['steps'])
        train_t   = res.get('train_time', 'N/A')
        rows.append([
            model_labels[m],
            str(min_steps),
            f"{best_mmd:.4f}",
            f"{best_sink:.4f}",
            f"{best_cov:.4f}",
            f"{train_t:.0f}s" if isinstance(train_t, float) else train_t,
            sampler_type[m],
        ])

    fig, ax = plt.subplots(figsize=(15, len(rows) * 0.7 + 2.5))
    ax.axis('off')
    fig.patch.set_facecolor('#fafafa')

    t = ax.table(
        cellText=rows,
        colLabels=col_headers,
        loc='center',
        cellLoc='center',
    )
    t.auto_set_font_size(False)
    t.set_fontsize(10)
    t.scale(1.0, 2.0)

    # style header
    for j in range(len(col_headers)):
        t[0, j].set_facecolor('#2d4059')
        t[0, j].set_text_props(color='white', fontweight='bold')

    # alternating row colors + highlight best values per metric column
    metric_cols = [2, 3, 4]   # Best MMD, Sinkhorn, Coverage
    for col_idx in metric_cols:
        vals = [float(rows[i][col_idx]) for i in range(len(rows))]
        best_idx = int(np.argmin(vals))
        for i, row in enumerate(rows):
            color = '#e8f5e9' if i % 2 == 0 else '#ffffff'
            if i == best_idx and col_idx in metric_cols:
                color = '#c8e6c9'
            t[i + 1, col_idx].set_facecolor(color)
            if i == best_idx:
                t[i + 1, col_idx].set_text_props(fontweight='bold', color='#1b5e20')

        for i in range(len(rows)):
            for j in range(len(col_headers)):
                if j not in metric_cols:
                    t[i + 1, j].set_facecolor('#f5f5f5' if i % 2 == 0 else '#ffffff')

    ax.set_title(
        'Generative Model Comparison — Datasaurus Dinosaur Dataset\n'
        '(all metrics: lower is better;  best per column highlighted in green)',
        fontsize=12, fontweight='bold', pad=20
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Comparison table saved → {save_path}")


def plot_all_models_samples(all_results, real_pts, save_path):
    """3x3 grid of final samples for all 9 models."""
    models_ordered = [
        'vae', 'ddpm', 'ddim', 'score', 'consistency',
        'flow', 'rectified', 'meanflow', 'drifting'
    ]
    labels = {
        'vae':'VAE','ddpm':'DDPM','ddim':'DDIM',
        'score':'Score Matching','consistency':'Consistency (CT)',
        'flow':'Flow Matching','rectified':'Rectified Flow',
        'meanflow':'MeanFlow','drifting':'Drifting Model'
    }
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle('Final Samples — All Models vs. Datasaurus Dinosaur',
                 fontsize=14, fontweight='bold', y=1.01)

    # compute limits from real data with a small margin
    pad = 0.3
    xlo = real_pts[:, 0].min() - pad;  xhi = real_pts[:, 0].max() + pad
    ylo = real_pts[:, 1].min() - pad;  yhi = real_pts[:, 1].max() + pad

    for ax, m in zip(axes.flat, models_ordered):
        ax.set_facecolor('#f0f4f8')
        ax.scatter(real_pts[:, 0], real_pts[:, 1],
                   s=3, alpha=0.3, c='#cc3333', label='Real')
        if m in all_results and all_results[m].get('best_samples') is not None:
            gen = all_results[m]['best_samples']
            ax.scatter(gen[:, 0], gen[:, 1],
                       s=3, alpha=0.5, c='#1a73e8', label='Generated')
        ax.set_title(labels[m], fontsize=10, fontweight='bold')
        ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2)

    # legend on last subplot
    axes.flat[-1].legend(markerscale=3, fontsize=8, loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  All-models grid saved → {save_path}")


# =============================================================================
#  7. MAIN EVALUATION LOOP
# =============================================================================

def evaluate_model(model_name, args, device, real_pts):
    """Full evaluation pipeline for one model. Returns results dict."""
    print(f"\n{'─'*55}")
    print(f"  Evaluating: {model_name.upper()}")
    print(f"{'─'*55}")

    # ── load ────────────────────────────────────────────────────
    ckpt_dir = os.path.join(args.ckpt_dir, model_name
                            if model_name != 'ddim' else 'ddpm')
    try:
        model, cfg = load_model(model_name, ckpt_dir, device)
    except FileNotFoundError as e:
        print(f"  SKIP: {e}")
        return None

    # load training time if saved
    train_time = cfg.get('train_time_s', 'N/A')

    save_dir = os.path.join(args.results_dir, model_name)
    os.makedirs(save_dir, exist_ok=True)

    # ── sample across step counts ────────────────────────────────
    sampler = SAMPLERS[model_name]
    t0 = time.time()

    if model_name == 'vae':
        results = sampler(model, args.n_samples, device)
    elif model_name in ('ddpm', 'ddim'):
        results = sampler(model, args.n_samples, device, cfg=cfg)
    elif model_name == 'score':
        results = sampler(model, args.n_samples, device)
    else:
        results = sampler(model, args.n_samples, device)

    sample_time = time.time() - t0
    print(f"  Sampling time: {sample_time:.2f}s  ({len(results)} step configs)")

    # ── compute metrics ──────────────────────────────────────────
    metrics_list = []
    best_mmd     = float('inf')
    best_samples = None

    for step_count, samples in results:
        # skip NaN samples (e.g. numerical blow-up)
        if samples is None or np.isnan(samples).any():
            print(f"    steps={step_count:6d}  SKIPPED (NaN samples)")
            metrics_list.append({'steps': step_count, 'mmd': float('nan'),
                                 'sinkhorn': float('nan'), 'coverage': float('nan')})
            continue
        m = evaluate(real_pts, samples, device)
        m['steps'] = step_count
        metrics_list.append(m)
        print(f"    steps={step_count:6d}  MMD={m['mmd']:.4f}  "
              f"Sinkhorn={m['sinkhorn']:.4f}  Coverage={m['coverage']:.4f}")
        if not np.isnan(m['mmd']) and m['mmd'] < best_mmd:
            best_mmd     = m['mmd']
            best_samples = samples

    # ── save metrics ─────────────────────────────────────────────
    metrics_path = os.path.join(save_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump({'model': model_name, 'metrics': metrics_list}, f, indent=2)

    # ── plots ────────────────────────────────────────────────────
    plot_metrics(model_name, metrics_list, save_dir)
    if best_samples is not None:
        plot_final_samples(model_name, best_samples, real_pts, save_dir)
    else:
        print(f"  WARNING: no valid samples for {model_name} — skipping final plot.")

    # ── GIF ──────────────────────────────────────────────────────
    if args.gif:
        try:
            from PIL import Image
            gif_path = os.path.join(save_dir, 'sampling_steps.gif')
            make_gif(model_name, results, real_pts, gif_path, fps=args.fps)
        except ImportError:
            print("  PIL not found — skipping GIF. Install with: pip install Pillow")

    return {
        'metrics':      metrics_list,
        'best_samples': best_samples,
        'train_time':   train_time,
        'sample_time':  sample_time,
    }


# =============================================================================
#  8. ARGPARSE + ENTRY POINT
# =============================================================================

def get_args():
    p = argparse.ArgumentParser(
        description='Evaluate generative models on Datasaurus dinosaur'
    )
    p.add_argument('--model', type=str, default='ddpm',
                   choices=['vae','ddpm','ddim','score','consistency',
                            'flow','rectified','meanflow','drifting','all'],
                   help='"all" runs every model and produces comparison table')
    p.add_argument('--ckpt_dir',    type=str, default='./checkpoints',
                   help='Directory containing model checkpoints')
    p.add_argument('--results_dir', type=str, default='./results',
                   help='Directory to save evaluation outputs')
    p.add_argument('--n_samples',   type=int, default=1000,
                   help='Number of points to generate per evaluation')
    p.add_argument('--n_real',      type=int, default=8000,
                   help='Number of real dinosaur points for comparison')
    p.add_argument('--gif',         action='store_true',
                   help='Save sampling animation as GIF (requires Pillow)')
    p.add_argument('--fps',         type=int, default=3,
                   help='Frames per second for GIF')
    p.add_argument('--seed',        type=int, default=42)
    return p.parse_args()


ALL_MODELS = [
    'vae','ddpm','ddim','score','consistency',
    'flow','rectified','meanflow','drifting'
]


def main():
    args   = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*55}")
    print(f"  Datasaurus Generative Model Evaluation")
    print(f"  Device  : {device}")
    print(f"  Model(s): {args.model}")
    print(f"{'='*55}")

    # load real data
    real_pts = load_dinosaur(n_samples=args.n_real)

    os.makedirs(args.results_dir, exist_ok=True)

    models_to_run = ALL_MODELS if args.model == 'all' else [args.model]
    all_results   = {}

    for m in models_to_run:
        res = evaluate_model(m, args, device, real_pts)
        if res is not None:
            all_results[m] = res

    # ── comparison table (always if ≥2 models evaluated) ────────
    if len(all_results) >= 2:
        table_path = os.path.join(args.results_dir, 'comparison_table.png')
        grid_path  = os.path.join(args.results_dir, 'all_models_samples.png')
        loss_path  = os.path.join(args.results_dir, 'train_losses.png')
        plot_comparison_table(all_results, table_path)
        plot_all_models_samples(all_results, real_pts, grid_path)
        plot_train_losses(args.ckpt_dir, loss_path)

    # ── print summary ────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("  SUMMARY — Best MMD per model")
    print(f"{'─'*55}")
    for m, res in all_results.items():
        best = min(r['mmd'] for r in res['metrics'])
        best_s = min(res['metrics'], key=lambda x: x['mmd'])['steps']
        print(f"  {m:<15}  MMD={best:.4f}  (at {best_s} steps)")
    print(f"{'='*55}\n")


if __name__ == '__main__':
    main()
