"""
=============================================================================
  DINOSAUR GENERATIVE MODELS — TRAINING
  train.py
=============================================================================
  Trains one of 9 generative models on the Datasaurus dinosaur 2D dataset.

  Models available:
    vae           : Variational Autoencoder
    ddpm          : Denoising Diffusion Probabilistic Model
    ddim          : DDIM (same training as DDPM, different sampling)
    score         : Score Matching (NCSN-style, multi-scale DSM)
    consistency   : Consistency Model (trained from scratch, CT)
    flow          : Flow Matching (linear interpolant)
    rectified     : Rectified Flow (1-RF then 2-RF reflow)
    meanflow      : Mean Flow
    drifting      : Drifting Model (kernel NCE)

  Usage (Colab or local):
    python train.py --model ddpm --epochs 5000 --lr 3e-4

  Outputs (saved to ./checkpoints/<model>/):
    model.pt          : final model weights
    train_loss.npy    : loss curve  [epochs]
    train_loss.png    : loss plot
    config.json       : hyperparameters

  Note: DDIM shares weights with DDPM — train ddpm, test with ddim.
        Rectified flow stage 2 (reflow) auto-runs after stage 1.
=============================================================================
"""

# ── standard imports ──────────────────────────────────────────────────────
import os, json, time, argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ── reproducibility ───────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# =============================================================================
#  1. DATASAURUS DINOSAUR DATA
# =============================================================================

# CSV path — looks next to this script first, then in the working directory
_CSV_CANDIDATES = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasaurus.csv"),
    "datasaurus.csv",
]

def load_dinosaur(n_samples=8000, noise=0.15, rng=None):
    """
    Returns the Datasaurus dinosaur as a 2D point cloud.

    Follows the tiny-rf / dataset.py convention exactly
    (YassineYousfi/tiny-rf, github.com):

        df = pd.read_csv('datasaurus.csv')
        df = df[df.dataset == 'dino']          # 142 canonical points
        x = df["x"] + N(0, 0.15)              # jitter
        y = df["y"] + N(0, 0.15)
        x = (x / 54 - 1) * 4                  # normalise
        y = (y / 48 - 1) * 4

    Normalised range (approx): x ∈ [-2.4, 3.3], y ∈ [-3.8, 4.3]

    Args:
        n_samples : number of points to return (sampled with replacement).
                    Default 8000, matching tiny-rf (16000 for full training run).
                    With 142 source points, each point is resampled ~56× at n=8000.
        noise     : std of Gaussian jitter added to each point (default 0.15,
                    matching tiny-rf)
        rng       : np.random.Generator; if None uses np.random.default_rng(42)

    Returns:
        pts : np.ndarray, shape [n_samples, 2], dtype float32
    """
    import pandas as pd

    if rng is None:
        rng = np.random.default_rng(42)

    # locate datasaurus.csv
    csv_path = None
    for candidate in _CSV_CANDIDATES:
        if os.path.isfile(candidate):
            csv_path = candidate
            break
    if csv_path is None:
        raise FileNotFoundError(
            "datasaurus.csv not found. Place it next to train.py or in the "
            "working directory. Download from: "
            "https://github.com/YassineYousfi/tiny-rf/blob/master/datasaurus.csv"
        )

    df = pd.read_csv(csv_path)
    df = df[df["dataset"] == "dino"]          # 142 canonical points

    # resample to n_samples with replacement
    ix = rng.integers(0, len(df), n_samples)
    x = df["x"].to_numpy()[ix] + rng.normal(size=n_samples) * noise
    y = df["y"].to_numpy()[ix] + rng.normal(size=n_samples) * noise

    # normalise — exactly as in tiny-rf dataset.py
    x = (x / 54.0 - 1.0) * 4.0
    y = (y / 48.0 - 1.0) * 4.0

    pts = np.stack([x, y], axis=1).astype(np.float32)
    return pts   # shape [N, 2]


# =============================================================================
#  2. SHARED NETWORK ARCHITECTURES
# =============================================================================

def make_mlp(in_dim, out_dim, hidden=256, depth=4, act=nn.SiLU):
    """Generic MLP used by most models."""
    layers = [nn.Linear(in_dim, hidden), act()]
    for _ in range(depth - 2):
        layers += [nn.Linear(hidden, hidden), act()]
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


class SinusoidalEmbed(nn.Module):
    """Sinusoidal time embedding — maps scalar t to dim-dimensional vector."""
    def __init__(self, dim=64):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: [B] scalar
        half = self.dim // 2
        freqs = torch.exp(
            -torch.arange(half, device=t.device).float() * np.log(10000) / (half - 1)
        )
        args  = t[:, None].float() * freqs[None]   # [B, half]
        return torch.cat([args.sin(), args.cos()], dim=-1)  # [B, dim]


class TimeCondMLP(nn.Module):
    """
    MLP conditioned on time t.
    Input: x (2D point) + sinusoidal embedding of t.
    Output: vector of out_dim.
    """
    def __init__(self, out_dim=2, t_dim=64, hidden=256, depth=4):
        super().__init__()
        self.embed = SinusoidalEmbed(t_dim)
        self.net   = make_mlp(2 + t_dim, out_dim, hidden, depth)

    def forward(self, x, t):
        # x: [B, 2],  t: [B] scalar (float, in [0,1] or [0,T])
        te = self.embed(t)              # [B, t_dim]
        return self.net(torch.cat([x, te], dim=-1))


# =============================================================================
#  3. INTERPOLANT SCHEDULE  (shared by DDPM, Score, Flow, etc.)
# =============================================================================

class TrigSchedule:
    """
    TrigFlow: α(t)=cos(πt/2), β(t)=sin(πt/2), t∈[0,1]
    α²+β²=1 exactly (variance preserving).
    """
    @staticmethod
    def alpha(t):  return torch.cos(0.5 * np.pi * t)
    @staticmethod
    def beta(t):   return torch.sin(0.5 * np.pi * t)
    @staticmethod
    def alpha_dot(t): return -0.5 * np.pi * torch.sin(0.5 * np.pi * t)
    @staticmethod
    def beta_dot(t):  return  0.5 * np.pi * torch.cos(0.5 * np.pi * t)


class LinearSchedule:
    """Flow Matching: α(t)=1−t, β(t)=t."""
    @staticmethod
    def alpha(t):     return 1 - t
    @staticmethod
    def beta(t):      return t
    @staticmethod
    def alpha_dot(t): return -torch.ones_like(t)
    @staticmethod
    def beta_dot(t):  return  torch.ones_like(t)


class DDPMSchedule:
    """
    DDPM cosine schedule. α(t)=sqrt(ᾱ_t), β(t)=sqrt(1−ᾱ_t).
    t is integer in [0, T].
    """
    def __init__(self, T=1000):
        self.T = T
        s = 0.008
        steps = torch.arange(T + 1, dtype=torch.float32)
        alphas_cumprod = torch.cos(((steps / T + s) / (1 + s)) * np.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        self.abar = alphas_cumprod                        # [T+1]
        self.betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        self.betas = self.betas.clamp(0, 0.999)
        self.alphas = 1 - self.betas

    def alpha(self, t_int):
        return self.abar[t_int].sqrt()

    def beta_coef(self, t_int):
        return (1 - self.abar[t_int]).sqrt()

    def q_sample(self, x0, t_int, eps):
        a = self.alpha(t_int)[:, None]
        b = self.beta_coef(t_int)[:, None]
        return a * x0 + b * eps

    def posterior_variance(self, t_int):
        abar_prev = torch.where(
            t_int > 0,
            self.abar[t_int - 1],
            torch.ones_like(self.abar[t_int])
        )
        return (1 - abar_prev) / (1 - self.abar[t_int]) * self.betas[t_int - 1]


# =============================================================================
#  4. MODEL DEFINITIONS
# =============================================================================

# ── 4.1  VAE ─────────────────────────────────────────────────────────────────

class VAE(nn.Module):
    """
    VAE for 2D data. Encoder → (μ, logσ²) → z → Decoder → x̂.
    Latent dim = 2 for easy visualization.
    """
    def __init__(self, latent_dim=2, hidden=128):
        super().__init__()
        self.encoder = make_mlp(2, latent_dim * 2, hidden, 3)
        self.decoder = make_mlp(latent_dim, 2, hidden, 3)
        self.latent_dim = latent_dim

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=-1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z  = self.reparameterize(mu, logvar)
        xr = self.decode(z)
        return xr, mu, logvar

    def elbo(self, x, beta=1.0):
        xr, mu, logvar = self(x)
        recon = ((x - xr) ** 2).sum(-1).mean()
        kl    = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(-1).mean()
        return recon + beta * kl, recon, kl

    @torch.no_grad()
    def sample(self, n, device):
        z  = torch.randn(n, self.latent_dim, device=device)
        return self.decode(z)


# ── 4.2  DDPM / DDIM (shared network) ────────────────────────────────────────

class DDPMModel(nn.Module):
    """ε-prediction network for DDPM / DDIM."""
    def __init__(self, hidden=256, depth=4):
        super().__init__()
        self.net = TimeCondMLP(out_dim=2, hidden=hidden, depth=depth)

    def forward(self, x_t, t_int, T=1000):
        # normalise t to [0,1] for sinusoidal embed
        t_norm = t_int.float() / T
        return self.net(x_t, t_norm)


# ── 4.3  Score Matching (NCSN-style) ─────────────────────────────────────────

class ScoreModel(nn.Module):
    """
    Score network: s_θ(x,t) ≈ ∇_x log q_t(x).
    Uses DSM at multiple noise levels σ_t.
    """
    def __init__(self, hidden=256, depth=4, sigma_min=0.01, sigma_max=1.0, L=10):
        super().__init__()
        self.net = TimeCondMLP(out_dim=2, hidden=hidden, depth=depth)
        # log-spaced noise levels
        sigmas = torch.exp(
            torch.linspace(np.log(sigma_min), np.log(sigma_max), L)
        )
        self.register_buffer('sigmas', sigmas)
        self.L = L

    def forward(self, x, t_idx):
        t_norm = t_idx.float() / (self.L - 1)
        return self.net(x, t_norm)

    def dsm_loss(self, x0):
        """Denoising score matching loss over all σ levels."""
        B = x0.shape[0]
        # sample random noise level per sample
        t_idx = torch.randint(0, self.L, (B,), device=x0.device)
        sigma = self.sigmas[t_idx][:, None]        # [B,1]

        eps   = torch.randn_like(x0)
        x_t   = x0 + sigma * eps                  # noisy sample

        # target score = -eps/sigma
        target = -eps / sigma

        s_pred = self(x_t, t_idx)
        # weight by σ² (standard DSM weighting)
        loss = (sigma.squeeze() ** 2 * ((s_pred - target) ** 2).sum(-1)).mean()
        return loss

    @torch.no_grad()
    def langevin_sample(self, n, device, steps=200, step_size=0.1):
        """Annealed Langevin sampling from high σ to low σ."""
        x = torch.randn(n, 2, device=device)
        for i in reversed(range(self.L)):
            t_idx = torch.full((n,), i, device=device, dtype=torch.long)
            for _ in range(steps // self.L):
                score = self(x, t_idx)
                sigma = self.sigmas[i].item()
                x = x + step_size * score + (2 * step_size) ** 0.5 * torch.randn_like(x)
        return x


# ── 4.4  Consistency Model (CT — train from scratch) ─────────────────────────

class ConsistencyModel(nn.Module):
    """
    Consistency Model trained from scratch (CT) — Song et al. 2023.

    Key design:
      f_θ(x_t, t) → x_0  for any t on the ODE trajectory.

    Parameterization (EDM-style boundary conditions):
      f = c_skip(t)·x_t + c_out(t)·F_θ(x_t, t)
      c_skip(t) = σ_data² / (t² + σ_data²)     → 1 as t→0 (identity)
      c_out(t)  = t·σ_data / √(t² + σ_data²)   → 0 as t→0 (boundary)

    N schedule (Song et al. 2023, Algorithm 2):
      N increases from s0=2 to s1 over training, making adjacent
      timestep pairs progressively closer — a curriculum of harder tasks.
      N(k) = min(⌊s0 · √(k/K · (s1²-s0²) + s0²) + 0.5⌋, s1)

      This is WHY the loss increases during training: the task gets harder
      as N grows, because t and t_prev move closer together. This is correct
      behaviour — NOT a sign of divergence.

    Loss:
      d(f_θ(x_t, t),  f_θ-(x_{t'}, t'))  pseudo-Huber distance
      where f_θ- is the EMA (stop-gradient) target network.
    """
    def __init__(self, hidden=256, depth=4, sigma_data=0.5,
                 t_min=0.002, t_max=80.0, s0=2, s1=150):
        super().__init__()
        self.net        = TimeCondMLP(out_dim=2, hidden=hidden, depth=depth)
        self.sigma_data = sigma_data
        self.t_min      = t_min
        self.t_max      = t_max
        self.s0         = s0
        self.s1         = s1

    def c_skip(self, t):
        """EDM skip weight: equals 1 at t=t_min (identity boundary)."""
        sd2 = self.sigma_data ** 2
        return sd2 / (t ** 2 + sd2)

    def c_out(self, t):
        """EDM output weight: equals 0 at t=t_min (boundary condition)."""
        return t * self.sigma_data / torch.sqrt(t ** 2 + self.sigma_data ** 2)

    def forward(self, x_t, t):
        # t in [t_min, t_max] — normalise for the embedding
        t_norm = torch.log(t / self.t_min) / np.log(self.t_max / self.t_min)
        cs = self.c_skip(t)[:, None]
        co = self.c_out(t)[:, None]
        return cs * x_t + co * self.net(x_t, t_norm)

    def N_schedule(self, step, total_steps):
        """
        Discretization schedule: N grows from s0 to s1 over training.
        This is the curriculum that makes the loss increase — the model
        faces harder consistency tasks as N increases.
        """
        s0, s1 = self.s0, self.s1
        return min(
            int(np.floor(s0 * np.sqrt(step / total_steps * (s1**2 - s0**2) + s0**2) + 0.5)),
            s1
        )

    def ct_loss(self, x0, step, total_steps, ema_model):
        """
        CT loss with N schedule.

        step / total_steps → current position in training → N → t-gap.
        As N grows the gap shrinks → task gets harder → loss goes up.
        This is expected behaviour, not divergence.
        """
        B   = x0.shape[0]
        N   = self.N_schedule(step, total_steps)

        # sample adjacent timestep pair using current N
        # t_n > t_{n-1}; both in [t_min, t_max] on log scale
        n   = torch.randint(1, N + 1, (B,), device=x0.device)
        t_n = (self.t_min ** (1/7) +
               n / N * (self.t_max ** (1/7) - self.t_min ** (1/7))) ** 7
        t_nm1 = (self.t_min ** (1/7) +
                 (n - 1) / N * (self.t_max ** (1/7) - self.t_min ** (1/7))) ** 7

        eps = torch.randn_like(x0)

        # noisy samples along the same trajectory (same x0, same eps)
        x_tn   = x0 + t_n[:, None]   * eps
        x_tnm1 = x0 + t_nm1[:, None] * eps

        # student prediction at t_n (with gradient)
        f_student = self(x_tn, t_n)

        # EMA target prediction at t_{n-1} (stop-gradient)
        with torch.no_grad():
            f_target = ema_model(x_tnm1, t_nm1)

        # pseudo-Huber: smoother than MSE, less sensitive to outliers
        # c scaled to data range (sigma_data * sqrt(2) ≈ unit scale)
        c    = 0.00054 * self.t_max  # scaled to data range
        diff = f_student - f_target
        loss = (torch.sqrt(diff.pow(2).sum(-1) + c**2) - c).mean()
        return loss

    @torch.no_grad()
    def sample(self, n, device, steps=1):
        """
        Consistency model sampling — matches Song et al. 2023 exactly.

        Algorithm 1 (1-step, S=1):
            z ~ N(0, t_max²·I)
            x = f_θ(z, t_max)

        Algorithm 2 (S-step multistep):
            z ~ N(0, t_1²·I),  x = f_θ(z, t_1)
            for i = 1 … S-1:
                x = x + sqrt(t_{i+1}² − t_min²) · z_i,  z_i ~ N(0,I)
                x = f_θ(x, t_{i+1})

        Time sequence uses the rho=7 (EDM) schedule, NOT linspace.
        This concentrates steps in the high-noise regime where
        they improve quality the most.

            t_i = (t_max^(1/ρ) + i/S · (t_min^(1/ρ) − t_max^(1/ρ)))^ρ

        With linspace the steps cluster near t=0 (near-clean),
        which wastes NFEs and degrades quality.
        """
        rho = 7.0
        # rho-schedule: S+1 points from t_max to t_min
        i_vals = torch.arange(steps + 1, device=device, dtype=torch.float32)
        t_seq  = (self.t_max ** (1 / rho)
                  + i_vals / steps
                  * (self.t_min ** (1 / rho) - self.t_max ** (1 / rho))) ** rho
        # t_seq[0] = t_max, t_seq[-1] = t_min

        # step 1: sample from prior and apply consistency function
        x = torch.randn(n, 2, device=device) * self.t_max
        x = self(x, t_seq[0].expand(n))

        # steps 2..S: re-noise to t_{i+1}, then denoise
        for i in range(1, steps):
            t_next = t_seq[i]
            eps    = torch.randn_like(x)
            # re-noise: add noise up to level t_{i+1} minus the floor t_min
            x = x + torch.sqrt(t_next ** 2 - self.t_min ** 2) * eps
            x = self(x, t_next.expand(n))

        return x




# ── 4.5  Flow Matching ────────────────────────────────────────────────────────

class FlowModel(nn.Module):
    """
    Flow matching with linear interpolant x_t = (1-t)x_0 + t·ε.
    Network predicts velocity v_θ(x_t, t) ≈ ε - x_0.
    """
    def __init__(self, hidden=256, depth=4):
        super().__init__()
        self.net = TimeCondMLP(out_dim=2, hidden=hidden, depth=depth)

    def forward(self, x_t, t):
        return self.net(x_t, t)

    def cfm_loss(self, x0):
        B = x0.shape[0]
        t   = torch.rand(B, device=x0.device)
        eps = torch.randn_like(x0)
        x_t = (1 - t[:, None]) * x0 + t[:, None] * eps
        target = eps - x0                          # constant velocity target
        v_pred = self(x_t, t)
        return ((v_pred - target) ** 2).sum(-1).mean()

    @torch.no_grad()
    def sample(self, n, device, steps=50):
        x = torch.randn(n, 2, device=device)
        dt = -1.0 / steps
        for i in range(steps):
            t_val = 1.0 - i / steps
            t     = torch.full((n,), t_val, device=device)
            v     = self(x, t)
            x     = x + v * dt
        return x


# ── 4.6  Rectified Flow ───────────────────────────────────────────────────────

class RectifiedFlow(nn.Module):
    """
    Stage-1: standard flow matching (random coupling).
    Stage-2 (reflow): retrain on ODE-induced coupled pairs.
    Same architecture as FlowModel.
    """
    def __init__(self, hidden=256, depth=4):
        super().__init__()
        self.net = TimeCondMLP(out_dim=2, hidden=hidden, depth=depth)

    def forward(self, x_t, t):
        return self.net(x_t, t)

    def loss(self, x0, eps=None):
        """Stage-1 or stage-2 loss. eps=None → random coupling."""
        B = x0.shape[0]
        t   = torch.rand(B, device=x0.device)
        if eps is None:
            eps = torch.randn_like(x0)             # random coupling (stage-1)
        x_t    = (1 - t[:, None]) * x0 + t[:, None] * eps
        target = eps - x0
        v_pred = self(x_t, t)
        return ((v_pred - target) ** 2).sum(-1).mean()

    @torch.no_grad()
    def generate_coupled_pairs(self, x0_all, device, steps=100):
        """
        Run ODE from ε → x̂_0 to get stage-2 pairs (x̂_0, ε).
        Returns: (x0_coupled, eps_coupled) both [N, 2]
        """
        N = len(x0_all)
        eps_all = torch.randn(N, 2, device=device)
        x = eps_all.clone()
        dt = -1.0 / steps
        for i in range(steps):
            t_val = 1.0 - i / steps
            t     = torch.full((N,), t_val, device=device)
            v     = self(x, t)
            x     = x + v * dt
        # x is now x̂_0; paired with eps_all
        return x.detach(), eps_all.detach()

    @torch.no_grad()
    def sample(self, n, device, steps=50):
        x = torch.randn(n, 2, device=device)
        dt = -1.0 / steps
        for i in range(steps):
            t_val = 1.0 - i / steps
            t     = torch.full((n,), t_val, device=device)
            x     = x + self(x, t) * dt
        return x


# ── 4.7  MeanFlow ─────────────────────────────────────────────────────────────

class MeanFlowModel(nn.Module):
    """
    MeanFlow: network predicts mean velocity ū_θ(x_t,t).
    Self-consistency: u_t = ū_t + t·∂_t ū_t
    Training: regress (ū_θ + t·∂_t ū_θ) → instantaneous velocity target.
    Jacobian computed via finite differences for efficiency.
    """
    def __init__(self, hidden=256, depth=4):
        super().__init__()
        self.net = TimeCondMLP(out_dim=2, hidden=hidden, depth=depth)

    def forward(self, x_t, t):
        return self.net(x_t, t)

    def mean_flow_loss(self, x0, schedule, h=0.01):
        """
        Loss = || ū_θ(x_t,t) + t·∂_t ū_θ(x_t,t) - target ||²
        ∂_t ū_θ estimated via finite differences (cheaper than autograd).
        """
        B   = x0.shape[0]
        t   = torch.rand(B, device=x0.device).clamp(h, 1.0 - h)
        eps = torch.randn_like(x0)

        # forward process
        a = schedule.alpha(t)[:, None]
        b = schedule.beta(t)[:, None]
        x_t = a * x0 + b * eps

        # instantaneous velocity target: α̇·x0 + β̇·ε
        ad = schedule.alpha_dot(t)[:, None]
        bd = schedule.beta_dot(t)[:, None]
        u_target = ad * x0 + bd * eps

        # mean velocity at t
        u_bar = self(x_t, t)

        # ∂_t ū via centered finite difference
        # Note: x_t changes with t — we perturb t while keeping x_t fixed
        # (stop-gradient approximation: treat x_t as constant)
        with torch.no_grad():
            t_p = (t + h).clamp(0, 1)
            t_m = (t - h).clamp(0, 1)
            du_dt = (self(x_t, t_p) - self(x_t, t_m)) / (2 * h)

        # self-consistency residual
        lhs  = u_bar + t[:, None] * du_dt
        loss = ((lhs - u_target) ** 2).sum(-1).mean()
        return loss

    @torch.no_grad()
    def sample(self, n, device, schedule, t_start=1.0):
        """One-step: x_0 = x_t - t·ū_θ(x_t, t)"""
        x_t = torch.randn(n, 2, device=device)
        t   = torch.full((n,), t_start, device=device)
        u_bar = self(x_t, t)
        return x_t - t_start * u_bar


# ── 4.8  Drifting Model ───────────────────────────────────────────────────────

class DriftingModel(nn.Module):
    """
    Drifting Model — follows Deng et al., "Generative Modeling via Drifting", ICML 2026.
    Reference implementation: Minimal-Drifting-Models / drifting.py.

    A direct z→x₀ generator trained by the mean-shift drifting field V.
    V pushes generated samples toward the data distribution. At equilibrium V→0.

    Training loss (Algorithm 1):
        L = E[ ||gen − stopgrad(gen + V(gen))||² ]
    where V is computed by Algorithm 2 (doubly-normalised affinity matrix).

    Sampling: one forward pass z∼N(0,I) → net(z).  1-NFE, no ODE.
    """

    def __init__(self, noise_dim=32, hidden=256, depth=4, temp=0.05):
        super().__init__()
        self.noise_dim = noise_dim
        self.temp      = temp
        # SELU activation: self-normalising, matches paper architecture
        layers = [nn.Linear(noise_dim, hidden), nn.SELU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SELU()]
        layers.append(nn.Linear(hidden, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)

    @staticmethod
    def compute_drift(x, y_pos, y_neg, temp=0.05):
        """
        Mean-shift drifting field V (Algorithm 2, Deng et al. 2026).

        V(xᵢ) = Σⱼ Σₖ A_pos[i,j] · A_neg[i,k] · (y_pos[j] − y_neg[k])

        where A is a doubly-normalised affinity matrix (geometric mean of
        row-softmax and column-softmax), computed in log-space for stability.

        Args:
            x:     query points = generated samples  [N, D]
            y_pos: positive samples = real data       [N_pos, D]
            y_neg: negative samples = generated (=x)  [N_neg, D]
            temp:  kernel temperature (paper default: 0.05)
        Returns:
            V: drift vectors [N, D]
        """
        N = x.shape[0]

        # pairwise L2 distances in log-kernel space (numerically stable)
        dist_pos = torch.cdist(x, y_pos)              # [N, N_pos]
        dist_neg = torch.cdist(x, y_neg)              # [N, N_neg]

        # mask self-interactions when y_neg is the same set as x
        if N == y_neg.shape[0]:
            dist_neg = dist_neg + torch.eye(N, device=x.device) * 1e6

        # joint logit matrix: -dist/temp for all positives and negatives
        logit = torch.cat([-dist_pos / temp,
                           -dist_neg / temp], dim=1)  # [N, N_pos+N_neg]

        # doubly-normalised affinity: geometric mean of row- and col-softmax
        A_row = logit.softmax(dim=-1)                 # normalise over y
        A_col = logit.softmax(dim=-2)                 # normalise over x
        A = (A_row * A_col).sqrt()                    # [N, N_pos+N_neg]

        N_pos = y_pos.shape[0]
        A_pos = A[:, :N_pos]                          # [N, N_pos]
        A_neg = A[:, N_pos:]                          # [N, N_neg]

        # factorised weights (avoids O(N·N_pos·N_neg) explicit triple sum)
        W_pos = A_pos * A_neg.sum(dim=1, keepdim=True)
        W_neg = A_neg * A_pos.sum(dim=1, keepdim=True)

        return W_pos @ y_pos - W_neg @ y_neg          # [N, D]

    def drifting_loss(self, x_real):
        """
        Algorithm 1: L = mean_i ||(gen_i) − stopgrad(gen_i + V_i)||²
        V is computed under no_grad; gradients flow only through gen.
        """
        z   = torch.randn(x_real.shape[0], self.noise_dim, device=x_real.device)
        gen = self(z)
        with torch.no_grad():
            V      = self.compute_drift(gen, x_real, gen, self.temp)
            target = (gen + V).detach()
        # sum over D per sample, then mean over batch (matches reference)
        return ((gen - target) ** 2).sum(dim=-1).mean()

    @torch.no_grad()
    def sample(self, n, device, **_):
        """One-step generation: z∼N(0,I) → net(z).  No ODE, 1-NFE."""
        z = torch.randn(n, self.noise_dim, device=device)
        return self(z)


# =============================================================================
#  5. EMA HELPER  (for ConsistencyModel target network)
# =============================================================================

class EMA:
    """Exponential moving average of model parameters."""
    def __init__(self, model, decay=0.999):
        import copy
        self.model = copy.deepcopy(model)
        self.model.eval()
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        for p_ema, p in zip(self.model.parameters(), model.parameters()):
            p_ema.data.mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)


# =============================================================================
#  6. TRAINING LOOPS
# =============================================================================

def train_vae(data, args, device, save_dir):
    model = VAE(latent_dim=2, hidden=256).to(device)
    opt   = optim.Adam(model.parameters(), lr=args.lr)
    losses = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0
        for xb, in DataLoader(TensorDataset(data), batch_size=args.batch, shuffle=True):
            xb = xb.to(device)
            loss, recon, kl = model.elbo(xb, beta=args.beta_vae)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        losses.append(total / (len(data) / args.batch))
        if epoch % 500 == 0:
            print(f"  [VAE] epoch {epoch:5d}  loss={losses[-1]:.4f}")

    return model, losses


def train_ddpm(data, args, device, save_dir):
    schedule = DDPMSchedule(T=args.T_ddpm).to(device) \
               if hasattr(DDPMSchedule(args.T_ddpm), 'to') \
               else DDPMSchedule(args.T_ddpm)
    # move buffers manually
    schedule.abar  = schedule.abar.to(device)
    schedule.betas = schedule.betas.to(device)
    schedule.alphas = schedule.alphas.to(device)

    model = DDPMModel(hidden=256, depth=4).to(device)
    opt   = optim.Adam(model.parameters(), lr=args.lr)
    losses = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0
        for xb, in DataLoader(TensorDataset(data), batch_size=args.batch, shuffle=True):
            xb  = xb.to(device)
            B   = xb.shape[0]
            t   = torch.randint(1, args.T_ddpm + 1, (B,), device=device)
            eps = torch.randn_like(xb)
            x_t = schedule.q_sample(xb, t, eps)
            eps_pred = model(x_t, t, T=args.T_ddpm)
            loss = ((eps_pred - eps) ** 2).sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        losses.append(total / (len(data) / args.batch))
        if epoch % 500 == 0:
            print(f"  [DDPM] epoch {epoch:5d}  loss={losses[-1]:.4f}")

    return model, losses, schedule


def train_score(data, args, device, save_dir):
    model = ScoreModel(hidden=256, depth=4, L=args.score_levels).to(device)
    opt   = optim.Adam(model.parameters(), lr=args.lr)
    losses = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0
        for xb, in DataLoader(TensorDataset(data), batch_size=args.batch, shuffle=True):
            xb   = xb.to(device)
            loss = model.dsm_loss(xb)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        losses.append(total / (len(data) / args.batch))
        if epoch % 500 == 0:
            print(f"  [Score] epoch {epoch:5d}  loss={losses[-1]:.4f}")

    return model, losses


def train_consistency(data, args, device, save_dir):
    model = ConsistencyModel(hidden=256, depth=4, sigma_data=0.5).to(device)
    ema   = EMA(model, decay=0.999)
    opt   = optim.Adam(model.parameters(), lr=args.lr)
    losses = []

    # total number of gradient steps — needed for N schedule
    steps_per_epoch = max(1, len(data) // args.batch)
    total_steps     = args.epochs * steps_per_epoch
    global_step     = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0
        for xb, in DataLoader(TensorDataset(data), batch_size=args.batch, shuffle=True):
            xb = xb.to(device)

            loss = model.ct_loss(xb, global_step, total_steps, ema)
            opt.zero_grad(); loss.backward(); opt.step()
            ema.update(model)

            total       += loss.item()
            global_step += 1

        losses.append(total / steps_per_epoch)
        if epoch % 500 == 0:
            N_now = model.N_schedule(global_step, total_steps)
            print(f"  [CM] epoch {epoch:5d}  loss={losses[-1]:.4f}  N={N_now}"
                  f"  (rising loss = harder task, NOT divergence)")

    return model, losses


def train_flow(data, args, device, save_dir):
    model = FlowModel(hidden=256, depth=4).to(device)
    opt   = optim.Adam(model.parameters(), lr=args.lr)
    losses = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0
        for xb, in DataLoader(TensorDataset(data), batch_size=args.batch, shuffle=True):
            xb   = xb.to(device)
            loss = model.cfm_loss(xb)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        losses.append(total / (len(data) / args.batch))
        if epoch % 500 == 0:
            print(f"  [Flow] epoch {epoch:5d}  loss={losses[-1]:.4f}")

    return model, losses


def train_rectified(data, args, device, save_dir):
    """Two-stage: stage-1 (random coupling) then stage-2 (reflow)."""
    model = RectifiedFlow(hidden=256, depth=4).to(device)
    opt   = optim.Adam(model.parameters(), lr=args.lr)
    losses_stage1, losses_stage2 = [], []

    # ── Stage 1 ───────────────────────────────────────────────────
    print("  [Rectified] Stage 1: random coupling")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0
        for xb, in DataLoader(TensorDataset(data), batch_size=args.batch, shuffle=True):
            xb   = xb.to(device)
            loss = model.loss(xb, eps=None)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        losses_stage1.append(total / (len(data) / args.batch))
        if epoch % 500 == 0:
            print(f"    stage1 epoch {epoch:5d}  loss={losses_stage1[-1]:.4f}")

    # ── Generate coupled pairs ─────────────────────────────────────
    print("  [Rectified] Generating reflow pairs...")
    model.eval()
    all_x0 = data.to(device)
    x0_coupled, eps_coupled = model.generate_coupled_pairs(all_x0, device, steps=100)
    coupled_ds = TensorDataset(x0_coupled, eps_coupled)

    # ── Stage 2 reflow ────────────────────────────────────────────
    print("  [Rectified] Stage 2: reflow")
    opt2 = optim.Adam(model.parameters(), lr=args.lr * 0.5)
    for epoch in range(1, args.epochs // 2 + 1):
        model.train()
        total = 0
        for xb, epsb in DataLoader(coupled_ds, batch_size=args.batch, shuffle=True):
            loss = model.loss(xb, eps=epsb)
            opt2.zero_grad(); loss.backward(); opt2.step()
            total += loss.item()
        losses_stage2.append(total / (len(x0_coupled) / args.batch))
        if epoch % 250 == 0:
            print(f"    stage2 epoch {epoch:5d}  loss={losses_stage2[-1]:.4f}")

    losses_all = losses_stage1 + losses_stage2
    return model, losses_all, len(losses_stage1)


def train_meanflow(data, args, device, save_dir):
    schedule = TrigSchedule()
    model    = MeanFlowModel(hidden=256, depth=4).to(device)
    opt      = optim.Adam(model.parameters(), lr=args.lr)
    losses   = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0
        for xb, in DataLoader(TensorDataset(data), batch_size=args.batch, shuffle=True):
            xb   = xb.to(device)
            loss = model.mean_flow_loss(xb, schedule)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        losses.append(total / (len(data) / args.batch))
        if epoch % 500 == 0:
            print(f"  [MeanFlow] epoch {epoch:5d}  loss={losses[-1]:.4f}")

    return model, losses


def train_drifting(data, args, device, save_dir):
    # drifting quality scales with batch size (more negatives = better signal)
    # use at least 2048 as in the reference, fall back to args.batch if larger
    batch = max(args.batch, 2048)
    model = DriftingModel(noise_dim=32, hidden=256, depth=4, temp=0.05).to(device)
    opt   = optim.Adam(model.parameters(), lr=args.lr)
    losses = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0
        for xb, in DataLoader(TensorDataset(data), batch_size=batch, shuffle=True):
            xb   = xb.to(device)
            loss = model.drifting_loss(xb)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        losses.append(total / max(1, len(data) // batch))
        if epoch % 500 == 0:
            print(f"  [Drifting] epoch {epoch:5d}  loss={losses[-1]:.4f}")

    return model, losses


# =============================================================================
#  7. SAVE UTILITIES
# =============================================================================

def save_checkpoint(save_dir, model, losses, config, extra=None):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, 'model.pt'))
    np.save(os.path.join(save_dir, 'train_loss.npy'), np.array(losses))
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    if extra:
        torch.save(extra, os.path.join(save_dir, 'extra.pt'))

    # loss plot
    plt.figure(figsize=(8, 4))
    plt.plot(losses, lw=1.2, color='steelblue')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title(f"Training Loss — {config['model']}")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'train_loss.png'), dpi=120)
    plt.close()
    print(f"  Saved to {save_dir}/")


# =============================================================================
#  8. MAIN
# =============================================================================

def get_args():
    p = argparse.ArgumentParser(description='Train generative models on Datasaurus dinosaur')
    p.add_argument('--model',       type=str,   default='ddpm',
                   choices=['vae','ddpm','ddim','score','consistency',
                            'flow','rectified','meanflow','drifting'])
    p.add_argument('--epochs',      type=int,   default=3000)
    p.add_argument('--lr',          type=float, default=3e-4)
    p.add_argument('--batch',       type=int,   default=512)
    p.add_argument('--n_data',      type=int,   default=16000)
    p.add_argument('--T_ddpm',      type=int,   default=1000,
                   help='Number of DDPM diffusion steps')
    p.add_argument('--score_levels',type=int,   default=10,
                   help='Number of noise levels for score matching')
    p.add_argument('--beta_vae',    type=float, default=1.0,
                   help='Beta weight on KL term in VAE')
    p.add_argument('--save_dir',    type=str,   default='./checkpoints')
    p.add_argument('--seed',        type=int,   default=42)
    return p.parse_args()


def main():
    args   = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # NOTE: ddim uses same training as ddpm — redirect
    train_model = args.model
    if train_model == 'ddim':
        print("DDIM shares weights with DDPM — training as DDPM.")
        train_model = 'ddpm'

    print(f"\n{'='*60}")
    print(f"  Model   : {args.model}")
    print(f"  Epochs  : {args.epochs}")
    print(f"  Device  : {device}")
    print(f"{'='*60}\n")

    # ── load data ────────────────────────────────────────────────
    pts  = load_dinosaur(n_samples=args.n_data)
    data = torch.tensor(pts)

    save_dir = os.path.join(args.save_dir, args.model)
    t0 = time.time()

    config = vars(args)
    config['device'] = str(device)

    # ── dispatch ─────────────────────────────────────────────────
    if train_model == 'vae':
        model, losses = train_vae(data, args, device, save_dir)
        save_checkpoint(save_dir, model, losses, config)

    elif train_model == 'ddpm':
        model, losses, schedule = train_ddpm(data, args, device, save_dir)
        save_checkpoint(save_dir, model, losses, config,
                        extra={'abar': schedule.abar.cpu(),
                               'betas': schedule.betas.cpu()})

    elif train_model == 'score':
        model, losses = train_score(data, args, device, save_dir)
        save_checkpoint(save_dir, model, losses, config)

    elif train_model == 'consistency':
        model, losses = train_consistency(data, args, device, save_dir)
        save_checkpoint(save_dir, model, losses, config)

    elif train_model == 'flow':
        model, losses = train_flow(data, args, device, save_dir)
        save_checkpoint(save_dir, model, losses, config)

    elif train_model == 'rectified':
        model, losses, stage1_len = train_rectified(data, args, device, save_dir)
        config['stage1_epochs'] = stage1_len
        save_checkpoint(save_dir, model, losses, config)

    elif train_model == 'meanflow':
        model, losses = train_meanflow(data, args, device, save_dir)
        save_checkpoint(save_dir, model, losses, config)

    elif train_model == 'drifting':
        model, losses = train_drifting(data, args, device, save_dir)
        save_checkpoint(save_dir, model, losses, config)

    t1 = time.time()
    print(f"\n  Training time : {t1-t0:.1f}s")
    print(f"  Final loss    : {losses[-1]:.6f}")
    print(f"  Checkpoint    : {save_dir}/\n")


if __name__ == '__main__':
    main()


# MODEL  = 'score'      # ← change me
# EPOCHS = 3000        # 3000 is fast; use 5000–8000 for best quality
# LR     = 3e-4
# BATCH  = 512
# N_DATA = 16000       # match tiny-rf: 16k points from 142 boundary pts

# !python train.py --model {MODEL} --epochs {EPOCHS} --lr {LR} --batch {BATCH} --n_data {N_DATA}

# MODELS  = ['vae','ddpm','score','consistency','flow','rectified','meanflow','drifting']
# EPOCHS  = {'vae':3000,'ddpm':3000,'score':3000,'consistency':5000,
#            'flow':3000,'rectified':4000,'meanflow':4000,'drifting':4000}
# timings = {}