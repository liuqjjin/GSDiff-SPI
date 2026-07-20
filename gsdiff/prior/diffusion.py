"""Video diffusion prior for ADMM z-step (plug-and-play).

Implements the same interface as TVPrior / TVPrior3D:
    proximal(x, weight) -> Tensor      x: [T,1,H,W], returns [T,1,H,W]
    energy(x) -> float

Math (PnP-ADMM / RED framework):
    z-step:  z* = D_sigma(v_hat)   where v_hat = R(theta) + u
    sigma follows an independent log-linear annealing schedule:
        sigma_start -> sigma_end  over the ADMM outer iterations.
    This decouples the denoising strength from rho, avoiding the
    pitfall of sigma = sqrt(tv_weight/rho) which makes sigma too small.

    Tweedie single-step:  z* = v_hat - sigma * eps_theta(v_hat, sigma)
    Multi-step DDIM:      z* via iterative denoising from sigma down to sigma_min
"""
from __future__ import annotations
import math
from typing import List, Optional, Tuple

import torch

from .noise_schedule import NoiseSchedule
from .unet3d import UNet3D


class DiffusionPrior:

    def __init__(self, checkpoint_path: str, device='cpu',
                 denoise_steps: int = 1,
                 clamp_range: Tuple[float, float] = (0.0, 1.0),
                 # UNet architecture (must match the trained checkpoint)
                 in_channels: int = 1, base_channels: int = 32,
                 channel_mults: Optional[List[int]] = None, emb_dim: int = 128,
                 sigma_min: float = 0.002, sigma_max: float = 0.5,
                 # Independent σ annealing schedule (decoupled from ρ)
                 sigma_start: float = 0.3, sigma_end: float = 0.05,
                 # Ablation flags (defaults preserve historical behaviour)
                 renoise: bool = False,          # z = D_σ(v + σ·ε) instead of D_σ(v)
                 ddim_spacing: str = 'linear'):  # 'linear' | 'log' σ ladder in DDIM
        self.schedule = NoiseSchedule(sigma_min=sigma_min, sigma_max=sigma_max)
        self.denoise_steps = denoise_steps
        self.clamp_range = clamp_range
        self.device = device
        # σ annealing: log-linear decay from sigma_start to sigma_end
        self.sigma_start = max(sigma_min, min(sigma_start, sigma_max))
        self.sigma_end = max(sigma_min, min(sigma_end, sigma_max))
        self._call_count = 0
        self._n_steps = 1  # will be set by caller (= num_outer - n_warmup)
        self.renoise = renoise
        assert ddim_spacing in ('linear', 'log')
        self.ddim_spacing = ddim_spacing

        self.net = UNet3D(in_channels=in_channels, base_channels=base_channels,
                          channel_mults=channel_mults, emb_dim=emb_dim)
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        self.net.load_state_dict(state)
        self.net.eval().to(device)

    def set_n_steps(self, n: int):
        """Set total number of z-step calls (= num_outer - n_warmup)."""
        self._n_steps = max(n, 1)
        self._call_count = 0

    def _current_sigma(self) -> float:
        """Log-linear σ annealing: sigma_start → sigma_end over n_steps calls."""
        t = min(self._call_count / self._n_steps, 1.0)
        # log-linear interpolation (same shape as noise schedule)
        log_s = (1.0 - t) * math.log(self.sigma_start) + t * math.log(self.sigma_end)
        return math.exp(log_s)

    # ── proximal: same signature as TVPrior.proximal ───────────
    def proximal(self, x: torch.Tensor, weight: float) -> torch.Tensor:
        """ADMM z-step via diffusion denoising.

        Parameters
        ----------
        x      : [T, 1, H, W]  v_hat = R(theta) + u
        weight : scalar          (kept for interface compat, ignored — σ from schedule)

        Returns
        -------
        z : [T, 1, H, W]  denoised video
        """
        assert x.ndim == 4 and x.shape[1] == 1, \
            f"Expected [T,1,H,W], got {x.shape}"

        sigma = self._current_sigma()
        self._call_count += 1

        with torch.no_grad():
            # [T,1,H,W] -> [1,1,T,H,W]  (batch=1, channel=1)
            v = x.permute(1, 0, 2, 3).unsqueeze(0)
            if self.renoise:
                # Re-noise so the denoiser sees an on-manifold input at level σ
                # (v's deviation from the clean manifold is neither Gaussian
                # nor at level σ — the known failure mode of naive PnP).
                v = v + sigma * torch.randn_like(v)
            sigma_t = torch.tensor([sigma], device=self.device, dtype=torch.float32)

            if self.denoise_steps == 1:
                z = self._tweedie(v, sigma_t, sigma)
            else:
                z = self._multistep_ddim(v, sigma)

            # [1,1,T,H,W] -> [T,1,H,W]
            z = z.squeeze(0).permute(1, 0, 2, 3)
            z = z.clamp(*self.clamp_range)
        return z

    # ── energy: compatible monitoring metric ───────────────────
    def energy(self, x: torch.Tensor) -> float:
        """Return spatial TV energy for monitoring (does not affect optimization)."""
        dy = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().sum()
        dx = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().sum()
        return (dy + dx).item()

    # ── internal: single-step Tweedie ──────────────────────────
    def _tweedie(self, v, sigma_t, sigma):
        """z = v - sigma * eps_theta(v, sigma)."""
        eps_pred = self.net(v, sigma_t)
        return v - sigma * eps_pred

    # ── internal: multi-step DDIM ──────────────────────────────
    def _multistep_ddim(self, z_noisy, sigma_start):
        """DDIM from sigma_start down to sigma_min."""
        if self.ddim_spacing == 'log':
            sigmas = torch.logspace(
                math.log10(sigma_start), math.log10(self.schedule.sigma_min),
                self.denoise_steps + 1, device=self.device)
        else:
            sigmas = torch.linspace(
                sigma_start, self.schedule.sigma_min,
                self.denoise_steps + 1, device=self.device)
        z = z_noisy
        z0_pred = z_noisy
        for i in range(self.denoise_steps):
            s_cur  = sigmas[i]
            s_next = sigmas[i + 1]
            eps = self.net(z, s_cur.unsqueeze(0))
            z0_pred = z - s_cur * eps
            z = z0_pred + s_next * eps
        return z0_pred
