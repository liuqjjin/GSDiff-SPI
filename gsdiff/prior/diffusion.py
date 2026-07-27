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


def log_annealed_sigma(index: int, count: int,
                       sigma_start: float, sigma_end: float) -> float:
    """Return one point on an endpoint-inclusive log-linear schedule."""
    if count < 1:
        raise ValueError("count must be >= 1")
    if not 0 <= index < count:
        raise IndexError(f"index {index} outside [0, {count})")
    if sigma_start <= 0 or sigma_end <= 0:
        raise ValueError("sigma_start and sigma_end must be positive")
    if count == 1:
        return float(sigma_end)
    if index == 0:
        return float(sigma_start)
    if index == count - 1:
        return float(sigma_end)
    t = index / (count - 1)
    return math.exp(
        (1.0 - t) * math.log(sigma_start) + t * math.log(sigma_end)
    )


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
        self._last_sigma = None
        # None until set_n_steps() is called (= num_outer - n_warmup). Left unset,
        # σ would collapse to sigma_end after a single z-step — proximal() raises
        # instead of silently degrading to a weak-TV-lookalike (see CLAUDE.md).
        self._n_steps = None
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
        if n < 1:
            raise ValueError("n must be >= 1")
        self._n_steps = n
        self._call_count = 0
        self._last_sigma = None

    @property
    def last_sigma(self) -> float | None:
        """Sigma consumed by the most recent proximal call, if any."""
        return self._last_sigma

    def _current_sigma(self) -> float:
        """Log-linear σ annealing: sigma_start → sigma_end over n_steps calls."""
        return log_annealed_sigma(
            self._call_count, self._n_steps, self.sigma_start, self.sigma_end
        )

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
        if self._n_steps is None:
            raise RuntimeError(
                "DiffusionPrior.set_n_steps(num_outer - n_warmup) must be called "
                "before the first z-step (σ schedule is otherwise undefined).")

        sigma = self._current_sigma()
        self._last_sigma = sigma
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
    def _ddim_sigma_ladder(self, sigma_start):
        """Build the endpoint-inclusive internal DDIM sigma ladder."""
        if self.ddim_spacing == 'log':
            sigmas = torch.logspace(
                math.log10(sigma_start), math.log10(self.schedule.sigma_min),
                self.denoise_steps + 1, device=self.device)
        else:
            sigmas = torch.linspace(
                sigma_start, self.schedule.sigma_min,
                self.denoise_steps + 1, device=self.device)
        sigmas[0] = sigma_start
        sigmas[-1] = self.schedule.sigma_min
        return sigmas

    def _multistep_ddim(self, z_noisy, sigma_start):
        """DDIM from sigma_start down to sigma_min."""
        sigmas = self._ddim_sigma_ladder(sigma_start)
        z = z_noisy
        z0_pred = z_noisy
        for i in range(self.denoise_steps):
            s_cur  = sigmas[i]
            s_next = sigmas[i + 1]
            eps = self.net(z, s_cur.unsqueeze(0))
            z0_pred = z - s_cur * eps
            z = z0_pred + s_next * eps
        return z0_pred
