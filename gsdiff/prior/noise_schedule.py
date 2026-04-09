"""Continuous noise schedule for diffusion prior (EDM-style log-linear).

sigma(t) = exp((1-t)*log(sigma_min) + t*log(sigma_max)),  t in [0,1]
t=0 -> sigma_min (clean),  t=1 -> sigma_max (max noise).
"""
import math
import torch


class NoiseSchedule:

    def __init__(self, sigma_min: float = 0.002, sigma_max: float = 0.5):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self._log_min = math.log(sigma_min)
        self._log_max = math.log(sigma_max)

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """t in [0,1] -> sigma.  Accepts any-shape tensor."""
        log_s = (1.0 - t) * self._log_min + t * self._log_max
        return torch.exp(log_s)

    def t_from_sigma(self, sigma: float) -> float:
        """sigma -> t in [0,1].  Scalar only (for proximal lookup)."""
        sigma = max(self.sigma_min, min(sigma, self.sigma_max))
        return (math.log(sigma) - self._log_min) / (self._log_max - self._log_min)
