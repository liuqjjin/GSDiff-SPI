"""INR + SE(2) representation control (the paper's load-bearing novelty control).

Replaces ONLY the scene representation with a compact canonical field (SIREN /
grid / low-rank), warped by the SAME SE2Motion and trained by the SAME
solver/loss/budget/GT-free selection as the Gaussian model. If Gaussians still
win, the advantage is the representation's inductive bias — not the motion model,
the physics, or the training protocol.

INRForwardModel subclasses SPIForwardModel and overrides ONLY render_video (the
inverse-SE(2) coordinate pullback μ(u,t)=A(t)⁻¹(u−c−d(t))+c); measure/forward are
reused verbatim, so the physics is literally the same code. Every scene exposes
`query(x_norm[N,2]) -> intensity[N] in [0,1]` and `.parameters()`, so the solver
touches it exactly like GaussianScene2D.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..forward.spi import SPIForwardModel
from ..utils import normalize_01


def normalize_pixel_coordinates(coordinates, center, H, W):
    half = coordinates.new_tensor([(H - 1) / 2.0, (W - 1) / 2.0])
    return (coordinates - center) / half


# ── SIREN canonical field ────────────────────────────────────
class _SineLayer(nn.Module):
    def __init__(self, i, o, w0=30.0, first=False):
        super().__init__()
        self.w0, self.lin = w0, nn.Linear(i, o)
        with torch.no_grad():
            b = 1.0 / i if first else math.sqrt(6.0 / i) / w0
            self.lin.weight.uniform_(-b, b)

    def forward(self, x):
        return torch.sin(self.w0 * self.lin(x))


class SIREN(nn.Module):
    """Coordinate-MLP canonical scene. query(x_norm[N,2]) -> [N] in [0,1]."""
    def __init__(self, hidden=128, n_hidden=2, w0=30.0):
        super().__init__()
        layers = [_SineLayer(2, hidden, w0, first=True)]
        for _ in range(n_hidden):
            layers.append(_SineLayer(hidden, hidden, w0))
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, 1)

    def query(self, x_norm):
        return torch.sigmoid(self.head(self.net(x_norm))).squeeze(-1)

    def prefit(self, target01, x_norm_grid, steps=400, lr=3e-3):
        """DGI pre-fit under identity warp (mirror dgi_adaptive): fit query(grid)≈target."""
        t = torch.as_tensor(target01.reshape(-1), dtype=torch.float32,
                            device=x_norm_grid.device)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            loss = F.mse_loss(self.query(x_norm_grid), t)
            loss.backward(); opt.step()


# ── Grid / low-rank canonical (param-matched controls) ───────
class GridCanonical(nn.Module):
    """Learnable canonical image + bilinear sample. ~Hc*Wc params."""
    def __init__(self, Hc=64, Wc=64, padding_mode="border"):
        super().__init__()
        self.Hc, self.Wc = Hc, Wc
        self.grid = nn.Parameter(torch.zeros(1, 1, Hc, Wc))
        self.padding_mode = padding_mode

    def query(self, x_norm):
        # x_norm is (y,x) in [-1,1]; grid_sample wants (x,y)
        g = torch.stack([x_norm[:, 1], x_norm[:, 0]], -1).view(1, 1, -1, 2)
        s = F.grid_sample(self.grid, g, mode="bilinear",
                          padding_mode=self.padding_mode, align_corners=True)
        return torch.sigmoid(s.reshape(-1))

    def prefit(self, target01, x_norm_grid, steps=300, lr=1e-2):
        with torch.no_grad():                       # invert sigmoid to seed the grid
            t = torch.as_tensor(target01, dtype=torch.float32, device=self.grid.device)
            expected = self.Hc * self.Wc
            if t.numel() != expected:
                raise ValueError(
                    f"grid prefit target must contain exactly {expected} values, "
                    f"got {t.numel()}")
            t = t.clamp(1e-3, 1 - 1e-3).reshape(1, 1, self.Hc, self.Wc)
            self.grid.data.copy_(torch.logit(t))


class LowRankCanonical(nn.Module):
    """Separable canonical G = U @ V.T, bilinear-sampled. 2*N*r params."""
    def __init__(self, H=64, W=64, r=32, padding_mode="border"):
        super().__init__()
        self.H, self.W = H, W
        self.U = nn.Parameter(0.01 * torch.randn(H, r))
        self.V = nn.Parameter(0.01 * torch.randn(W, r))
        self.padding_mode = padding_mode

    def query(self, x_norm):
        G = (self.U @ self.V.t()).view(1, 1, self.H, self.W)
        g = torch.stack([x_norm[:, 1], x_norm[:, 0]], -1).view(1, 1, -1, 2)
        s = F.grid_sample(G, g, mode="bilinear",
                          padding_mode=self.padding_mode, align_corners=True)
        return torch.sigmoid(s.reshape(-1))

    def prefit(self, target01, x_norm_grid, steps=300, lr=1e-2):
        H, W = self.H, self.W
        t = torch.as_tensor(np.clip(target01, 1e-3, 1 - 1e-3),
                            dtype=torch.float32, device=self.U.device).reshape(H, W)
        u, s, vh = torch.linalg.svd(torch.logit(t), full_matrices=False)
        r = self.U.shape[1]
        with torch.no_grad():
            self.U.data.copy_((u[:, :r] * s[:r].sqrt()))
            self.V.data.copy_((vh[:r].t() * s[:r].sqrt()))


def build_scene(kind="siren", **kw):
    if kind == "siren":       return SIREN(hidden=kw.get("hidden", 128), w0=kw.get("w0", 30.0))
    if kind == "siren_small": return SIREN(hidden=54, w0=kw.get("w0", 30.0))
    if kind == "grid":
        return GridCanonical(Hc=kw.get("Hc", kw.get("H", 64)),
                             Wc=kw.get("Wc", kw.get("W", 64)))
    if kind == "lowrank":
        return LowRankCanonical(H=kw.get("H", 64), W=kw.get("W", 64),
                                r=kw.get("r", 32))
    if kind == "recinr_se2":  # ReCINR's canonical + renderer, rigid SE(2) motion
        # Tuned: a COARSE canonical with short-side resolution 16 stops the
        # per-pixel feature grid from absorbing motion (full-res → 9 dB,
        # coarse → ~24 dB).
        from .recinr import ReCINRCanonicalScene
        return ReCINRCanonicalScene(kw.get("H", 64), kw.get("W", 64), C=kw.get("C", 32),
                                    grid_size=kw.get("grid_size", 16))
    raise ValueError(f"unknown scene kind: {kind}")


# ── INR forward model (drop-in for SPIForwardModel) ──────────
class INRForwardModel(SPIForwardModel):
    """Same interface as SPIForwardModel; render_video uses the inverse SE(2) warp."""

    def _pixel_grid(self, dev):
        gy, gx = torch.meshgrid(torch.arange(self.H, device=dev, dtype=torch.float32),
                                torch.arange(self.W, device=dev, dtype=torch.float32),
                                indexing="ij")
        return torch.stack([gy.reshape(-1), gx.reshape(-1)], -1)     # [N,2] (y,x)

    def render_video(self, t_grid):
        dev = t_grid.device
        c = self.motion.center                                       # [2]
        A_t = self.motion._A(t_grid)                                 # [T,2,2]
        d_t = self.motion._displacement(t_grid)                     # [T,2]
        A_inv = torch.linalg.inv(A_t)                                # [T,2,2]
        U = self._pixel_grid(dev)                                    # [N,2]
        rel = U.unsqueeze(0) - c - d_t.unsqueeze(1)                  # [T,N,2]
        mu = torch.einsum("tij,tnj->tni", A_inv, rel) + c            # [T,N,2]
        x_norm = normalize_pixel_coordinates(mu, c, self.H, self.W)  # [-1,1], (y,x)
        T = t_grid.shape[0]
        inside = (x_norm.abs() <= 1.0).all(dim=-1).view(T, self.H, self.W)
        queried = self.scene.query(x_norm.reshape(-1, 2)).view(T, self.H, self.W)
        inten = queried * inside.to(queried.dtype)
        return inten.unsqueeze(1)                                    # [T,1,H,W]

    def norm_grid(self, dev):
        """Identity-warp normalized grid for DGI pre-fit."""
        return normalize_pixel_coordinates(
            self._pixel_grid(dev), self.motion.center, self.H, self.W)
