"""
VENDORED VERBATIM from the user's ReCINR project (D:/SPI/ReCINR/src/recinr/model.py),
UNMODIFIED. ReCINR's forward already computes a general pattern inner product
(Y_key = I_flat @ A_flat.t()), so only the measurement PATTERNS and the time
assignment change for GSDiff-SPI — the representation, warp, priors, and renderer
are reused exactly. The GSDiff-SPI adapter (random/bernoulli patterns + ppf frame
assignment + DGI warm start) lives in gsdiff/baselines/recinr.py; this file is a
copy, not an edit of the ReCINR source.

model.py — ReCINR: canonical + warp INR for spinning cyclic-S single-pixel imaging.

Keeps the factorization (canonical feature tensor -> warp -> renderer) and the
bucket forward model, with independently switchable upgrades, each motivated by
theory / literature (see docs/theory and docs/research):

  U1  Annealed positional encoding on the motion-MLP input (BARF-style
      coarse-to-fine): explicit control of the deformation field's spatio-
      temporal bandwidth; opens high frequencies only late in training.
  U2  Canonical-frame gauge fix: the flow is anchored so that
      flow(x, y, tau0) == 0 exactly (pre-tanh subtraction), pinning the
      canonical frame to the scene at tau0 and killing warp<->appearance drift.
  U3  Closed-form warm start: the canonical branch is pre-fitted to the exact
      S^{-1} full-rotation reconstruction (motion-blurred but structurally
      correct) before joint fitting.
  U4  Deformation regularization: temporal smoothness + spatial smoothness
      (elastic-lite, cf. Nerfies) of the flow field.
  U5  Temporal TV on the rendered states.

All upgrades off + warp_arch='baseline' reduces V2 to the baseline model.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ReCINRConfig:
    img_W: int = 61
    img_H: int = 59
    K: int = 25
    hidden: int = 32
    warp_layers: int = 3
    # U1: positional encoding (0 = raw coordinate only)
    pe_xy: int = 2
    pe_t: int = 5
    pe_anneal_frac: float = 0.6      # fraction of epochs over which bands open
    # U2: gauge anchor
    anchor_tau: float | None = 0.5
    # U3: warm start
    warm_epochs: int = 150
    # U9: staged curriculum — after the warm start, train the flow ONLY with the
    #     canonical frozen (the 90k fast canonical parameters otherwise outrun
    #     the warp MLP and collapse into the smooth null-space competitor before
    #     the flow can lock onto the true motion), then joint polish.
    flow_only_epochs: int = 500
    # U4 / U5: regularizer weights
    lam_flow_t: float = 0.5
    lam_flow_xy: float = 0.2      # tuned (autoresearch, held-out)
    lam_flow_mag: float = 0.0        # tiny bias toward minimal motion
    lam_ttv: float = 0.0
    # U6: canonical-image TV — closes the "speckled canonical + micro-warp
    #     flicker" escape route (the data term alone cannot, by the static-
    #     absorption theorem: S is square/invertible so ANY b is perfectly
    #     explained by some static image).
    lam_tv_canon: float = 1e-5      # canonical-image TV: marginal (held-out
    #                                 ablation motion-dependent), kept light
    # U7: Morozov discrepancy servo — multiplicatively adapts the image-prior
    #     weights so the data loss settles at ~discrepancy_target (the z-domain
    #     noise floor), replacing the razor-thin manual lambda window.
    auto_lambda: bool = False
    discrepancy_target: float | None = None   # z-domain noise floor sigma^2/var(b)
    # U8: positivity + scale-invariant sparsity. The dynamic operator's null
    #     space contains smooth "band" components that every smoothness prior
    #     PREFERS over the true compact target; sparsity (mean|I|/rms(I),
    #     ~0.32 for stencil scenes vs ~0.85 for bands) is the discriminator.
    out_act: str = "softplus"        # 'softplus' | 'leaky' (baseline)
    lam_l1: float = 0.05
    # low-rank (bandlimited) continuous warp
    warp_order: int = 0              # 0 = lean {1,x,y,x r^2,y r^2} (Occam optimum)
    warp_t_harmonics: int = 2        # tuned: R=2's record needs less temporal capacity
    # temporal basis of the warp coefficients (P3 capacity ladder):
    #   'fourier' (paper) — raw tau + warp_t_harmonics sin/cos pairs (global support)
    #   'bspline'         — raw tau + spline_knots cardinal cubic B-splines
    #                       (local support; suited to stop-and-go / aperiodic motion)
    warp_t_basis: str = "fourier"
    spline_knots: int = 8
    # A1 amendment (PROJECT_MEMORY §11): BARF-style annealing of the temporal
    # harmonics — harmonics 1..2 open from the start (an effectively-F2
    # landscape while the optimizer finds the motion basin), harmonics 3..th
    # cosine-opened over pe_anneal_frac of training (full capacity late).
    warp_t_anneal: bool = False
    # shared with baseline
    tv_xy: float = 3e-7           # summed-form TV (paper operating point)
    flow_scale: float = 1.5
    warp_arch: str = "lowrank"       # 'lowrank' (paper) | 'pe_relu' | 'baseline'
    epochs: int = 1500            # tuned: 4000 was over-trained
    lr0: float = 3e-3
    lr1: float = 1e-3
    seed: int = 42
    # architecture knobs (searchable)
    render_layers: int = 3        # number of C->C hidden layers in the renderer MLP


class AnnealedPE(nn.Module):
    """Per-dimension Fourier features with BARF-style band annealing.

    Input  [..., D] with per-dim frequency counts L_d.
    Output [..., D + 2 * sum(L_d)]; band l carries weight
    w_l = (1 - cos(pi * clamp(alpha - l, 0, 1))) / 2, alpha in [0, L_d].
    """

    def __init__(self, freqs_per_dim: list[int]):
        super().__init__()
        self.freqs = freqs_per_dim
        self.alpha_frac = 1.0  # 0..1 training progress of annealing
        self.out_dim = len(freqs_per_dim) + 2 * sum(freqs_per_dim)

    def set_progress(self, frac: float) -> None:
        self.alpha_frac = float(min(max(frac, 0.0), 1.0))

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        parts = [v]
        for d, L in enumerate(self.freqs):
            if L == 0:
                continue
            alpha = self.alpha_frac * L
            x = v[..., d:d + 1]
            for l in range(L):
                w = 0.5 * (1.0 - math.cos(math.pi * min(max(alpha - l, 0.0), 1.0)))
                arg = (2.0 ** l) * math.pi * x
                parts += [w * torch.sin(arg), w * torch.cos(arg)]
        return torch.cat(parts, dim=-1)


class _Sine(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class LowRankWarp(nn.Module):
    """Bandlimited continuous deformation field.

    u_phi(x, y, tau) = sum_b c_b(tau) * Phi_b(x, y), where {Phi_b} is a fixed
    low-order spatial polynomial basis and c_b(tau) are continuous temporal
    coefficients predicted by a small MLP on low-harmonic Fourier features of
    tau. This is a genuine coordinate->displacement INR, but the fixed low
    spatial order bandlimits the flow so it can represent the smooth, globally
    coherent deformations of real target motion (translation, rotation, shear,
    swirl are all order <= 3) while it CANNOT synthesize the high-frequency
    pixel-shuffling warps that would let a static image explain an ordered
    single-cycle bucket stream (the static-absorber degeneracy). Time enters
    only here; the canonical field and renderer are shared across tau.
    """

    def __init__(self, n_basis_order: int, t_harmonics: int, hidden: int,
                 t_basis: str = "fourier", spline_knots: int = 8,
                 t_anneal: bool = False, t_anneal_base: int = 2):
        super().__init__()
        self.order = n_basis_order
        self.t_harmonics = t_harmonics
        self.t_basis = t_basis
        self.spline_knots = spline_knots
        self.t_anneal = t_anneal
        self.t_anneal_base = t_anneal_base   # harmonics <= base are always open
        self.alpha_frac = 1.0                # annealing progress in [0, 1]
        self.nb = self._num_basis(n_basis_order)
        if t_basis == "fourier":
            in_t = 1 + 2 * t_harmonics
        elif t_basis == "bspline":
            in_t = 1 + spline_knots
        else:
            raise ValueError(f"unknown warp_t_basis {t_basis!r}")
        self.coeff = nn.Sequential(
            nn.Linear(in_t, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 2 * self.nb),
        )
        nn.init.zeros_(self.coeff[-1].weight)   # start at identity (zero flow)
        nn.init.zeros_(self.coeff[-1].bias)

    @staticmethod
    def _num_basis(order: int) -> int:
        # order 0 = LEAN affine + radial-cubic {1,x,y,x r^2,y r^2} (5): the
        #   basis-ablation optimum — the pure-quadratic terms {x^2,xy,y^2} earn
        #   nothing (order-2 == order-1 on every motion) while the radial-cubic
        #   terms are essential for non-rigid swirl (order-3 33 vs order-1 21 dB).
        # order 1 = affine {1,x,y} (3) — exact for translation/rotation/shear/scale
        # order 2 = + quadratic (6);  order 3 = + radial-cubic (8, full)
        return {0: 5, 1: 3, 2: 6, 3: 8}[order]

    def _basis(self, xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
        r2 = xs * xs + ys * ys
        cols = [torch.ones_like(xs), xs, ys]                # affine
        if self.order == 0:                                 # lean: + radial-cubic
            cols += [xs * r2, ys * r2]
            return torch.stack(cols, dim=-1)
        if self.order >= 2:
            cols += [xs * xs, xs * ys, ys * ys]
        if self.order >= 3:
            cols += [xs * r2, ys * r2]
        return torch.stack(cols, dim=-1)                    # [..., nb]

    def _tfeat(self, t: torch.Tensor) -> torch.Tensor:
        if self.t_basis == "bspline":
            # raw tau + cardinal cubic B-splines on uniform centers over [0,1]
            n = self.spline_knots
            c = torch.linspace(0.0, 1.0, n, dtype=t.dtype, device=t.device)
            h = 1.0 / (n - 1)
            u = (t - c) / h                                   # [..., n]
            a = u.abs()
            b3 = torch.where(
                a < 1.0, (4.0 - 6.0 * a * a + 3.0 * a ** 3) / 6.0,
                torch.where(a < 2.0, (2.0 - a) ** 3 / 6.0, torch.zeros_like(a)))
            return torch.cat([t, b3], dim=-1)
        feats = [t]
        for h in range(1, self.t_harmonics + 1):
            w = 1.0
            if self.t_anneal and h > self.t_anneal_base:
                # BARF window (same form as AnnealedPE): band h opens when the
                # annealing front alpha passes it. alpha sweeps the annealed
                # bands (base, th] as training progresses.
                n_ann = self.t_harmonics - self.t_anneal_base
                alpha = self.alpha_frac * n_ann
                l = h - self.t_anneal_base - 1          # 0-based annealed index
                w = 0.5 * (1.0 - math.cos(math.pi * min(max(alpha - l, 0.0), 1.0)))
            feats += [w * torch.sin(2 * math.pi * h * t),
                      w * torch.cos(2 * math.pi * h * t)]
        return torch.cat(feats, dim=-1)

    def set_progress(self, frac: float) -> None:
        self.alpha_frac = float(min(max(frac, 0.0), 1.0))

    def forward(self, grid: torch.Tensor, t_grid: torch.Tensor) -> torch.Tensor:
        """grid [M,H,W,2], t_grid [M] -> flow [M,H,W,2] (pre-scale)."""
        M, H, W, _ = grid.shape
        Phi = self._basis(grid[..., 0], grid[..., 1])       # [M,H,W,nb]
        c = self.coeff(self._tfeat(t_grid.view(M, 1)))      # [M, 2*nb]
        cx, cy = c[:, :self.nb], c[:, self.nb:]
        dx = torch.einsum("mhwb,mb->mhw", Phi, cx)
        dy = torch.einsum("mhwb,mb->mhw", Phi, cy)
        return torch.stack([dx, dy], dim=-1)                # [M,H,W,2]


def _make_warp_net(cfg: ReCINRConfig) -> tuple[nn.Module, nn.Module | None]:
    """Returns (warp_net, pe). pe is None for the baseline arch."""
    if cfg.warp_arch == "baseline":
        sine = _Sine()
        layers: list[nn.Module] = [nn.Linear(3, cfg.hidden)]
        for _ in range(cfg.warp_layers):
            layers += [nn.Linear(cfg.hidden, cfg.hidden), nn.LayerNorm(cfg.hidden), sine]
        layers.append(nn.Linear(cfg.hidden, 2))
        return nn.Sequential(*layers), None
    if cfg.warp_arch == "pe_relu":
        pe = AnnealedPE([cfg.pe_xy, cfg.pe_xy, cfg.pe_t])
        layers = [nn.Linear(pe.out_dim, cfg.hidden), nn.ReLU()]
        for _ in range(cfg.warp_layers - 1):
            layers += [nn.Linear(cfg.hidden, cfg.hidden), nn.ReLU()]
        last = nn.Linear(cfg.hidden, 2)
        nn.init.zeros_(last.weight)   # flow starts exactly at identity
        nn.init.zeros_(last.bias)
        layers.append(last)
        return nn.Sequential(*layers), pe
    if cfg.warp_arch == "lowrank":
        return LowRankWarp(cfg.warp_order, cfg.warp_t_harmonics, cfg.hidden,
                           t_basis=getattr(cfg, "warp_t_basis", "fourier"),
                           spline_knots=getattr(cfg, "spline_knots", 8),
                           t_anneal=getattr(cfg, "warp_t_anneal", False)), None
    raise ValueError(f"unknown warp_arch {cfg.warp_arch!r}")


class WarpedCanonicalField(nn.Module):
    """Canonical feature tensor + anchored/annealed warp + ReLU renderer."""

    def __init__(self, cfg: ReCINRConfig):
        super().__init__()
        self.cfg = cfg
        H, W, C = cfg.img_H, cfg.img_W, cfg.hidden
        self.features = nn.Parameter(0.1 * torch.randn(1, C, H, W))
        self.flow_scale = nn.Parameter(torch.tensor(float(cfg.flow_scale)))
        self.warp_net, self.pe = _make_warp_net(cfg)
        rl: list[nn.Module] = [nn.Linear(C, C), nn.ReLU()]
        for _ in range(max(1, cfg.render_layers) - 1):
            rl += [nn.Linear(C, C), nn.ReLU()]
        rl.append(nn.Linear(C, 1))
        self.renderer = nn.Sequential(*rl)

    def set_pe_progress(self, frac: float) -> None:
        if self.pe is not None:
            self.pe.set_progress(frac)
        if hasattr(self.warp_net, "set_progress"):   # A1: annealed harmonics
            self.warp_net.set_progress(frac)

    def _out_act(self, o: torch.Tensor) -> torch.Tensor:
        if self.cfg.out_act == "softplus":
            return F.softplus(o)
        return F.leaky_relu(o, 0.001)

    def canonical_image(self) -> torch.Tensor:
        """Render the un-warped canonical frame, [1, 1, H, W]."""
        H, W, C = self.cfg.img_H, self.cfg.img_W, self.cfg.hidden
        out = self.renderer(self.features.permute(0, 2, 3, 1).reshape(H * W, C))
        return self._out_act(out.view(1, 1, H, W))

    def _grid(self, M: int, device) -> torch.Tensor:
        H, W = self.cfg.img_H, self.cfg.img_W
        ys, xs = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing="ij",
        )
        return torch.stack([xs, ys], dim=-1).unsqueeze(0).expand(M, -1, -1, -1)

    def _raw_flow(self, t_grid: torch.Tensor) -> torch.Tensor:
        """Raw warp-net output at all grid points, [M, H, W, 2]."""
        M = t_grid.shape[0]
        H, W = self.cfg.img_H, self.cfg.img_W
        grid = self._grid(M, t_grid.device)
        if self.cfg.warp_arch == "lowrank":
            return self.warp_net(grid, t_grid)
        t = t_grid.view(M, 1, 1, 1).expand(-1, H, W, -1)
        inp = torch.cat([grid, t], dim=-1)
        if self.pe is not None:
            inp = self.pe(inp)
        return self.warp_net(inp.reshape(-1, inp.shape[-1])).view(M, H, W, 2)

    def flow(self, t_grid: torch.Tensor) -> torch.Tensor:
        """Anchored, bounded flow field, [M, H, W, 2]."""
        raw = self._raw_flow(t_grid)
        if self.cfg.anchor_tau is not None:
            t0 = torch.tensor([self.cfg.anchor_tau], device=t_grid.device)
            raw = raw - self._raw_flow(t0)          # flow(tau0) == 0 exactly
        if self.cfg.warp_arch == "lowrank":
            return raw * self.flow_scale            # low-order basis is self-bounding
        return torch.tanh(raw) * self.flow_scale

    def forward(self, t_grid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (frames [M,1,H,W], flow [M,H,W,2])."""
        M = t_grid.shape[0]
        H, W, C = self.cfg.img_H, self.cfg.img_W, self.cfg.hidden
        fl = self.flow(t_grid)
        grid = torch.clamp(self._grid(M, t_grid.device) + fl, -1, 1)
        feats = self.features.expand(M, -1, -1, -1)
        warped = F.grid_sample(feats, grid, mode="bilinear",
                               padding_mode="border", align_corners=True)
        out = self.renderer(warped.permute(0, 2, 3, 1).reshape(M * H * W, C))
        return self._out_act(out.view(M, 1, H, W)), fl


class ReCINR(nn.Module):
    """Bucket forward model identical to the baseline (linear temporal interp)."""

    def __init__(self, cfg: ReCINRConfig):
        super().__init__()
        self.cfg = cfg
        self.scene = WarpedCanonicalField(cfg)

    def get_key_estimates(self, t_nodes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        I, fl = self.scene(t_nodes)
        I = (I - I.mean()) / (I.std(unbiased=False) + 1e-8)
        return I, fl

    def raw_states(self, t_nodes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Un-normalized non-negative states (for the sparsity prior)."""
        return self.scene(t_nodes)

    def forward(self, patterns: torch.Tensor, tau_meas: torch.Tensor,
                t_nodes: torch.Tensor, n_cycles: int = 1):
        M, K = t_nodes.shape[0], patterns.shape[0]
        device = patterns.device
        I_raw, fl = self.scene(t_nodes)
        I_key = (I_raw - I_raw.mean()) / (I_raw.std(unbiased=False) + 1e-8)
        A_flat = patterns[:, 0].reshape(K, -1)
        I_flat = I_key[:, 0].reshape(M, -1)
        Y_key = I_flat @ A_flat.t()
        spos = tau_meas.clamp(0.0, 1.0) * (M - 1)
        m0 = spos.floor().long()
        m1 = (m0 + 1).clamp(max=M - 1)
        alpha = spos - m0.float()
        idx = torch.arange(K, device=device)
        y = (1.0 - alpha) * Y_key[m0, idx] + alpha * Y_key[m1, idx]
        # standardize per reconstruction unit (cycle) — matches the loss model
        if n_cycles > 1 and K % n_cycles == 0:
            seg = y.view(n_cycles, -1)
            seg = (seg - seg.mean(1, keepdim=True)) / (seg.std(1, unbiased=False, keepdim=True) + 1e-8)
            y = seg.reshape(1, K)
        else:
            y = y.view(1, K)
            y = (y - y.mean()) / (y.std(unbiased=False) + 1e-8)
        return y, I_key, fl, I_raw
