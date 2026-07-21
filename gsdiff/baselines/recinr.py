"""ReCINR as the INR baseline for GSDiff-SPI (the user's own tuned INR method).

ReCINR (canonical feature field + low-rank bandlimited warp + renderer, with
BARF-annealed PE, gauge anchor, flow smoothness, scale-invariant sparsity, TV,
and a warm→flow-only→joint curriculum) already measures via a GENERAL pattern
inner product Y = I @ Pᵀ, not a hard-coded S-matrix. So adapting it to GSDiff-SPI
needs only: (i) pass GSDiff's random/bernoulli patterns instead of the cyclic
S-matrix, (ii) map each measurement to its frame time (tau_meas = t_grid[frame_idx],
t_nodes = t_grid, R=1), (iii) warm-start the canonical from DGI instead of S⁻¹.

The representation, priors, and curriculum are ReCINR's, verbatim (recinr_model.py
is a copy of the ReCINR source). This is the fair "the user's own INR method vs
Gaussian splatting" comparison. GT-free selection via held-out measurement residual.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .recinr_model import ReCINR, ReCINRConfig
from .common import dgi_image, evaluate_video, holdout_residual


# ── Variant (b): ReCINR's canonical REPRESENTATION + rigid SE(2) motion ──
class ReCINRCanonicalScene(nn.Module):
    """ReCINR's feature-tensor canonical + renderer MLP, exposed as
    query(x_norm) so it plugs into INRForwardModel's rigid inverse-SE(2) warp
    (gsdiff/baselines/inr.py) instead of ReCINR's flexible low-rank deformation.

    This isolates ReCINR's scene REPRESENTATION from its warp: with only the
    shared 3-DOF SE(2) motion (which cannot deform the scene to absorb motion),
    does ReCINR's canonical recover the moving scene? Same feature tensor and
    renderer architecture as recinr_model.WarpedCanonicalField, softplus output.
    """
    def __init__(self, H, W, C=32, render_layers=3, grid_size=None):
        super().__init__()
        self.H, self.W, self.C = H, W, C
        # grid_size < H makes the canonical coarse (bilinear-upsampled) → smooth,
        # so the rigid SE(2) motion cannot be absorbed by per-pixel flexibility
        # (the spatial analog of the continuous-warp fix that unlocked variant a).
        gh = grid_size or H
        self.gh = gh
        self.features = nn.Parameter(0.1 * torch.randn(1, C, gh, gh))
        rl = [nn.Linear(C, C), nn.ReLU()]
        for _ in range(max(1, render_layers) - 1):
            rl += [nn.Linear(C, C), nn.ReLU()]
        rl.append(nn.Linear(C, 1))
        self.renderer = nn.Sequential(*rl)

    def _feat_grid(self):
        if self.gh == self.H:
            return self.features
        return F.interpolate(self.features, size=(self.H, self.W),
                             mode="bilinear", align_corners=True)

    def query(self, x_norm):
        # x_norm [N,2] (y,x) in [-1,1]; grid_sample wants (x,y)
        g = torch.stack([x_norm[:, 1], x_norm[:, 0]], -1).view(1, 1, -1, 2)
        feat = F.grid_sample(self._feat_grid(), g, mode="bilinear",
                             padding_mode="border", align_corners=True)  # [1,C,1,N]
        feat = feat[0, :, 0, :].t()                                       # [N,C]
        return F.softplus(self.renderer(feat).squeeze(-1))               # [N]

    def prefit(self, target01, x_norm_grid, steps=300, lr=3e-3):
        t = torch.as_tensor(target01.reshape(-1), dtype=torch.float32,
                            device=x_norm_grid.device)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            F.mse_loss(self.query(x_norm_grid), t).backward(); opt.step()


def _tv_l1(x):
    return (torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).sum()
            + torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).sum())


def _paper_cfg(H, W, T, seed, **kw):
    """ReCINR config, RE-TUNED for GSDiff's underdetermined random-pattern regime.

    ReCINR's spinning-S-matrix paper knobs (K=25, flow_scale=1.5, tv_xy=3e-7) do
    NOT transfer: here each frame is underdetermined, so a GT-free coordinate
    descent + K sweep + best-holdout early stopping moved the operating point to
    flow_scale=0.5, tv_xy=1e-5, lam_tv_canon=1e-4 and (crucially) n_nodes≈1.7·T
    render nodes so the 20 measured times interpolate between more warp nodes,
    forcing smooth low-rank motion instead of per-frame overfitting (11→17 dB).
    """
    d = dict(img_H=H, img_W=W, K=T, hidden=32, render_layers=3, out_act="softplus",
             warp_arch="lowrank", warp_order=0, warp_t_harmonics=2, flow_scale=0.5,
             pe_xy=2, pe_t=5, pe_anneal_frac=0.6, anchor_tau=0.5,
             warm_epochs=300, flow_only_epochs=400, epochs=1200, lr0=3e-3, lr1=1e-3,
             lam_flow_t=0.5, lam_flow_xy=0.2, lam_l1=0.05, tv_xy=1e-5,
             lam_tv_canon=1e-4, lam_ttv=0.0, seed=seed)
    d.update(kw)
    return ReCINRConfig(**d)


def recinr_baseline(data, device="cuda", n_nodes=None, interp=False, seed=None, **cfg_kw):
    """Run ReCINR on GSDiff-SPI data. Returns (recon [T,H,W], info).

    n_nodes = number of continuous render KEY NODES during training (ReCINR's K —
    a bias-variance knob decoupled from the T=20 GT frames). Measurements at the
    T frame times interpolate between these nodes; the continuous field is queried
    at the T GT frame times for evaluation. Defaults to T.
    """
    H, W, T = data.H, data.W, data.T
    K_nodes = int(n_nodes) if n_nodes else int(round(1.7 * T))   # tuned: K≈1.7·T
    _seed = int(seed if seed is not None else getattr(data, "seed", 42) or 42)
    cfg = _paper_cfg(H, W, T, seed=_seed, **cfg_kw)
    cfg.K = K_nodes
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)

    # ── data setup: GSDiff patterns + ppf time assignment (R=1, one "cycle") ──
    patterns = torch.as_tensor(data.patterns, dtype=torch.float32,
                               device=device).unsqueeze(1)            # [K,1,H,W]
    t_grid = torch.as_tensor(data.t_grid, dtype=torch.float32, device=device)  # [T]
    fidx = torch.as_tensor(data.frame_idx, dtype=torch.long, device=device)
    Kmeas = patterns.shape[0]
    if interp:
        # ReCINR-native continuous-time sampling: measurement k at t_k=k/(K-1),
        # matching data generated with time_assignment_mode='interpolation'. Gives
        # the continuous warp fine temporal sampling instead of 20 blocked times.
        tau_meas = torch.linspace(0, 1, Kmeas, device=device)        # [K] continuous
    else:
        tau_meas = t_grid[fidx]                                      # [K] each meas' frame time
    t_nodes = torch.linspace(0, 1, K_nodes, device=device)           # [K_nodes] render nodes
    t_eval = t_grid                                                  # [T] GT frame times (eval)
    y = torch.as_tensor(data.measurements, dtype=torch.float32, device=device)
    y_target = ((y - y.mean()) / (y.std(unbiased=False) + 1e-8)).view(1, -1)   # z-score (R=1)

    # GT-free holdout: prefer the non-invasive eval set, else k%10==7
    if getattr(data, "eval_patterns", None) is not None:
        ep = torch.as_tensor(data.eval_patterns, dtype=torch.float32, device=device)
        em, ef = data.eval_measurements, data.eval_frame_idx
        val_pack = (ep, em, ef)
    else:
        val_pack = None
    K = patterns.shape[0]
    idx = torch.arange(K, device=device)
    val_mask = (idx % 10 == 7) if val_pack is None else torch.zeros(K, dtype=torch.bool, device=device)
    train_mask = ~val_mask

    net = ReCINR(cfg).to(device)

    # ── warm start: fit the canonical (at anchor time) to the DGI motion-blur
    #    image (GSDiff analog of ReCINR's S⁻¹ static reconstruction) ──
    dgi = dgi_image(data.patterns, data.measurements).to(device)      # z-scored [H,W]
    tz = ((dgi - dgi.mean()) / (dgi.std() + 1e-8)).view(1, 1, H, W)
    t_anchor = torch.tensor([cfg.anchor_tau], device=device)
    if cfg.warm_epochs > 0:
        opt_w = torch.optim.Adam(net.parameters(), lr=cfg.lr0)
        net.scene.set_pe_progress(0.0)
        for _ in range(cfg.warm_epochs):
            opt_w.zero_grad()
            I, _ = net.scene(t_anchor)
            Iz = (I - I.mean()) / (I.std(unbiased=False) + 1e-8)
            F.mse_loss(Iz, tz).backward(); opt_w.step()

    # ── flow-only (canonical frozen) → joint, with annealed PE + ReCINR priors ──
    flow_ep = int(cfg.flow_only_epochs)
    total_ep = flow_ep + cfg.epochs
    canon_params = [net.scene.features, *net.scene.renderer.parameters()]
    warp_params = [p for p in net.parameters() if not any(p is q for q in canon_params)]
    opt = torch.optim.Adam(warp_params, lr=cfg.lr0)
    sched = None
    anneal_epochs = max(1, int(cfg.pe_anneal_frac * total_ep))
    best_val, best_recon = np.inf, None
    for ep in range(total_ep):
        if ep == flow_ep:
            for p in canon_params:
                p.requires_grad_(True)
            opt = torch.optim.Adam(net.parameters(), lr=cfg.lr0)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=cfg.lr1)
        if ep < flow_ep:
            for p in canon_params:
                p.requires_grad_(False)
        net.scene.set_pe_progress(min(1.0, ep / anneal_epochs))
        opt.zero_grad()
        y_pred, I_key, fl, I_raw = net(patterns=patterns, tau_meas=tau_meas,
                                       t_nodes=t_nodes, n_cycles=1)
        loss = F.mse_loss(y_pred[0, train_mask], y_target[0, train_mask])
        prior = cfg.tv_xy * _tv_l1(I_key)
        if cfg.lam_tv_canon > 0:
            Ic = net.scene.canonical_image()
            Icz = (Ic - Ic.mean()) / (Ic.std(unbiased=False) + 1e-8)
            prior = prior + cfg.lam_tv_canon * _tv_l1(Icz)
        if cfg.lam_l1 > 0:
            Iq = I_raw - torch.quantile(I_raw, 0.02)
            prior = prior + cfg.lam_l1 * (Iq.abs().mean() / (Iq.pow(2).mean().sqrt() + 1e-8))
        loss = loss + prior
        if cfg.lam_flow_t > 0:
            loss = loss + cfg.lam_flow_t * ((fl[1:] - fl[:-1]) ** 2).mean()
        if cfg.lam_flow_xy > 0:
            dx = fl[:, :, 1:] - fl[:, :, :-1]; dy = fl[:, 1:, :] - fl[:, :-1, :]
            loss = loss + cfg.lam_flow_xy * ((dx ** 2).mean() + (dy ** 2).mean())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if sched is not None:
            sched.step()
        # Best-holdout snapshot early stopping (ReCINR's own protocol): ReCINR
        # over-trains on GSDiff's underdetermined data (full budget < reduced
        # budget), so keep the recon at its lowest GT-free holdout residual.
        if ep >= flow_ep and (ep % 50 == 0 or ep == total_ep - 1):
            with torch.no_grad():
                if val_pack is not None:
                    rec = net.get_key_estimates(t_eval)[0][:, 0].cpu().numpy()
                    v = holdout_residual(rec, *val_pack)
                else:
                    v = float(F.mse_loss(y_pred[0, val_mask], y_target[0, val_mask]).item())
                    rec = None
                if v < best_val:
                    best_val = v
                    best_recon = rec if rec is not None else \
                        net.get_key_estimates(t_eval)[0][:, 0].cpu().numpy()

    with torch.no_grad():
        net.scene.set_pe_progress(1.0)
        if best_recon is None:                               # no snapshot taken
            best_recon = net.get_key_estimates(t_eval)[0][:, 0].cpu().numpy()
    recon, holdout = best_recon, best_val if best_val < np.inf else None
    psnrs, mean_p = evaluate_video(data.gt_frames, recon)
    return recon, {"method": "recinr", "mean_psnr": mean_p, "per_frame_psnr": psnrs,
                   "holdout": holdout, "note": "ReCINR INR (vendored, random-pattern forward)"}
