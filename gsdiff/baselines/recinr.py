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
import torch.nn.functional as F

from .recinr_model import ReCINR, ReCINRConfig
from .common import dgi_image, evaluate_video, holdout_residual


def _tv_l1(x):
    return (torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).sum()
            + torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).sum())


def _paper_cfg(H, W, T, seed, **kw):
    """ReCINR's tuned paper operating point (configs/paper_r2.toml), sized to our grid."""
    d = dict(img_H=H, img_W=W, K=T, hidden=32, render_layers=3, out_act="softplus",
             warp_arch="lowrank", warp_order=0, warp_t_harmonics=2, flow_scale=1.5,
             pe_xy=2, pe_t=5, pe_anneal_frac=0.6, anchor_tau=0.5,
             warm_epochs=600, flow_only_epochs=600, epochs=1500, lr0=3e-3, lr1=1e-3,
             lam_flow_t=0.5, lam_flow_xy=0.2, lam_l1=0.05, tv_xy=3e-7,
             lam_tv_canon=1e-5, lam_ttv=0.0, seed=seed)
    d.update(kw)
    return ReCINRConfig(**d)


def recinr_baseline(data, device="cuda", **cfg_kw):
    """Run ReCINR on GSDiff-SPI data. Returns (recon [T,H,W], info)."""
    H, W, T = data.H, data.W, data.T
    cfg = _paper_cfg(H, W, T, seed=int(getattr(data, "seed", 42) or 42), **cfg_kw)
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)

    # ── data setup: GSDiff patterns + ppf time assignment (R=1, one "cycle") ──
    patterns = torch.as_tensor(data.patterns, dtype=torch.float32,
                               device=device).unsqueeze(1)            # [K,1,H,W]
    t_grid = torch.as_tensor(data.t_grid, dtype=torch.float32, device=device)  # [T]
    fidx = torch.as_tensor(data.frame_idx, dtype=torch.long, device=device)
    tau_meas = t_grid[fidx]                                           # [K] each meas' frame time
    t_nodes = t_grid                                                  # [T] render nodes
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

    with torch.no_grad():
        net.scene.set_pe_progress(1.0)
        I_final, _ = net.get_key_estimates(t_nodes)          # [T,1,H,W] z-scored
        recon = I_final[:, 0].cpu().numpy()                  # [T,H,W]
        yv, _, _, _ = net(patterns=patterns, tau_meas=tau_meas, t_nodes=t_nodes, n_cycles=1)
        holdout = (float(F.mse_loss(yv[0, val_mask], y_target[0, val_mask]).item())
                   if val_mask.any() else
                   holdout_residual(recon, *val_pack) if val_pack else None)

    psnrs, mean_p = evaluate_video(data.gt_frames, recon)
    return recon, {"method": "recinr", "mean_psnr": mean_p, "per_frame_psnr": psnrs,
                   "holdout": holdout, "note": "ReCINR INR (vendored, random-pattern forward)"}
