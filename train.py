
"""GSDiff-SPI  Training.

Usage:
    python train.py                          # ADMM (default)
    python train.py --solver sgd             # SGD baseline
    python train.py --config configs/default.yaml
"""
import argparse, json, os, time
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, torch, yaml

from gsdiff.utils import (set_seed, get_device, ensure_dir, normalize_01,
                           psnr_fn, to_native, save_gif, _to_ns)
from gsdiff.scene import GaussianScene2D
from gsdiff.motion import SE2Motion
from gsdiff.forward import SPIForwardModel
from gsdiff.prior import TVPrior
from gsdiff.solver import ADMMSolver, SGDSolver
from gsdiff.data import generate_spi_data, dgi_reconstruct


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--solver", default=None)
    return p.parse_args()


def evaluate(gt_frames, recon_np, label=""):
    """Per-frame PSNR between GT and reconstruction (both [T,H,W] numpy)."""
    T = gt_frames.shape[0]
    psnrs = []
    for t in range(T):
        g = normalize_01(gt_frames[t])
        r = normalize_01(np.clip(recon_np[t], 0, None))
        psnrs.append(psnr_fn(r, g))
    m = float(np.mean(psnrs))
    print(f"  {label:8s} PSNR: {m:.2f} dB  [first={psnrs[0]:.1f}, last={psnrs[-1]:.1f}]")
    return psnrs, m


# ─── Visualization ────────────────────────────────────────────
def save_comparison_figure(gt, dgi_img, recon_np, out_path, title=""):
    """Save GT vs DGI vs Recon for first/mid/last frames."""
    T = gt.shape[0]
    idxs = [0, T // 2, T - 1]
    fig, axes = plt.subplots(3, len(idxs), figsize=(4 * len(idxs), 12))
    for col, t in enumerate(idxs):
        # GT
        axes[0, col].imshow(gt[t], cmap='gray', vmin=0, vmax=1)
        axes[0, col].set_title(f'GT t={t}', fontsize=10); axes[0, col].axis('off')
        # DGI (same for all frames - it's a single image)
        axes[1, col].imshow(normalize_01(dgi_img), cmap='gray', vmin=0, vmax=1)
        axes[1, col].set_title('DGI (motion-blur)', fontsize=10); axes[1, col].axis('off')
        # Recon
        rc = normalize_01(np.clip(recon_np[t], 0, None))
        p = psnr_fn(rc, normalize_01(gt[t]))
        axes[2, col].imshow(rc, cmap='gray', vmin=0, vmax=1)
        axes[2, col].set_title(f'Recon t={t} ({p:.1f}dB)', fontsize=10); axes[2, col].axis('off')
    fig.suptitle(title, fontsize=13)
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()


def save_y_comparison(y_gt, y_pred_np, out_path):
    """Plot GT measurements vs predicted measurements."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    K = len(y_gt)
    axes[0].plot(y_gt, linewidth=0.5, alpha=0.8, label='GT y')
    axes[0].plot(y_pred_np, linewidth=0.5, alpha=0.8, label='Pred y')
    axes[0].legend(); axes[0].set_title('Raw measurements'); axes[0].set_ylabel('y_k')
    # Z-scored comparison
    y_gt_zs = (y_gt - y_gt.mean()) / (y_gt.std() + 1e-8)
    y_pred_zs = (y_pred_np - y_pred_np.mean()) / (y_pred_np.std() + 1e-8)
    axes[1].plot(y_gt_zs, linewidth=0.5, alpha=0.8, label='GT (z-scored)')
    axes[1].plot(y_pred_zs, linewidth=0.5, alpha=0.8, label='Pred (z-scored)')
    axes[1].legend(); axes[1].set_title('Z-scored measurements'); axes[1].set_xlabel('pattern k')
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()


def save_loss_curve(history, out_path, solver_type):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy([h["loss_data"] for h in history], label="data fidelity")
    if "prim_res" in history[0]:
        ax.semilogy([h["prim_res"] for h in history], label="primal residual")
    ax.set_xlabel("iteration"); ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_title(f"{solver_type} convergence")
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()


# ─── Main ─────────────────────────────────────────────────────
def main():
    args = parse_args()
    with open(args.config, encoding='utf-8') as f:
        raw = yaml.safe_load(f)
    cfg = _to_ns(raw)
    if args.solver:
        cfg.solver.type = args.solver

    set_seed(cfg.seed)
    dev = get_device()
    out_dir = cfg.output_dir
    ensure_dir(out_dir)

    with open(os.path.join(out_dir, "config.yaml"), "w", encoding='utf-8') as f:
        yaml.dump(raw, f)

    print(f"Device: {dev}, Solver: {cfg.solver.type}")

    # ── 1. Generate data ──────────────────────────────────────
    H, W = cfg.data.image_size
    T, K = cfg.data.num_frames, cfg.data.num_patterns

    data = generate_spi_data(
        H=H, W=W, T=T, K=K,
        pattern_type=cfg.data.pattern_type,
        motion_type=cfg.data.motion_type,
        speed_factor=cfg.data.speed_factor,
        snr_db=cfg.data.snr_db,
        seed=cfg.seed,
        shape=cfg.data.shape,
        motion_mode=cfg.data.motion_mode,
        gt_velocity=list(cfg.data.gt_velocity) if hasattr(cfg.data, 'gt_velocity') else None,
        gt_omega=cfg.data.gt_omega if hasattr(cfg.data, 'gt_omega') else None,
    )

    # Update T in case motion_mode=1 changed it
    T = data.T

    print(f"Data: {H}x{W}, T={T}, K={K}, motion={data.motion_type}")
    print(f"GT vel={data.gt_velocity}, omega={data.gt_omega:.4f}")
    print(f"Meas range: [{data.measurements.min():.1f}, {data.measurements.max():.1f}]")

    # ── 2. DGI baseline ──────────────────────────────────────
    print("\nDGI reconstruction...")
    dgi_img = dgi_reconstruct(data.patterns, data.measurements)
    dgi_psnr = psnr_fn(normalize_01(dgi_img), normalize_01(data.canonical))
    print(f"  DGI PSNR (vs canonical): {dgi_psnr:.2f} dB  (motion-blurred)")

    # Save DGI
    plt.imsave(os.path.join(out_dir, "dgi_recon.png"),
               normalize_01(dgi_img), cmap='gray')

    # Save GT GIF
    save_gif(data.gt_frames, os.path.join(out_dir, "gt_video.gif"), fps=5)

    # ── 3. Convert to torch ───────────────────────────────────
    pat = torch.tensor(data.patterns, device=dev)
    y = torch.tensor(data.measurements, device=dev)
    fidx = torch.tensor(data.frame_idx, device=dev)
    tg = torch.tensor(data.t_grid, device=dev)

    # ── 4. Build model ────────────────────────────────────────
    scene = GaussianScene2D(cfg.scene.num_gaussians, H, W, cfg.scene.init_scale).to(dev)
    motion = SE2Motion(((H-1)/2.0, (W-1)/2.0), cfg.motion.enable_rotation).to(dev)
    fwd = SPIForwardModel(scene, motion, H, W).to(dev)
    print(f"Params: scene={scene.num_params()}, motion={motion.num_params()}")

    # ── 4.5. Optional: DGI warm-start for scene amplitudes (方案B) ────
    if getattr(cfg.scene, 'init_mode', 'random') == 'dgi':
        scene.init_from_image(dgi_img)

    # ── 5. Solve ──────────────────────────────────────────────
    t0 = time.time()
    history = []

    if cfg.solver.type == "admm":
        prior = TVPrior(max_iter=50)
        solver = ADMMSolver(
            fwd, prior, pat, y, fidx, tg,
            rho=cfg.solver.rho, tv_weight=cfg.solver.tv_weight,
            lr_scene=cfg.solver.lr_scene, lr_motion=cfg.solver.lr_motion,
            n_inner=cfg.solver.num_inner, rho_growth=cfg.solver.rho_growth,
            loss_norm=getattr(cfg.solver, 'loss_norm', 'zscore'),
            device=dev)

        L = cfg.solver.num_outer
        for l in range(1, L + 1):
            info = solver.step()
            history.append(info)
            mp = motion.get_params_dict()
            # Per-pixel TV for readable numbers
            npix = T * H * W
            tv_pp = info['tv'] / npix
            print(f"[{l:3d}/{L}]  data={info['loss_data']:.4f}  "
                  f"prim={info['prim_res']:.6f}  TV/px={tv_pp:.4f}  "
                  f"rho={info['rho']:.3f}  v={[f'{v:.2f}' for v in mp['velocity']]}  "
                  f"ω={mp.get('omega', 0):.4f}")

    elif cfg.solver.type == "sgd":
        solver = SGDSolver(
            fwd, pat, y, fidx, tg,
            tv_weight=cfg.solver.tv_weight,
            lr_scene=cfg.solver.lr_scene, lr_motion=cfg.solver.lr_motion,
            n_steps=cfg.solver.sgd_steps,
            loss_norm=getattr(cfg.solver, 'loss_norm', 'zscore'),
            device=dev)

        N = cfg.solver.sgd_steps
        for i in range(1, N + 1):
            info = solver.step()
            history.append(info)
            if i % 100 == 0 or i == N:
                mp = motion.get_params_dict()
                print(f"[{i:5d}/{N}]  data={info['loss_data']:.4f}  "
                      f"tv={info['tv']:.4f}  v={[f'{v:.2f}' for v in mp['velocity']]}  "
                      f"ω={mp.get('omega', 0):.4f}")
    else:
        raise ValueError(f"Unknown solver: {cfg.solver.type}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")

    # ── 6. Get reconstruction ─────────────────────────────────
    with torch.no_grad():
        recon_video = fwd.render_video(tg)  # [T,1,H,W]
        y_pred, _ = fwd(pat, fidx, tg)      # [K]
    recon_np = recon_video[:, 0].cpu().numpy()  # [T,H,W]
    y_pred_np = y_pred.cpu().numpy()

    # ── 7. Evaluate ───────────────────────────────────────────
    print("\n=== Results ===")
    psnrs, mean_p = evaluate(data.gt_frames, recon_np, cfg.solver.type)

    me = motion.get_params_dict()
    vel_err = [abs(float(me['velocity'][i]) - float(data.gt_velocity[i])) for i in range(2)]
    print(f"  GT vel:  {data.gt_velocity.tolist()}")
    print(f"  Est vel: {[f'{v:.3f}' for v in me['velocity']]}")
    print(f"  Vel err: {[f'{e:.3f}' for e in vel_err]}")
    if 'omega' in me:
        print(f"  GT ω: {data.gt_omega:.4f}, Est ω: {me['omega']:.4f}, "
              f"err: {abs(me['omega'] - data.gt_omega):.4f}")
    print(f"  DGI PSNR: {dgi_psnr:.2f} dB (motion-blurred baseline)")

    # ── 8. Save everything ────────────────────────────────────
    # Comparison figure
    save_comparison_figure(data.gt_frames, dgi_img, recon_np,
                           os.path.join(out_dir, "comparison.png"),
                           f"{cfg.solver.type.upper()} | PSNR={mean_p:.1f}dB")

    # Y-value comparison
    save_y_comparison(data.measurements, y_pred_np,
                      os.path.join(out_dir, "y_comparison.png"))

    # Loss curve
    save_loss_curve(history, os.path.join(out_dir, "loss.png"), cfg.solver.type)

    # Reconstruction GIF
    save_gif(recon_np, os.path.join(out_dir, "recon_video.gif"), fps=5)

    # Individual frames
    frames_dir = os.path.join(out_dir, "frames")
    ensure_dir(frames_dir)
    for t in range(T):
        plt.imsave(os.path.join(frames_dir, f"gt_{t:03d}.png"),
                   data.gt_frames[t], cmap='gray', vmin=0, vmax=1)
        rc = normalize_01(np.clip(recon_np[t], 0, None))
        plt.imsave(os.path.join(frames_dir, f"recon_{t:03d}.png"),
                   rc, cmap='gray', vmin=0, vmax=1)

    # Results JSON
    results = to_native({
        "solver": cfg.solver.type, "elapsed": elapsed,
        "mean_psnr": mean_p, "per_frame_psnr": psnrs,
        "dgi_psnr": dgi_psnr,
        "est_velocity": me['velocity'],
        "gt_velocity": data.gt_velocity.tolist(),
        "velocity_error": vel_err,
        "est_omega": me.get('omega', 0),
        "gt_omega": data.gt_omega,
        "motion_type": data.motion_type,
    })
    with open(os.path.join(out_dir, "results.json"), "w", encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    # Checkpoint
    torch.save({"scene": scene.state_dict(), "motion": motion.state_dict(),
                "config": raw}, os.path.join(out_dir, "checkpoint.pt"))

    print(f"\nAll results saved to: {out_dir}")
    return mean_p


if __name__ == "__main__":
    main()
