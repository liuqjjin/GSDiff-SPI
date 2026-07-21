"""Run the classical / deep baselines on one (scene, motion) config and write a
comparison JSON. GSDiff-SPI itself and the INR representation control run through
train.py (scene.type=gaussian|siren|…) so they reuse the full solver; this script
covers the motion-free / no-shared-model baselines that share the linear infra.

All baselines are scored with the SAME per-frame normalize_01+psnr_fn as GSDiff
and tuned GT-free (held-out measurement residual, never PSNR).

Usage:
    python scripts/run_baselines.py --config configs/default.yaml --name eval/tank_transrot \
        --baselines dgi static_cs perframe_cs monin gidc3dtv
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from gsdiff.data import generate_spi_data
from gsdiff.baselines import cs, monin, gidc, recinr, tv3d
from gsdiff.baselines.common import dgi_image, evaluate_video

ALL = ["dgi", "static_cs", "perframe_cs", "tv3d", "monin", "gidc3dtv", "recinr"]


def _gen(cfg):
    dd = cfg["data"]
    return generate_spi_data(
        H=dd["image_size"][0], W=dd["image_size"][1], T=dd["num_frames"],
        K=dd["num_patterns"], pattern_type=dd.get("pattern_type", "random"),
        motion_type=dd.get("motion_type", "custom_se2"),
        speed_factor=dd.get("speed_factor", 1.0), snr_db=dd.get("snr_db", 25),
        seed=cfg.get("seed", 42), shape=dd["shape"], motion_mode=dd.get("motion_mode", 2),
        gt_velocity=dd.get("gt_velocity"), gt_omega=dd.get("gt_omega"),
        gt_accel=dd.get("gt_accel"), gt_beta=dd.get("gt_beta"),
        noise_sigma_abs=dd.get("noise_sigma_abs"),
        pattern_order=dd.get("pattern_order", "sequential"),
        holdout_extra=int(dd.get("holdout_extra", 250) or 250))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--baselines", nargs="+", default=ALL)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    data = _gen(cfg)
    dev = args.device
    out_dir = os.path.join(REPO, "results", args.name)
    os.makedirs(out_dir, exist_ok=True)
    rows = {}

    def record(name, recon, info=None):
        psnrs, mean_p = evaluate_video(data.gt_frames, recon)
        row = {"mean_psnr": mean_p, "per_frame_psnr": psnrs}
        if info:
            for k in ("velocity_error", "est_velocity", "lambda", "xi", "xi_xy", "xi_t", "note"):
                if k in info:
                    row[k] = info[k]
        rows[name] = row
        print(f"  {name:14s} {mean_p:6.2f} dB"
              + (f"  v_err={[round(v,2) for v in info['velocity_error']]}"
                 if info and "velocity_error" in info else ""), flush=True)

    t0 = time.time()
    if "dgi" in args.baselines:
        dgi = dgi_image(data.patterns, data.measurements).cpu().numpy()
        record("dgi", np.repeat(dgi[None], data.T, 0))
    if "static_cs" in args.baselines:
        recon, info = cs.static_tvcs(data, device=dev); record("static_cs", recon, info)
    if "perframe_cs" in args.baselines:
        recon, info = cs.perframe_tvcs(data, device=dev); record("perframe_cs", recon, info)
    if "tv3d" in args.baselines:
        recon, info = tv3d.tv3d(data, device=dev); record("tv3d", recon, info)
    if "monin" in args.baselines:
        recon, info = monin.monin(data, device=dev); record("monin", recon, info)
    if "gidc3dtv" in args.baselines:
        recon, info = gidc.dynamic_gidc3dtv(data, device=dev, n_steps=2000)
        record("gidc3dtv", recon, info)
    if "recinr" in args.baselines:
        recon, info = recinr.recinr_baseline(data, device=dev, seed=cfg.get("seed", 42))
        record("recinr", recon, info)

    # MERGE with an existing baselines.json so re-running a subset of baselines
    # (e.g. a fixed one) updates only those rows and keeps the rest.
    bpath = os.path.join(out_dir, "baselines.json")
    merged = {}
    if os.path.isfile(bpath):
        try:
            merged = json.load(open(bpath, encoding="utf-8")).get("baselines", {})
        except Exception:
            merged = {}
    merged.update(rows)
    summary = {"name": args.name, "config": args.config, "seed": cfg.get("seed"),
               "shape": cfg["data"]["shape"], "gt_velocity": cfg["data"].get("gt_velocity"),
               "gt_omega": cfg["data"].get("gt_omega"), "snr_db": cfg["data"].get("snr_db"),
               "num_patterns": cfg["data"].get("num_patterns"),
               "elapsed": time.time() - t0, "baselines": merged}
    with open(bpath, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"→ {out_dir}/baselines.json  ({summary['elapsed']:.0f}s)")


if __name__ == "__main__":
    main()
