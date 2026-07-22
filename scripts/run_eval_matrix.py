"""Multi-scene x multi-motion evaluation matrix.

For each (target, motion) cell: run GSDiff-SPI (train.py, tuned default config),
the INR representation control (train.py scene.type=siren), and the classical /
deep baselines (run_baselines.py). Aggregate into results/eval_matrix/table.json
and a printed comparison table. Every method is scored with the identical
per-frame normalize_01+psnr_fn; baselines/INR are tuned GT-free.

Occam-sized default grid (edit TARGETS/MOTIONS to expand):
    targets : tank, char:5 (digit), char:R (letter), usaf (resolution chart)
    motions : translation, rotation, translation+rotation, accelerated

Usage:
    python scripts/run_eval_matrix.py [--seed 7] [--targets tank char:5] \
        [--motions trans transrot] [--methods gsdiff inr baselines]
"""
import argparse, copy, json, os, subprocess, sys, time
import numpy as np
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)      # for the inline recinr(a) import
PY = sys.executable

TARGETS = {"tank": "assets/tank.png", "digit5": "char:5", "letterR": "char:R",
           "usaf": "assets/usaf_tar_small.png"}
MOTIONS = {
    "trans":    dict(gt_velocity=[8, 8], gt_omega=0.0, enable_rotation=False),
    "rot":      dict(gt_velocity=[0, 0], gt_omega=0.4, enable_rotation=True),
    "transrot": dict(gt_velocity=[8, 8], gt_omega=0.3, enable_rotation=True),
    "accel":    dict(gt_velocity=[6, 6], gt_omega=0.2, enable_rotation=True,
                     gt_accel=[3, 3], gt_beta=0.1, poly_degree=2),
}


def set_dotted(d, k, v):
    for p in k.split(".")[:-1]:
        d = d.setdefault(p, {})
    d[k.split(".")[-1]] = v


def make_cfg(base, tgt_shape, motion, seed, out_dir, scene_type="gaussian"):
    cfg = copy.deepcopy(base)
    cfg["seed"] = seed
    cfg["data"]["shape"] = tgt_shape
    cfg["data"]["num_patterns"] = 2560          # above the K collapse floor
    for k in ("gt_velocity", "gt_omega", "gt_accel", "gt_beta"):
        cfg["data"][k] = motion.get(k, None if k in ("gt_accel", "gt_beta") else cfg["data"].get(k))
    cfg["motion"]["enable_rotation"] = motion.get("enable_rotation", True)
    cfg["motion"]["poly_degree"] = motion.get("poly_degree", 1)
    if scene_type == "siren":
        # STRICT GT-free-selected INR config (scripts/retune_inr_gtfree.py, selects
        # on the held-out per-frame residual, never PSNR): ω₀=8 is the most robust
        # (ω₀≥12 collapses on translation-only; SIREN's default 30 injects grid-noise
        # that also breaks joint motion convergence). Random init (a DGI pre-fit
        # freezes the field at a motion-blur local min that breaks motion).
        cfg["scene"] = {"type": "siren", "siren_w0": 8.0, "init_mode": "random"}
        cfg["solver"]["type"] = "sgd"; cfg["solver"]["sgd_steps"] = 4000
        cfg["solver"]["lr_scene"] = 3e-3
    elif scene_type == "recinr_se2":
        # ReCINR canonical + rigid SE(2). STRICT GT-free-selected canonical grid=20
        # (retune_inr_gtfree.py; grid=32 collapses on translation-only) + motion
        # warmup so the coarse field cannot absorb the motion.
        cfg["scene"] = {"type": "recinr_se2", "init_mode": "random",
                        "recinr_grid_size": 20}
        cfg["solver"]["type"] = "sgd"; cfg["solver"]["sgd_steps"] = 3000
        cfg["solver"]["lr_scene"] = 3e-3; cfg["solver"]["motion_warmup"] = 500
    cfg["output_dir"] = out_dir
    return cfg


def run_train(cfg, tag):
    d = os.path.join(REPO, "results", "eval_matrix", tag)
    if os.path.isfile(os.path.join(d, "results.json")):
        return json.load(open(os.path.join(d, "results.json"), encoding="utf-8"))
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "_cfg.yaml")
    yaml.dump(cfg, open(p, "w", encoding="utf-8"))
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    r = subprocess.run([PY, "train.py", "--config", p], cwd=REPO, env=env,
                       capture_output=True, text=True, encoding="utf-8")
    rp = os.path.join(d, "results.json")
    if not os.path.isfile(rp):
        print(f"    TRAIN FAILED {tag}: {(r.stderr or '')[-500:]}")
        return None
    return json.load(open(rp, encoding="utf-8"))


def run_baselines(cfg, tag):
    d = os.path.join(REPO, "results", "eval_matrix", tag)
    bp = os.path.join(d, "baselines.json")
    if os.path.isfile(bp):
        return json.load(open(bp, encoding="utf-8"))
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "_cfg_bl.yaml")
    yaml.dump(cfg, open(p, "w", encoding="utf-8"))
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    subprocess.run([PY, "scripts/run_baselines.py", "--config", p,
                    "--name", f"eval_matrix/{tag}"], cwd=REPO, env=env,
                   capture_output=True, text=True, encoding="utf-8")
    return json.load(open(bp, encoding="utf-8")) if os.path.isfile(bp) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--targets", nargs="+", default=list(TARGETS))
    ap.add_argument("--motions", nargs="+", default=list(MOTIONS))
    ap.add_argument("--methods", nargs="+", default=["gsdiff", "baselines", "inr"])
    args = ap.parse_args()

    base = yaml.safe_load(open(os.path.join(REPO, "configs/default.yaml"), encoding="utf-8"))
    table, t0 = {}, time.time()
    for tname in args.targets:
        for mname in args.motions:
            cell = f"{tname}_{mname}_s{args.seed}"
            print(f"\n=== {cell} ===", flush=True)
            row = {}
            if "gsdiff" in args.methods:
                r = run_train(make_cfg(base, TARGETS[tname], MOTIONS[mname], args.seed,
                                       f"./results/eval_matrix/{cell}"), cell)
                if r: row["gsdiff"] = r["mean_psnr"]; print(f"  gsdiff   {r['mean_psnr']:.2f}", flush=True)
            if "inr" in args.methods:
                r = run_train(make_cfg(base, TARGETS[tname], MOTIONS[mname], args.seed,
                                       f"./results/eval_matrix/{cell}_inr", scene_type="siren"),
                              f"{cell}_inr")
                if r: row["inr_siren"] = r["mean_psnr"]; print(f"  inr      {r['mean_psnr']:.2f}", flush=True)
            if "recinr_se2" in args.methods:
                r = run_train(make_cfg(base, TARGETS[tname], MOTIONS[mname], args.seed,
                                       f"./results/eval_matrix/{cell}_rcse2", scene_type="recinr_se2"),
                              f"{cell}_rcse2")
                if r: row["recinr_se2"] = r["mean_psnr"]; print(f"  recinr_se2 {r['mean_psnr']:.2f}", flush=True)
            if "recinr" in args.methods:
                import json as _json
                rp = os.path.join(REPO, "results", "eval_matrix", cell, "recinr.json")
                if os.path.isfile(rp):
                    row["recinr"] = _json.load(open(rp))["mean_psnr"]
                else:
                    from gsdiff.data import generate_spi_data
                    from gsdiff.baselines.recinr import recinr_baseline
                    cfg = make_cfg(base, TARGETS[tname], MOTIONS[mname], args.seed, "x")
                    dd = cfg["data"]
                    data = generate_spi_data(
                        H=dd["image_size"][0], W=dd["image_size"][1], T=dd["num_frames"],
                        K=dd["num_patterns"], pattern_type=dd.get("pattern_type", "random"),
                        motion_type=dd.get("motion_type"), snr_db=dd.get("snr_db"),
                        speed_factor=dd.get("speed_factor", 1.0),
                        motion_mode=dd.get("motion_mode", 2),
                        seed=cfg["seed"], shape=dd["shape"], gt_velocity=dd.get("gt_velocity"),
                        gt_omega=dd.get("gt_omega"), gt_accel=dd.get("gt_accel"),
                        gt_beta=dd.get("gt_beta"), noise_sigma_abs=dd.get("noise_sigma_abs"),
                        pattern_order=dd.get("pattern_order", "sequential"), holdout_extra=250)
                    data.seed = cfg["seed"]
                    rec, info = recinr_baseline(data, device="cuda", seed=cfg["seed"])
                    os.makedirs(os.path.dirname(rp), exist_ok=True)
                    _json.dump({"mean_psnr": info["mean_psnr"], "holdout": info["holdout"]},
                               open(rp, "w"))
                    row["recinr"] = info["mean_psnr"]
                print(f"  recinr   {row['recinr']:.2f}", flush=True)
            if "baselines" in args.methods:
                b = run_baselines(make_cfg(base, TARGETS[tname], MOTIONS[mname], args.seed,
                                           f"./results/eval_matrix/{cell}"), cell)
                if b:
                    for k, v in b["baselines"].items():
                        row[k] = v["mean_psnr"]
            table[cell] = row

    out = os.path.join(REPO, "results", "eval_matrix", "table.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"table": table, "elapsed": time.time() - t0}, open(out, "w"), indent=2)
    # print the comparison table
    methods = ["dgi", "static_cs", "perframe_cs", "monin", "gidc3dtv",
               "recinr", "inr_siren", "recinr_se2", "gsdiff"]
    print("\n\n=== EVAL MATRIX (mean PSNR dB) ===")
    print(f"{'cell':22s} " + " ".join(f"{m:>11s}" for m in methods))
    for cell, row in table.items():
        print(f"{cell:22s} " + " ".join(f"{row.get(m, float('nan')):11.2f}" for m in methods))
    print(f"\n({time.time()-t0:.0f}s total) → {out}")


if __name__ == "__main__":
    main()
