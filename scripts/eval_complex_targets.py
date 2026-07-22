"""Task 3 — validate GSDiff's advantage on COMPLEX targets.

Complex 64x64 targets (fine local structure / multi-object) where a compact
many-Gaussian representation should beat a coarse-canonical INR and linear
baselines. For each target x transrot motion we run:
  - gsdiff_diff : GSDiff, diffusion PnP prior (the full method / default config)
  - gsdiff_tv   : GSDiff, 3D-TV prior (isolates the REPRESENTATION advantage,
                  no diffusion domain-gap confound on these new targets)
  - recinr_se2  : ReCINR canonical + rigid SE(2) (INR representation control)
  - baselines   : dgi, tv3d, monin, gidc3dtv, recinr
All scored with the identical per-frame normalize_01+psnr_fn; baselines/INR GT-free.

Usage: python scripts/eval_complex_targets.py [--seed 7] [--targets cx_camera ...]
"""
import argparse, copy, json, os, subprocess, sys, time
import numpy as np, yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
PY = sys.executable

TARGETS = {"cx_camera": "assets/cx_camera.png", "cx_coins": "assets/cx_coins.png",
           "cx_clutter": "assets/cx_clutter.png", "cx_text": "assets/cx_text.png"}
MOTION = dict(gt_velocity=[8, 8], gt_omega=0.3, enable_rotation=True)


def base_cfg(base, shape, seed):
    cfg = copy.deepcopy(base)
    cfg["seed"] = seed
    cfg["data"]["shape"] = shape
    cfg["data"]["num_patterns"] = 2560
    cfg["data"]["holdout_extra"] = 250
    cfg["data"]["gt_velocity"] = MOTION["gt_velocity"]
    cfg["data"]["gt_omega"] = MOTION["gt_omega"]
    cfg["motion"]["enable_rotation"] = True
    cfg["motion"]["poly_degree"] = 1
    return cfg


def cfg_gsdiff(base, shape, seed, prior):
    cfg = base_cfg(base, shape, seed)
    cfg["solver"]["type"] = "admm"
    cfg["solver"]["prior_type"] = prior          # 'diffusion' | 'tv'
    return cfg


def cfg_recinr_se2(base, shape, seed):
    cfg = base_cfg(base, shape, seed)
    cfg["scene"] = {"type": "recinr_se2", "init_mode": "random"}
    cfg["solver"]["type"] = "sgd"; cfg["solver"]["sgd_steps"] = 3000
    cfg["solver"]["lr_scene"] = 3e-3; cfg["solver"]["motion_warmup"] = 500
    return cfg


def run_train(cfg, tag):
    d = os.path.join(REPO, "results", "complex_eval", tag)
    rp = os.path.join(d, "results.json")
    if os.path.isfile(rp):
        return json.load(open(rp, encoding="utf-8"))
    os.makedirs(d, exist_ok=True)
    cfg["output_dir"] = f"./results/complex_eval/{tag}"
    p = os.path.join(d, "_cfg.yaml"); yaml.dump(cfg, open(p, "w", encoding="utf-8"))
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    subprocess.run([PY, "train.py", "--config", p], cwd=REPO, env=env,
                   capture_output=True, text=True, encoding="utf-8", errors="replace")
    return json.load(open(rp, encoding="utf-8")) if os.path.isfile(rp) else None


def run_baselines(cfg, tag):
    d = os.path.join(REPO, "results", "complex_eval", tag)
    bp = os.path.join(d, "baselines.json")
    if os.path.isfile(bp):
        return json.load(open(bp, encoding="utf-8"))
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "_cfg_bl.yaml"); yaml.dump(cfg, open(p, "w", encoding="utf-8"))
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    subprocess.run([PY, "scripts/run_baselines.py", "--config", p, "--name",
                    f"complex_eval/{tag}", "--baselines", "dgi", "tv3d", "monin",
                    "gidc3dtv", "recinr"], cwd=REPO, env=env,
                   capture_output=True, text=True, encoding="utf-8", errors="replace")
    return json.load(open(bp, encoding="utf-8")) if os.path.isfile(bp) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--targets", nargs="+", default=list(TARGETS))
    args = ap.parse_args()
    base = yaml.safe_load(open(os.path.join(REPO, "configs/default.yaml"), encoding="utf-8"))
    table, t0 = {}, time.time()
    for tname in args.targets:
        shape = TARGETS[tname]; cell = f"{tname}_transrot_s{args.seed}"
        print(f"\n=== {cell} ===", flush=True)
        row = {}
        r = run_train(cfg_gsdiff(base, shape, args.seed, "diffusion"), f"{cell}_gsdiff_diff")
        if r: row["gsdiff_diff"] = r["mean_psnr"]; print(f"  gsdiff_diff {r['mean_psnr']:.2f}", flush=True)
        r = run_train(cfg_gsdiff(base, shape, args.seed, "tv"), f"{cell}_gsdiff_tv")
        if r: row["gsdiff_tv"] = r["mean_psnr"]; print(f"  gsdiff_tv   {r['mean_psnr']:.2f}", flush=True)
        r = run_train(cfg_recinr_se2(base, shape, args.seed), f"{cell}_rcse2")
        if r: row["recinr_se2"] = r["mean_psnr"]; print(f"  recinr_se2  {r['mean_psnr']:.2f}", flush=True)
        b = run_baselines(base_cfg(base, shape, args.seed), cell)
        if b:
            for k, v in b["baselines"].items():
                row[k] = v["mean_psnr"]
        table[cell] = row
    out = os.path.join(REPO, "results", "complex_eval", "table.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"table": table, "elapsed": time.time() - t0}, open(out, "w"), indent=2)
    methods = ["dgi", "tv3d", "monin", "gidc3dtv", "recinr", "recinr_se2",
               "gsdiff_tv", "gsdiff_diff"]
    print("\n\n=== COMPLEX-TARGET EVAL (mean PSNR dB) ===")
    print(f"{'cell':26s} " + " ".join(f"{m:>11s}" for m in methods))
    for cell, row in table.items():
        print(f"{cell:26s} " + " ".join(f"{row.get(m, float('nan')):11.2f}" for m in methods))
    print(f"\n({time.time()-t0:.0f}s) → {out}")


if __name__ == "__main__":
    main()
