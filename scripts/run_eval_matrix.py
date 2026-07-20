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
    if scene_type != "gaussian":
        cfg["scene"] = {"type": scene_type, "siren_w0": 20.0, "init_mode": "dgi"}
        cfg["solver"]["type"] = "sgd"; cfg["solver"]["sgd_steps"] = 4000
        cfg["solver"]["lr_scene"] = 3e-3
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
    methods = ["dgi", "static_cs", "perframe_cs", "monin", "gidc3dtv", "inr_siren", "gsdiff"]
    print("\n\n=== EVAL MATRIX (mean PSNR dB) ===")
    print(f"{'cell':22s} " + " ".join(f"{m:>11s}" for m in methods))
    for cell, row in table.items():
        print(f"{cell:22s} " + " ".join(f"{row.get(m, float('nan')):11.2f}" for m in methods))
    print(f"\n({time.time()-t0:.0f}s total) → {out}")


if __name__ == "__main__":
    main()
