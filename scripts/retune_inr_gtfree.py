"""Task 2 — STRICT GT-free re-tuning of the INR representation controls.

Fairness fix: the INR baselines (SIREN w0, ReCINR-SE2 canonical grid_size) were
previously picked while looking at PSNR. Here we sweep each knob and SELECT purely
on the GT-free per-frame held-out measurement residual (data.eval_*), never PSNR.
PSNR is recorded only to report what the holdout-selected config achieves.

Selection metric = mean over (motion x seed) of results.json['holdout_residual']
(the per-frame z-scored eval residual train.py already computes on fresh patterns).

Usage:
  python scripts/retune_inr_gtfree.py --which siren     # sweeps siren_w0
  python scripts/retune_inr_gtfree.py --which recinr_se2  # sweeps recinr_grid_size
"""
import argparse, copy, json, os, subprocess, sys
import numpy as np, yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable

MOTIONS = {
    "trans":    dict(gt_velocity=[8, 8], gt_omega=0.0, enable_rotation=False),
    "transrot": dict(gt_velocity=[8, 8], gt_omega=0.3, enable_rotation=True),
}
SWEEPS = {
    "siren":      ("scene.siren_w0",        [8.0, 10.0, 12.0, 15.0, 20.0, 30.0]),
    "recinr_se2": ("scene.recinr_grid_size", [12, 16, 20, 24, 32]),
}


def make_cfg(base, scene_type, motion, seed, val_key, val):
    cfg = copy.deepcopy(base)
    cfg["seed"] = seed
    cfg["data"]["shape"] = "assets/tank.png"
    cfg["data"]["num_patterns"] = 2560
    cfg["data"]["holdout_extra"] = 250          # GT-free eval set ON
    for k in ("gt_velocity", "gt_omega"):
        cfg["data"][k] = motion[k]
    cfg["motion"]["enable_rotation"] = motion["enable_rotation"]
    cfg["motion"]["poly_degree"] = 1
    if scene_type == "siren":
        cfg["scene"] = {"type": "siren", "siren_w0": 20.0, "init_mode": "random"}
        cfg["solver"]["type"] = "sgd"; cfg["solver"]["sgd_steps"] = 4000
        cfg["solver"]["lr_scene"] = 3e-3
    else:
        cfg["scene"] = {"type": "recinr_se2", "init_mode": "random"}
        cfg["solver"]["type"] = "sgd"; cfg["solver"]["sgd_steps"] = 3000
        cfg["solver"]["lr_scene"] = 3e-3; cfg["solver"]["motion_warmup"] = 500
    # apply the swept value (dotted)
    d = cfg
    for p in val_key.split(".")[:-1]:
        d = d.setdefault(p, {})
    d[val_key.split(".")[-1]] = val
    return cfg


def run(cfg, tag):
    d = os.path.join(REPO, "results", "retune_inr", tag)
    rp = os.path.join(d, "results.json")
    if os.path.isfile(rp):
        return json.load(open(rp, encoding="utf-8"))
    os.makedirs(d, exist_ok=True)
    cfg["output_dir"] = f"./results/retune_inr/{tag}"
    p = os.path.join(d, "_cfg.yaml")
    yaml.dump(cfg, open(p, "w", encoding="utf-8"))
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    subprocess.run([PY, "train.py", "--config", p], cwd=REPO, env=env,
                   capture_output=True, text=True, encoding="utf-8", errors="replace")
    return json.load(open(rp, encoding="utf-8")) if os.path.isfile(rp) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=list(SWEEPS), required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[7])
    args = ap.parse_args()
    base = yaml.safe_load(open(os.path.join(REPO, "configs/default.yaml"), encoding="utf-8"))
    val_key, vals = SWEEPS[args.which]

    table = {}
    for val in vals:
        res, psn = [], []
        for mname, motion in MOTIONS.items():
            for seed in args.seeds:
                tag = f"{args.which}_{val_key.split('.')[-1]}{val}_{mname}_s{seed}"
                r = run(make_cfg(base, args.which, motion, seed, val_key, val), tag)
                if r is None:
                    print(f"  FAIL {tag}"); continue
                # GT-free selection metric: the NON-INVASIVE per-frame eval residual
                # (eval_residual_pf) — NOT holdout_residual (that is the legacy
                # holdout_mod field, None here, correctly avoided per CLAUDE.md).
                hv = r.get("eval_residual_pf") or r.get("eval_residual")
                if hv is None:
                    print(f"  NO-EVAL {tag}"); continue
                res.append(hv); psn.append(r.get("mean_psnr"))
                print(f"  {tag:44s} eval_pf={hv:.6f}  PSNR={r.get('mean_psnr'):.2f}", flush=True)
        res = [x for x in res if x is not None]
        table[val] = dict(holdout=float(np.mean(res)) if res else None,
                          psnr=float(np.mean(psn)) if psn else None,
                          n=len(res))
        print(f"== {val_key}={val}: mean holdout={table[val]['holdout']}  mean PSNR={table[val]['psnr']}", flush=True)

    # SELECT on holdout (GT-free), not PSNR
    valid = {k: v for k, v in table.items() if v["holdout"] is not None}
    best = min(valid, key=lambda k: valid[k]["holdout"])
    print(f"\n### GT-free selected {val_key} = {best}  "
          f"(holdout={valid[best]['holdout']:.6f}, PSNR={valid[best]['psnr']:.2f})")
    psnr_best = max(valid, key=lambda k: valid[k]["psnr"])
    print(f"### (for reference, PSNR-argmax would pick {val_key}={psnr_best}, "
          f"PSNR={valid[psnr_best]['psnr']:.2f}) — MATCH={best==psnr_best}")
    out = os.path.join(REPO, "results", "retune_inr", f"summary_{args.which}.json")
    json.dump({"val_key": val_key, "table": {str(k): v for k, v in table.items()},
               "gtfree_selected": best, "psnr_argmax": psnr_best}, open(out, "w"), indent=2)
    print(f"→ {out}")


if __name__ == "__main__":
    main()
