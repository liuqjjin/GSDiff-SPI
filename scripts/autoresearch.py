"""autoresearch.py — coordinate-descent hyperparameter search for GSDiff-SPI.

Protocol (after karpathy/autoresearch via ReCINR's campaign): fixed budget per
candidate, ONE selection metric, keep-if-improves, full log. Two hard rules
(learned the hard way in the sibling ReCINR project):

  1. The selection metric is GROUND-TRUTH-FREE: the mean residual on a
     NON-INVASIVE eval set (results.json eval_residual; data.holdout_extra
     fresh random-pattern measurements never shown to the solver — training is
     byte-identical to production). PSNR is recorded but NEVER used to accept
     a candidate. Do NOT use in-training holdout (data.holdout_mod) here:
     removing measurements was measured to destabilize the knife-edge joint
     scene+motion basin (seed42 rot: 27.34 → 10.15 dB).
  2. Every accept/reject uses the multi-(motion x seed) MEAN. A single lucky
     seed can never promote a change. Optionally require an improvement margin
     (--min-improve) calibrated to seed noise.

Idempotent: each (candidate, motion, seed) run is cached by its results.json;
the campaign log is rewritten after every step, so the search is resumable.

Usage:
    python scripts/autoresearch.py --base configs/default.yaml
    python scripts/autoresearch.py --base <cfg> --moves-set diffusion
"""
from __future__ import annotations
import argparse, copy, json, os, subprocess, sys, time
import numpy as np
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
SEEDS = (7, 11, 42)

# Motion cases: name -> dotted-key overrides applied to the base config.
MOTIONS = {
    "tr":  {"data.gt_velocity": [8, 8], "data.gt_omega": 0.0,
            "motion.enable_rotation": False},
    "rot": {"data.gt_velocity": [8, 8], "data.gt_omega": 0.3,
            "motion.enable_rotation": True},
}

# Pre-registered moves (coordinate descent, one at a time), ordered by
# hypothesis strength. Values must differ from the incumbent to be tried.
MOVES_COMMON = [
    ("solver.rho", 0.05), ("solver.rho", 0.2),
    ("solver.rho_growth", 1.0), ("solver.rho_growth", 1.1),
    ("solver.admm_soft_tv_weight", 0.004), ("solver.admm_soft_tv_weight", 0.009),
    ("solver.temporal_tv_weight", 0.03), ("solver.temporal_tv_weight", 0.07),
    ("solver.admm_n_warmup", 10), ("solver.admm_n_warmup", 30),
    ("solver.admm_lr_scene", 0.006), ("solver.admm_lr_scene", 0.012),
    ("solver.admm_lr_motion", 0.10), ("solver.admm_lr_motion", 0.20),
    ("scene.num_gaussians", 250), ("scene.num_gaussians", 1000),
]
MOVES_TV = [("solver.admm_tv_weight", 0.003), ("solver.admm_tv_weight", 0.008)]
MOVES_DIFFUSION = [
    ("solver.diffusion_prior.sigma_start", 0.2),
    ("solver.diffusion_prior.sigma_start", 0.4),
    ("solver.diffusion_prior.sigma_end", 0.02),
    ("solver.diffusion_prior.sigma_end", 0.1),
    ("solver.diffusion_prior.denoise_steps", 3),
    ("solver.diffusion_prior.renoise", True),
]


def set_dotted(d, key, value):
    parts = key.split('.')
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value


def get_dotted(d, key, default=None):
    for p in key.split('.'):
        if not isinstance(d, dict) or p not in d:
            return default
        d = d[p]
    return d


def run_one(base_cfg, overrides, out_rel):
    """Run train.py once (cached by results.json); returns results dict or None."""
    run_dir = os.path.join(REPO, "results", out_rel)
    res_path = os.path.join(run_dir, "results.json")
    if not os.path.isfile(res_path):
        cfg = copy.deepcopy(base_cfg)
        for k, v in overrides.items():
            set_dotted(cfg, k, v)
        cfg["output_dir"] = f"./results/{out_rel}"
        os.makedirs(run_dir, exist_ok=True)
        tmp = os.path.join(run_dir, "_run_config.yaml")
        with open(tmp, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f)
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        r = subprocess.run([PY, "train.py", "--config", tmp], cwd=REPO, env=env,
                           capture_output=True, text=True, encoding="utf-8")
        if r.returncode != 0 or not os.path.isfile(res_path):
            print(f"    RUN FAILED: {out_rel}\n{(r.stderr or '')[-800:]}", flush=True)
            return None
    return json.load(open(res_path, encoding="utf-8"))


COLLAPSE_THR = 0.05   # eval_residual above this = collapsed basin (GT-free alarm)


def evaluate(base_cfg, cand_overrides, name):
    """-> (metric [lower=better], n_collapses, mean PSNR [reported only], per-motion PSNR).

    Metric = mean over motion cases of the per-case MEDIAN eval_residual over
    seeds (median is robust to a minority of bimodal-basin collapses, which
    would otherwise turn coordinate descent into a collapse lottery).
    """
    per_case, psnrs, per, ncol = {}, [], {}, 0
    for mtag, movr in MOTIONS.items():
        for s in SEEDS:
            ovr = dict(movr); ovr.update(cand_overrides); ovr["seed"] = s
            res = run_one(base_cfg, ovr, f"ar/{name}_{mtag}_s{s}")
            if res is None or res.get("eval_residual") is None:
                return float("inf"), 99, 0.0, {}
            per_case.setdefault(mtag, []).append(res["eval_residual"])
            ncol += int(res["eval_residual"] > COLLAPSE_THR)
            psnrs.append(res["mean_psnr"])
            per.setdefault(mtag, []).append(res["mean_psnr"])
    per = {k: round(float(np.mean(v)), 2) for k, v in per.items()}
    metric = float(np.mean([np.median(v) for v in per_case.values()]))
    return metric, ncol, float(np.mean(psnrs)), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="base YAML config (incumbent)")
    ap.add_argument("--moves-set", choices=["tv", "diffusion", "common"], default="common")
    ap.add_argument("--min-improve", type=float, default=0.0,
                    help="required holdout improvement (calibrate to seed noise)")
    ap.add_argument("--log", default=os.path.join(REPO, "results", "ar", "campaign.json"))
    args = ap.parse_args()

    with open(args.base, encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)
    if int(get_dotted(base_cfg, "data.holdout_extra", 0) or 0) <= 0:
        set_dotted(base_cfg, "data.holdout_extra", 250)  # selection metric prerequisite
        print("NOTE: enabling data.holdout_extra=250 (non-invasive GT-free eval set)")

    moves = list(MOVES_COMMON)
    if args.moves_set == "tv":
        moves += MOVES_TV
    elif args.moves_set == "diffusion":
        moves += MOVES_DIFFUSION

    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    hist, best_ovr = [], {}
    t0 = time.time()
    best_m, best_ncol, base_p, base_per = evaluate(base_cfg, best_ovr, "base")
    print(f"[base] metric={best_m:.6f} collapses={best_ncol}  PSNR={base_p:.2f}  {base_per}",
          flush=True)
    hist.append({"name": "base", "overrides": {}, "metric": best_m,
                 "collapses": best_ncol, "psnr": base_p, "per": base_per,
                 "accepted": True})

    for i, (k, v) in enumerate(moves):
        incumbent = best_ovr.get(k, get_dotted(base_cfg, k))
        if incumbent == v:
            continue
        cand = dict(best_ovr); cand[k] = v
        name = f"{k.split('.')[-1]}_{v}".replace(".", "p").replace("-", "m")
        cm, cncol, cp, cper = evaluate(base_cfg, cand, name)
        # Accept iff median-metric improves AND collapse count does not grow
        ok = (cm < best_m - args.min_improve) and (cncol <= best_ncol)
        print(f"[{i+1}/{len(moves)}] {k}={v}: metric={cm:.6f} col={cncol} "
              f"(best {best_m:.6f}/{best_ncol}) PSNR={cp:.2f} "
              f"{'ACCEPT' if ok else 'reject'}  {cper}", flush=True)
        hist.append({"name": name, "overrides": dict(cand), "metric": cm,
                     "collapses": cncol, "psnr": cp, "per": cper,
                     "accepted": bool(ok)})
        if ok:
            best_ovr, best_m, best_ncol = cand, cm, cncol
        json.dump({"best_overrides": best_ovr, "best_metric": best_m,
                   "best_collapses": best_ncol, "history": hist},
                  open(args.log, "w", encoding="utf-8"), indent=2)

    print(f"\n=== BEST ({time.time()-t0:.0f}s) holdout={best_m:.6f} ===")
    print(json.dumps(best_ovr, indent=2))


if __name__ == "__main__":
    main()
