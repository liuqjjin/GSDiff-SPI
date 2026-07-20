"""Multi-seed experiment runner + aggregator.

Runs train.py once per seed on a base config, collects results.json,
and reports mean +/- std. Idempotent: seeds whose results.json already
exists are skipped (delete the seed dir to re-run).

Usage:
    python scripts/run_multiseed.py --config configs/default.yaml \
        --seeds 7 11 42 --name myexp
    python scripts/run_multiseed.py --config configs/default.yaml \
        --seeds 7 11 42 --name ab_hqs --override solver.hqs=true

Overrides use dotted keys; values are parsed as YAML (true/1.5/[a,b] all work).
Outputs: results/<name>/seed<seed>/...  and  results/<name>/summary.json
"""
import argparse, copy, json, os, subprocess, sys
import numpy as np
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def set_dotted(d, key, value):
    parts = key.split('.')
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 42])
    ap.add_argument("--name", required=True, help="results/<name>/ output root")
    ap.add_argument("--override", action="append", default=[],
                    help="dotted.key=yaml_value, repeatable")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        base = yaml.safe_load(f)
    for ov in args.override:
        key, _, val = ov.partition("=")
        set_dotted(base, key.strip(), yaml.safe_load(val))

    root = os.path.join(REPO, "results", args.name)
    os.makedirs(root, exist_ok=True)

    rows = []
    for seed in args.seeds:
        run_dir = os.path.join(root, f"seed{seed}")
        res_path = os.path.join(run_dir, "results.json")
        if not os.path.isfile(res_path):
            cfg = copy.deepcopy(base)
            cfg["seed"] = seed
            cfg["output_dir"] = os.path.relpath(run_dir, REPO).replace("\\", "/")
            tmp_cfg = os.path.join(root, f"config_seed{seed}.yaml")
            with open(tmp_cfg, "w", encoding="utf-8") as f:
                yaml.dump(cfg, f)
            print(f"── seed {seed}: running…", flush=True)
            env = dict(os.environ, PYTHONIOENCODING="utf-8")
            r = subprocess.run([args.python, "train.py", "--config", tmp_cfg],
                               cwd=REPO, env=env,
                               capture_output=True, text=True, encoding="utf-8")
            if r.returncode != 0 or not os.path.isfile(res_path):
                print(r.stdout[-2000:] if r.stdout else "")
                print(r.stderr[-2000:] if r.stderr else "")
                print(f"seed {seed} FAILED (exit {r.returncode})")
                continue
        else:
            print(f"── seed {seed}: cached")
        rows.append((seed, json.load(open(res_path, encoding="utf-8"))))

    if not rows:
        sys.exit("no successful runs")

    def stat(key):
        vals = [r[key] for _, r in rows if r.get(key) is not None]
        return (float(np.mean(vals)), float(np.std(vals))) if vals else (None, None)

    psnr_m, psnr_s = stat("mean_psnr")
    hold_m, hold_s = stat("holdout_residual")
    verr = np.array([r["velocity_error"] for _, r in rows], dtype=float)
    summary = {
        "name": args.name, "config": args.config, "overrides": args.override,
        "seeds": [s for s, _ in rows],
        "psnr_mean": psnr_m, "psnr_std": psnr_s,
        "psnr_per_seed": {s: r["mean_psnr"] for s, r in rows},
        "holdout_mean": hold_m, "holdout_std": hold_s,
        "vel_err_mean": verr.mean(axis=0).tolist(),
        "vel_err_std": verr.std(axis=0).tolist(),
    }
    with open(os.path.join(root, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{args.name}: PSNR {psnr_m:.2f} ± {psnr_s:.2f} dB over seeds {summary['seeds']}")
    if hold_m is not None:
        print(f"          holdout {hold_m:.6f} ± {hold_s:.6f}")
    print(f"          vel_err {summary['vel_err_mean']}")


if __name__ == "__main__":
    main()
