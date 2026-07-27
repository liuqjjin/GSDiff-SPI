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
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from gsdiff.data import (
    ReconstructionOutput,
    dgi_reconstruct,
    generate_spi_data,
    load_acquisition_data,
    method_execution_policy,
    write_method_child_outputs,
)

ALL = ["dgi", "static_cs", "perframe_cs", "tv3d", "monin", "gidc3dtv", "recinr"]


def _write_baselines_json(path, summary):
    """Write a legacy-labelled baseline payload with explicit row aliases."""
    payload = dict(summary)
    payload["metric_definition_version"] = "legacy-per-frame-minmax-v1"
    payload["baselines"] = {
        name: {
            **row,
            "mean_psnr_legacy_per_frame_minmax": row["mean_psnr"],
            "per_frame_psnr_legacy_per_frame_minmax": row[
                "per_frame_psnr"
            ],
        }
        for name, row in summary["baselines"].items()
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=float)


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


def _compatibility_data(acquisition, truth):
    return SimpleNamespace(
        **vars(acquisition),
        frame_idx=acquisition.frame_indices,
        t_grid=acquisition.time_grid,
        eval_patterns=acquisition.holdout_patterns,
        eval_measurements=acquisition.holdout_measurements,
        eval_frame_idx=acquisition.holdout_frame_indices,
        gt_frames=truth.gt_frames,
        gt_velocity=truth.gt_velocity,
    )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--baselines", nargs="+", default=ALL)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--measurements-path", default=None)
    ap.add_argument("--truth-path", default=None)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args(argv)

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    measurements_path = (
        Path(args.measurements_path).expanduser().resolve(strict=True)
        if args.measurements_path
        else None
    )
    truth_path = (
        Path(args.truth_path).expanduser().resolve(strict=True)
        if args.truth_path
        else None
    )
    if truth_path is not None and measurements_path is None:
        raise ValueError("--truth-path requires --measurements-path")
    blind_method_child = measurements_path is not None and truth_path is None
    execution_policy = method_execution_policy(truth_path=truth_path)
    acquisition = None
    if measurements_path is not None:
        if not isinstance(cfg.get("dataset_spec"), dict):
            raise ValueError(
                "method-child config requires complete dataset_spec"
            )
        acquisition = load_acquisition_data(
            measurements_path, expected_spec=cfg["dataset_spec"]
        )
        data = acquisition
        if truth_path is not None:
            from gsdiff.data import load_evaluation_truth

            truth = load_evaluation_truth(
                truth_path,
                expected_dataset_identity_sha256=(
                    acquisition.dataset_identity_sha256
                ),
            )
            data = _compatibility_data(acquisition, truth)
    else:
        data = _gen(cfg)
    dev = args.device
    out_dir = args.output_dir or os.path.join(REPO, "results", args.name)
    os.makedirs(out_dir, exist_ok=True)
    rows = {}
    raw_result = None
    if blind_method_child and (
        len(args.baselines) != 1 or args.baselines[0] != "dgi"
    ):
        raise ValueError(
            "blind run_baselines currently supports one dgi method child"
        )
    if not blind_method_child:
        from gsdiff.baselines import cs, gidc, monin, recinr, tv3d
        from gsdiff.baselines.common import dgi_image

    def record(name, recon, info=None):
        nonlocal raw_result
        if blind_method_child:
            raw_result = (name, np.asarray(recon), info or {})
            return
        from gsdiff.baselines.common import evaluate_video

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
        if blind_method_child:
            dgi = np.asarray(
                dgi_reconstruct(data.patterns, data.measurements),
                dtype=np.float32,
            )
        else:
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

    if blind_method_child:
        if raw_result is None:
            raise RuntimeError("selected method did not produce a reconstruction")
        method_name, reconstruction, info = raw_result
        dgi = np.asarray(reconstruction[0], dtype=np.float32)
        output = ReconstructionOutput(
            dataset_identity_sha256=acquisition.dataset_identity_sha256,
            reconstruction=reconstruction.astype(np.float32, copy=False),
            dgi=dgi,
            estimated_motion_trajectory=np.zeros(
                (acquisition.T, 3), dtype=np.float32
            ),
            frame_indices=np.arange(acquisition.T, dtype=np.int64),
            time_grid=acquisition.time_grid,
            method_name=method_name,
            method_metadata={
                "baseline": method_name,
                "device": dev,
                "motion_estimate_available": False,
                **info,
            },
            execution_policy=execution_policy,
        )
        write_method_child_outputs(Path(out_dir), output, history=[])
        print(f"→ {out_dir}/reconstruction.npz")
        return

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
    if truth_path is not None:
        summary.update({
            "execution_class": execution_policy.execution_class,
            "truth_access": execution_policy.truth_access,
            "promotion_eligible": execution_policy.promotion_eligible,
        })
    _write_baselines_json(bpath, summary)
    print(f"→ {out_dir}/baselines.json  ({summary['elapsed']:.0f}s)")


if __name__ == "__main__":
    main()
