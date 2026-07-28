"""Run one frozen baseline method child.

Usage:
    python scripts/run_baselines.py --method dgi --dataset <absolute-measurements> \
        --dataset-identity-sha256 <sha256> --method-config <absolute-config> \
        --algorithm-seed <u32> --device cpu --output-dir <absolute-output>

Legacy truth-visible evaluation is explicit and nonpromotable:
    python scripts/run_baselines.py --legacy-compatibility --truth-path <truth> \
        --config configs/default.yaml --name legacy-evaluation \
        --measurements-path <measurements> \
        --dataset-identity-sha256 <sha256>
"""
import argparse, contextlib, io, json, os, sys, time
from datetime import datetime, timezone
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
STRICT_SINGLETON_OPTIONS = (
    "--method",
    "--dataset",
    "--dataset-identity-sha256",
    "--method-config",
    "--algorithm-seed",
    "--device",
    "--output-dir",
)
LEGACY_ONLY_OPTIONS = (
    "--config",
    "--measurements-path",
    "--truth-path",
    "--name",
    "--baselines",
    "--solver",
    "--override",
)


def _utc_now():
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _option_present(argv, option):
    return any(
        value == option or value.startswith(option + "=")
        for value in argv
    )


def _legacy_truth_supplied(argv):
    for index, value in enumerate(argv):
        if value.startswith("--truth-path="):
            return bool(value.partition("=")[2].strip())
        if value == "--truth-path":
            return (
                index + 1 < len(argv)
                and bool(argv[index + 1].strip())
                and not argv[index + 1].startswith("--")
            )
    return False


def _require_parsed_legacy_truth(value):
    if not isinstance(value, str) or not value.strip():
        print(
            "truthless legacy method output is disabled; "
            "use the strict method interface",
            file=sys.stderr,
        )
        raise SystemExit(2)


def _reject_duplicate_legacy_truth(argv):
    count = sum(
        value == "--truth-path"
        or value.startswith("--truth-path=")
        for value in argv
    )
    if count > 1:
        raise SystemExit("duplicate legacy option: --truth-path")


def _reject_duplicate_singletons(argv):
    for option in STRICT_SINGLETON_OPTIONS:
        count = sum(
            value == option or value.startswith(option + "=")
            for value in argv
        )
        if count > 1:
            raise SystemExit(f"duplicate strict option: {option}")


def _strict_seed(value):
    try:
        seed = int(value, 10)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "algorithm seed must be an unsigned 32-bit integer"
        ) from error
    if not 0 <= seed <= 2**32 - 1:
        raise argparse.ArgumentTypeError(
            "algorithm seed must be an unsigned 32-bit integer"
        )
    return seed


def _strict_parser():
    parser = argparse.ArgumentParser(
        description="Run one frozen baseline method child.",
        allow_abbrev=False,
    )
    parser.add_argument("--method", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-identity-sha256", required=True)
    parser.add_argument("--method-config", required=True)
    parser.add_argument(
        "--algorithm-seed", required=True, type=_strict_seed
    )
    parser.add_argument("--device", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def _path_key(value):
    return os.path.normcase(os.path.abspath(os.fspath(value)))


def _crosslock_path(name, supplied, expected):
    supplied_path = Path(supplied)
    if not supplied_path.is_absolute():
        raise ValueError(f"{name} crosslock mismatch")
    if _path_key(supplied_path) != _path_key(expected):
        raise ValueError(f"{name} crosslock mismatch")


def _crosslock_method_config_before_load(supplied):
    """Bind config loading to the materializer-owned stage beside cwd."""
    cwd = Path.cwd()
    if cwd.name != "code":
        raise ValueError("method config path crosslock mismatch")
    expected = cwd.parent / "config" / "method-config.json"
    _crosslock_path("method config path", supplied, expected)
    return Path(supplied)


def _crosslock_request(args, request, method_config_path):
    if args.method != request.method.method_id:
        raise ValueError("method crosslock mismatch")
    _crosslock_path(
        "dataset path", args.dataset, request.measurements_path
    )
    if args.dataset_identity_sha256 != request.dataset_identity_sha256:
        raise ValueError("dataset identity crosslock mismatch")
    expected_config_path = (
        request.measurements_path.parent.parent
        / "config"
        / "method-config.json"
    )
    _crosslock_path(
        "method config path", method_config_path, expected_config_path
    )
    _crosslock_path(
        "output path", args.output_dir, request.child_output_dir
    )
    if args.algorithm_seed != request.algorithm_seed.seed_u32:
        raise ValueError("algorithm seed crosslock mismatch")
    if args.device != request.child_runtime_device:
        raise ValueError("child device crosslock mismatch")
    if request.checkpoint_paths:
        raise ValueError("baseline checkpoint crosslock mismatch")


def strict_main(argv=None):
    """Run one canonical baseline from one frozen materialized request."""
    strict_argv = list(sys.argv[1:] if argv is None else argv)
    _reject_duplicate_singletons(strict_argv)
    args = _strict_parser().parse_args(strict_argv)
    method_config_path = _crosslock_method_config_before_load(
        args.method_config
    )

    import gsdiff.data as data
    import gsdiff.experiments.adapters as adapters
    import gsdiff.experiments.child_outputs as child_outputs
    import gsdiff.experiments.execution as execution

    request = execution.load_materialized_method_request(
        method_config_path
    )
    _crosslock_request(args, request, method_config_path)
    if (
        request.method.execution_family != "baseline"
        or request.method.method_id not in ALL
    ):
        raise ValueError("baseline method family mismatch")

    child_started_at_utc = _utc_now()
    acquisition = data.load_acquisition_data(
        request.measurements_path,
        expected_dataset_identity_sha256=(
            request.dataset_identity_sha256
        ),
        expected_acquisition_spec=request.expected_acquisition_spec,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        result = adapters.run_canonical_method(
            request.method,
            acquisition,
            algorithm_seed=request.algorithm_seed,
            checkpoint_paths=request.checkpoint_paths,
            device=request.child_runtime_device,
        )
    child_finished_at_utc = _utc_now()
    child_outputs.write_method_child_outputs_v2(
        request.child_output_dir,
        method=request.method,
        acquisition=acquisition,
        measurements_file_sha256=request.measurements_file_sha256,
        algorithm_seed=request.algorithm_seed,
        result=result,
        child_started_at_utc=child_started_at_utc,
        child_finished_at_utc=child_finished_at_utc,
    )
    print(f"completed method child: {request.method.method_id}")
    print(request.child_output_dir / "reconstruction.npz")
    print(request.child_output_dir / "method-info.json")
    return 0


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


def legacy_main(argv=None):
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--config", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--baselines", nargs="+", default=ALL)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--measurements-path", default=None)
    ap.add_argument("--dataset-identity-sha256", default=None)
    ap.add_argument("--truth-path", default=None)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args(argv)
    _require_parsed_legacy_truth(args.truth_path)

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
    if (measurements_path is None) != (
        args.dataset_identity_sha256 is None
    ):
        raise ValueError(
            "--measurements-path and --dataset-identity-sha256 "
            "must be provided together"
        )
    blind_method_child = measurements_path is not None and truth_path is None
    execution_policy = method_execution_policy(truth_path=truth_path)
    acquisition = None
    if measurements_path is not None:
        if not isinstance(cfg.get("acquisition_spec"), dict):
            raise ValueError(
                "method-child config requires blind acquisition_spec"
            )
        acquisition = load_acquisition_data(
            measurements_path,
            expected_dataset_identity_sha256=(
                args.dataset_identity_sha256
            ),
            expected_acquisition_spec=cfg["acquisition_spec"],
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


def main(argv=None):
    """Dispatch strict-by-default or explicit legacy compatibility mode."""
    dispatch_argv = list(sys.argv[1:] if argv is None else argv)
    marker_count = dispatch_argv.count("--legacy-compatibility")
    if marker_count > 1:
        raise SystemExit("duplicate strict option: --legacy-compatibility")
    if marker_count == 1:
        dispatch_argv.remove("--legacy-compatibility")
        _reject_duplicate_legacy_truth(dispatch_argv)
        if not _legacy_truth_supplied(dispatch_argv):
            print(
                "truthless legacy method output is disabled; "
                "use the strict method interface",
                file=sys.stderr,
            )
            raise SystemExit(2)
        print(
            "LEGACY COMPATIBILITY: outputs are nonpromotable.",
            file=sys.stderr,
        )
        return legacy_main(dispatch_argv)
    if any(
        _option_present(dispatch_argv, option)
        for option in LEGACY_ONLY_OPTIONS
    ):
        print(
            "legacy flags require explicit --legacy-compatibility",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return strict_main(dispatch_argv)


if __name__ == "__main__":
    main()
