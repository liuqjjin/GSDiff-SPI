
"""Run one frozen GSDiff method child.

Usage:
    python train.py --method gsdiff_tv --dataset <absolute-measurements> \
        --dataset-identity-sha256 <sha256> --method-config <absolute-config> \
        --algorithm-seed <u32> --device cpu --output-dir <absolute-output>

Legacy truth-visible evaluation is explicit and nonpromotable:
    python train.py --legacy-compatibility --truth-path <truth> \
        --config configs/default.yaml --measurements-path <measurements> \
        --dataset-identity-sha256 <sha256>
"""
import argparse, contextlib, io, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import numpy as np, torch, yaml

from gsdiff.utils import (set_seed, get_device, ensure_dir, normalize_01,
                           psnr_fn, to_native, save_gif, _to_ns)
from gsdiff.scene import GaussianScene2D
from gsdiff.motion import SE2Motion
from gsdiff.forward import SPIForwardModel
from gsdiff.prior import TVPrior, TVPrior3D
from gsdiff.solver import ADMMSolver, SGDSolver
from gsdiff.data import (
    ReconstructionOutput,
    dgi_reconstruct,
    generate_spi_data,
    load_acquisition_data,
    method_execution_policy,
    write_method_child_outputs,
)

GSDIFF_METHOD_IDS = (
    "siren",
    "recinr_se2",
    "gsdiff_tv",
    "gsdiff_diffusion",
)
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
        description="Run one frozen GSDiff method child.",
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
    parser.add_argument("--checkpoint", action="append", default=[])
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


def _checkpoint_mapping(assignments):
    result = {}
    for assignment in assignments:
        logical_id, separator, path_text = assignment.partition("=")
        if (
            not separator
            or not logical_id
            or not path_text
            or logical_id in result
        ):
            raise ValueError("checkpoint mapping is malformed or duplicate")
        checkpoint_path = Path(path_text)
        if not checkpoint_path.is_absolute():
            raise ValueError("checkpoint path crosslock mismatch")
        result[logical_id] = checkpoint_path
    return result


def _crosslock_request(
    args,
    request,
    method_config_path,
    checkpoint_mapping,
):
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
    if set(checkpoint_mapping) != set(request.checkpoint_paths):
        raise ValueError("checkpoint logical-ID crosslock mismatch")
    for logical_id, expected_path in request.checkpoint_paths.items():
        _crosslock_path(
            f"checkpoint {logical_id!r} path",
            checkpoint_mapping[logical_id],
            expected_path,
        )


def strict_main(argv=None):
    """Run one canonical GSDiff method from a frozen materialized request."""
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
    checkpoint_mapping = _checkpoint_mapping(args.checkpoint)
    _crosslock_request(
        args,
        request,
        method_config_path,
        checkpoint_mapping,
    )
    if (
        request.method.execution_family != "gsdiff"
        or request.method.method_id not in GSDIFF_METHOD_IDS
    ):
        raise ValueError("GSDiff method family mismatch")

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


def _compatibility_pyplot():
    """Load plotting only for the legacy evaluation/figure workflow."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    return pyplot


def parse_args(argv=None):
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--solver", default=None)
    p.add_argument("--measurements-path", default=None)
    p.add_argument("--dataset-identity-sha256", default=None)
    p.add_argument("--truth-path", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--device", default=None)
    return p.parse_args(argv)


def evaluate(gt_frames, recon_np, label=""):
    """Compatibility adapter for explicitly labelled legacy PSNR."""
    from gsdiff.evaluation.metrics import (
        evaluate_video_legacy_per_frame,
    )

    result = evaluate_video_legacy_per_frame(gt_frames, recon_np)
    psnrs = result["per_frame_psnr_legacy_per_frame_minmax"]
    m = result["psnr_legacy_per_frame_minmax"]
    print(f"  {label:8s} PSNR: {m:.2f} dB  [first={psnrs[0]:.1f}, last={psnrs[-1]:.1f}]")
    return psnrs, m


def _record_sigma_used(info, prior):
    """Record and return the diffusion sigma consumed by the completed step."""
    sigma_used = getattr(prior, "last_sigma", None)
    if sigma_used is not None:
        info["sigma_used"] = sigma_used
    return sigma_used


def _admm_warmup_counts(solver_config):
    """Resolve independent warmups while preserving the legacy split name."""
    splitting = getattr(
        solver_config,
        "splitting_warmup_outer",
        getattr(solver_config, "admm_n_warmup", 0),
    )
    motion = getattr(solver_config, "motion_warmup_outer", 0)
    return splitting, motion


def _write_results_json(path, results, history):
    """Persist explicitly labelled legacy compatibility results and history."""
    scalar_history = [
        {key: value for key, value in entry.items() if key != "video"}
        for entry in history
    ]
    payload = {**results, "history": scalar_history}
    payload["metric_definition_version"] = "legacy-per-frame-minmax-v1"
    aliases = {
        "mean_psnr": "mean_psnr_legacy_per_frame_minmax",
        "per_frame_psnr": "per_frame_psnr_legacy_per_frame_minmax",
        "dgi_psnr": "dgi_psnr_legacy_canonical_minmax_60db",
    }
    for compatibility_name, explicit_name in aliases.items():
        if compatibility_name in payload:
            payload[explicit_name] = payload[compatibility_name]
    payload = to_native(payload)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _write_metrics_json(path, gt_frames, recon_np):
    """Evaluate the final video once and persist the primary metrics payload."""
    from gsdiff.evaluation.metrics import evaluate_video_global_affine

    payload = evaluate_video_global_affine(gt_frames, recon_np)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, allow_nan=False)
    return payload


# ─── Visualization ────────────────────────────────────────────
def save_comparison_figure(gt, dgi_img, recon_np, out_path, title=""):
    """Save GT vs DGI vs Recon for first/mid/last frames."""
    plt = _compatibility_pyplot()
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
    plt = _compatibility_pyplot()
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
    plt = _compatibility_pyplot()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy([h["loss_data"] for h in history], label="data fidelity")
    if "prim_res" in history[0]:
        # During ADMM warmup z≡0, so prim_res=MSE(video,0)=video energy, NOT a
        # residual — plotting it next to data fidelity makes the two curves form a
        # spurious "X" that looks like the losses cancelling. Mask the warmup span
        # (in_warmup=True) so only the true post-warmup primal residual is drawn,
        # and overlay the actual ADMM coupling term (ρ/2)‖R(θ)-(z-u)‖².
        pr = [
            h["prim_res"]
            if not h.get(
                "in_splitting_warmup", h.get("in_warmup", False)
            )
            else float("nan")
            for h in history
        ]
        ax.semilogy(pr, label="primal residual (post-warmup)")
        if any(h.get("loss_consist", 0) for h in history):
            lc = [h["loss_consist"] if h.get("loss_consist", 0) > 0 else float("nan")
                  for h in history]
            ax.semilogy(lc, label=r"consistency (ρ/2)‖R-(z-u)‖²")
        n_warmup = sum(
            bool(
                h.get(
                    "in_splitting_warmup",
                    h.get("in_warmup", False),
                )
            )
            for h in history
        )
        if 0 < n_warmup < len(history):
            ax.axvline(n_warmup, ls="--", color="gray", alpha=0.5,
                       label=f"warmup end (iter {n_warmup})")
    ax.set_xlabel("iteration"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_title(f"{solver_type} convergence")
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()


# ─── Main ─────────────────────────────────────────────────────
def _compatibility_data(acquisition, truth):
    return SimpleNamespace(
        **vars(acquisition),
        frame_idx=acquisition.frame_indices,
        t_grid=acquisition.time_grid,
        eval_patterns=acquisition.holdout_patterns,
        eval_measurements=acquisition.holdout_measurements,
        eval_frame_idx=acquisition.holdout_frame_indices,
        canonical=truth.canonical_image,
        gt_frames=truth.gt_frames,
        gt_velocity=truth.gt_velocity,
        gt_omega=truth.gt_omega,
        motion_type=truth.motion_model,
    )


def _estimated_motion_trajectory(motion_parameters, time_grid):
    times = np.asarray(time_grid, dtype=np.float64)
    velocity = np.asarray(motion_parameters["velocity"], dtype=np.float64)
    acceleration = np.asarray(
        motion_parameters.get("accel", [0.0, 0.0]), dtype=np.float64
    )
    translation = (
        times[:, None] * velocity + times[:, None] ** 2 * acceleration
    )
    rotation = (
        times * float(motion_parameters.get("omega", 0.0))
        + times**2 * float(motion_parameters.get("beta", 0.0))
    )
    return np.column_stack((translation, rotation)).astype(np.float32)


def _run_legacy_compatibility(argv=None):
    """Run the legacy evaluation/figure CLI retained for compatibility."""
    args = parse_args(argv)
    _require_parsed_legacy_truth(args.truth_path)
    with open(args.config, encoding='utf-8') as f:
        raw = yaml.safe_load(f)
    cfg = _to_ns(raw)
    if args.solver:
        cfg.solver.type = args.solver

    set_seed(cfg.seed)
    dev = torch.device(args.device) if args.device else get_device()
    out_dir = args.output_dir or cfg.output_dir
    ensure_dir(out_dir)

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
    plt = None if blind_method_child else _compatibility_pyplot()

    if not blind_method_child:
        with open(os.path.join(out_dir, "config.yaml"), "w", encoding='utf-8') as f:
            yaml.dump(raw, f)

    print(f"Device: {dev}, Solver: {cfg.solver.type}")

    # ── 1. Generate or load data ──────────────────────────────
    if measurements_path is not None:
        if not isinstance(raw.get("acquisition_spec"), dict):
            raise ValueError(
                "method-child config requires blind acquisition_spec"
            )
        acquisition = load_acquisition_data(
            measurements_path,
            expected_dataset_identity_sha256=(
                args.dataset_identity_sha256
            ),
            expected_acquisition_spec=raw["acquisition_spec"],
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
        H, W, T, K = acquisition.H, acquisition.W, acquisition.T, acquisition.K
        _time_mode = acquisition.time_assignment_mode
    else:
        H, W = cfg.data.image_size
        T, K = cfg.data.num_frames, cfg.data.num_patterns
        _time_mode = getattr(cfg.data, 'time_assignment_mode', 'uniform')
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
            gt_accel=list(cfg.data.gt_accel) if getattr(cfg.data, 'gt_accel', None) else None,
            gt_beta=getattr(cfg.data, 'gt_beta', None),
            noise_sigma_abs=getattr(cfg.data, 'noise_sigma_abs', None),
            holdout_extra=int(getattr(cfg.data, 'holdout_extra', 0) or 0),
            time_assignment_mode=_time_mode,
            pattern_order=getattr(cfg.data, 'pattern_order', 'sequential'),
        )

    # Update T in case motion_mode=1 changed it
    T = data.T

    motion_label = "blind" if measurements_path is not None else data.motion_type
    print(f"Data: {H}x{W}, T={T}, K={K}, motion={motion_label}")
    if not blind_method_child:
        print(f"GT vel={data.gt_velocity}, omega={data.gt_omega:.4f}")
    print(f"Meas range: [{data.measurements.min():.1f}, {data.measurements.max():.1f}]")

    # ── 2. DGI baseline ──────────────────────────────────────
    print("\nDGI reconstruction...")
    dgi_img = dgi_reconstruct(data.patterns, data.measurements)
    dgi_psnr = None
    if not blind_method_child:
        dgi_psnr = psnr_fn(normalize_01(dgi_img), normalize_01(data.canonical))
        print(f"  DGI PSNR (vs canonical): {dgi_psnr:.2f} dB  (motion-blurred)")
        plt.imsave(os.path.join(out_dir, "dgi_recon.png"),
                   normalize_01(dgi_img), cmap='gray')
        save_gif(data.gt_frames, os.path.join(out_dir, "gt_video.gif"), fps=5)

    # ── 3. Convert to torch ───────────────────────────────────
    pat = torch.tensor(data.patterns, device=dev)
    y = torch.tensor(data.measurements, device=dev)
    fidx = torch.tensor(data.frame_idx, device=dev)
    tg = torch.tensor(data.t_grid, device=dev)

    # ── 3.5. Optional GT-free measurement holdout ─────────────
    # holdout_mod=10 → every measurement with k % 10 == holdout_offset is
    # excluded from the training loss and used only for model selection.
    _hold_mod = int(getattr(cfg.data, 'holdout_mod', 0) or 0)
    if _hold_mod > 1:
        _hold_off = int(getattr(cfg.data, 'holdout_offset', 7)) % _hold_mod
        hold_mask = torch.tensor((np.arange(K) % _hold_mod) == _hold_off, device=dev)
        pat_tr, y_tr, fidx_tr = pat[~hold_mask], y[~hold_mask], fidx[~hold_mask]
        print(f"Holdout: {int(hold_mask.sum())}/{K} measurements (k%{_hold_mod}=={_hold_off})")
    else:
        hold_mask = None
        pat_tr, y_tr, fidx_tr = pat, y, fidx

    # ── 4. Build model ────────────────────────────────────────
    # scene.type selects the representation: gaussian (default) or an INR/grid/
    # low-rank canonical (the representation control — same motion, loss, solver).
    _scene_type = getattr(cfg.scene, 'type', 'gaussian')
    motion = SE2Motion(((H-1)/2.0, (W-1)/2.0), cfg.motion.enable_rotation,
                       poly_degree=getattr(cfg.motion, 'poly_degree', 1),
                       enable_affine=getattr(cfg.motion, 'enable_affine', False)).to(dev)
    if _scene_type == 'gaussian':
        scene = GaussianScene2D(cfg.scene.num_gaussians, H, W, cfg.scene.init_scale,
                                min_scale=getattr(cfg.scene, 'min_scale', 0.0)).to(dev)
        fwd = SPIForwardModel(scene, motion, H, W, time_assignment_mode=_time_mode).to(dev)
    else:
        from gsdiff.baselines.inr import build_scene, INRForwardModel
        scene = build_scene(_scene_type, w0=getattr(cfg.scene, 'siren_w0', 20.0),
                            r=getattr(cfg.scene, 'lowrank_r', 32),
                            grid_size=getattr(cfg.scene, 'recinr_grid_size', 16),
                            H=H, W=W).to(dev)
        fwd = INRForwardModel(scene, motion, H, W, time_assignment_mode=_time_mode).to(dev)
    n_scene = sum(p.numel() for p in scene.parameters())
    print(f"Params: scene[{_scene_type}]={n_scene}, motion={motion.num_params()}")

    # ── 4.5. Optional: DGI warm-start for scene amplitudes ───────────
    # NOTE: dgi_reconstruct returns a z-scored image (mean≈0); init_from_image
    # expects [0,1], so normalize first or the amplitude match collapses to 0.
    _init_mode = getattr(cfg.scene, 'init_mode', 'random')
    if _scene_type != 'gaussian':
        # INR/grid/low-rank DGI pre-fit analog (fair only if the Gaussian run also
        # starts from DGI): fit the canonical field to the motion-blur DGI image.
        if _init_mode in ('dgi', 'dgi_adaptive'):
            scene.prefit(normalize_01(dgi_img), fwd.norm_grid(dev))
    elif _init_mode == 'dgi':
        scene.init_from_image(normalize_01(dgi_img))
    elif _init_mode == 'dgi_adaptive':
        scene.init_adaptive(dgi_img)     # centers ∝ |∇DGI|, amps from intensity
        scene.init_from_image(normalize_01(dgi_img))   # global amplitude scale

    # ── 5. Solve ──────────────────────────────────────────────
    t0 = time.time()
    history = []

    # Select TV prior (2D or 3D) for ADMM z-step
    _use_3dtv = getattr(cfg.solver, 'use_3dtv', False)
    _temp_w   = getattr(cfg.solver, 'temporal_tv_weight', 1.0)

    if cfg.solver.type == "admm":
        _s = cfg.solver
        _splitting_warmup, _motion_warmup = _admm_warmup_counts(_s)
        _prior_type = getattr(_s, 'prior_type', 'tv')
        if _prior_type == 'tv':
            prior = TVPrior3D(max_iter=50, temporal_weight=_temp_w) if _use_3dtv \
                    else TVPrior(max_iter=50)
        elif _prior_type == 'diffusion':
            from gsdiff.prior.diffusion import DiffusionPrior
            _dp = _s.diffusion_prior
            prior = DiffusionPrior(
                checkpoint_path=_dp.checkpoint,
                device=dev,
                denoise_steps=getattr(_dp, 'denoise_steps', 1),
                clamp_range=tuple(getattr(_dp, 'clamp_range', [0.0, 1.0])),
                sigma_start=getattr(_dp, 'sigma_start', 0.3),
                sigma_end=getattr(_dp, 'sigma_end', 0.05),
                renoise=getattr(_dp, 'renoise', False),
                ddim_spacing=getattr(_dp, 'ddim_spacing', 'linear'),
            )
            # Tell the prior how many z-step calls to expect
            _n_zsteps = _s.num_outer - _splitting_warmup
            prior.set_n_steps(_n_zsteps)
        else:
            raise ValueError(f"Unknown prior_type: {_prior_type}")
        solver = ADMMSolver(
            fwd, prior, pat_tr, y_tr, fidx_tr, tg,
            rho=_s.rho,
            tv_weight=getattr(_s, 'admm_tv_weight', _s.tv_weight),
            lr_scene=getattr(_s, 'admm_lr_scene', _s.lr_scene),
            lr_motion=getattr(_s, 'admm_lr_motion', _s.lr_motion),
            n_inner=_s.num_inner,
            rho_growth=_s.rho_growth,
            loss_norm=getattr(_s, 'loss_norm', 'zscore'),
            splitting_warmup_outer=_splitting_warmup,
            motion_warmup_outer=_motion_warmup,
            n_outer=_s.num_outer,
            soft_tv_weight=getattr(_s, 'admm_soft_tv_weight', 0.0),
            temporal_tv_weight=_temp_w if _use_3dtv else 0.0,
            hqs=getattr(_s, 'hqs', False),
            device=dev)

        L = cfg.solver.num_outer
        for l in range(1, L + 1):
            info = solver.step()
            sigma_used = None
            if _prior_type == 'diffusion':
                sigma_used = _record_sigma_used(info, prior)
            history.append(info)
            mp = motion.get_params_dict()
            # Per-pixel TV for readable numbers
            npix = T * H * W
            tv_pp = info['tv'] / npix
            _sigma_str = ""
            if sigma_used is not None:
                _sigma_str = f"  σ={sigma_used:.4f}"
            # Honest ADMM diagnostics: prim_res is only a residual post-warmup
            # (z≡0 during warmup ⇒ prim=video energy); consist/u_norm/dual_res are
            # the real coupling terms. Print them so the convergence is legible.
            _cons = (
                f"  consist={info.get('loss_consist', 0):.5f}  "
                f"u={info.get('u_norm', 0):.4f}"
                if not info.get('in_splitting_warmup', True)
                else "  (warmup)"
            )
            print(f"[{l:3d}/{L}]  data={info['loss_data']:.4f}  "
                  f"prim={info['prim_res']:.6f}  TV/px={tv_pp:.4f}  "
                  f"rho={info['rho']:.3f}{_sigma_str}{_cons}  "
                  f"v={[f'{v:.2f}' for v in mp['velocity']]}  "
                  f"ω={mp.get('omega', 0):.4f}")

    elif cfg.solver.type == "sgd":
        # Optional RED-diff single-loop diffusion regularizer
        _red_w = float(getattr(cfg.solver, 'red_weight', 0.0) or 0.0)
        _red_prior = None
        if _red_w > 0:
            from gsdiff.prior.diffusion import DiffusionPrior
            _dp = cfg.solver.diffusion_prior
            _red_prior = DiffusionPrior(
                checkpoint_path=_dp.checkpoint, device=dev, denoise_steps=1,
                clamp_range=tuple(getattr(_dp, 'clamp_range', [0.0, 1.0])),
                sigma_start=getattr(_dp, 'sigma_start', 0.3),
                sigma_end=getattr(_dp, 'sigma_end', 0.05))
            _red_prior.set_n_steps(cfg.solver.sgd_steps)
        _mwarm = int(getattr(cfg.solver, 'motion_warmup', 0))
        solver = SGDSolver(
            fwd, pat_tr, y_tr, fidx_tr, tg,
            tv_weight=cfg.solver.tv_weight,
            lr_scene=cfg.solver.lr_scene, lr_motion=cfg.solver.lr_motion,
            n_steps=cfg.solver.sgd_steps,
            loss_norm=getattr(cfg.solver, 'loss_norm', 'zscore'),
            temporal_tv_weight=_temp_w if _use_3dtv else 0.0,
            red_prior=_red_prior, red_weight=_red_w,
            freeze_motion=getattr(cfg.solver, 'freeze_motion', False),
            motion_warmup=_mwarm, device=dev)

        N = cfg.solver.sgd_steps
        for i in range(1, N + 1):
            info = solver.step()
            history.append(info)
            if i % 100 == 0 or i == N:
                mp = motion.get_params_dict()
                print(f"[{i:5d}/{N}]  data={info['loss_data']:.4f}  "
                      f"tv={info['tv']:.6f}  v={[f'{v:.2f}' for v in mp['velocity']]}  "
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

    # GT-free residuals (z-scored, matches the training loss convention)
    def _zs_residual(pred, target):
        zs = lambda v: (v - v.mean()) / (v.std() + 1e-8)
        return float(0.5 * torch.nn.functional.mse_loss(zs(pred), zs(target)))
    holdout_residual = None
    if hold_mask is not None:
        holdout_residual = _zs_residual(y_pred[hold_mask], y[hold_mask])
        train_residual = _zs_residual(y_pred[~hold_mask], y[~hold_mask])
        print(f"  Holdout residual: {holdout_residual:.6f}  (train: {train_residual:.6f})")
    else:
        train_residual = _zs_residual(y_pred, y)
    # Non-invasive eval set (fresh random patterns; training data untouched)
    eval_residual = None
    eval_residual_pf = None
    if data.eval_patterns is not None:
        ev_fidx_t = torch.tensor(data.eval_frame_idx, device=dev)
        with torch.no_grad():
            y_pred_ev, _ = fwd(torch.tensor(data.eval_patterns, device=dev),
                               ev_fidx_t, tg)
        y_ev = torch.tensor(data.eval_measurements, device=dev)
        eval_residual = _zs_residual(y_pred_ev, y_ev)
        # Per-frame z-scored variant: removes the per-frame affine gauge, the
        # same gauge the per-frame normalize_01 PSNR evaluation ignores. The
        # global variant penalizes frame-flux-trajectory error that stronger
        # priors trade for shape quality — misaligned with the paper metric.
        pf = []
        for f in range(T):
            m = ev_fidx_t == f
            if int(m.sum()) >= 4:
                pf.append(_zs_residual(y_pred_ev[m], y_ev[m]))
        eval_residual_pf = float(np.mean(pf)) if pf else None
        eval_residual_pf_text = (
            f"{eval_residual_pf:.6f}"
            if eval_residual_pf is not None
            else "n/a"
        )
        print(f"  Eval residual (Ke={len(data.eval_measurements)}): "
              f"global {eval_residual:.6f}  per-frame {eval_residual_pf_text}")

    if blind_method_child:
        me = motion.get_params_dict()
        output = ReconstructionOutput(
            dataset_identity_sha256=acquisition.dataset_identity_sha256,
            reconstruction=recon_np.astype(np.float32, copy=False),
            dgi=np.asarray(dgi_img, dtype=np.float32),
            estimated_motion_trajectory=_estimated_motion_trajectory(
                me, acquisition.time_grid
            ),
            frame_indices=np.arange(T, dtype=np.int64),
            time_grid=acquisition.time_grid,
            method_name=f"gsdiff-{cfg.solver.type}",
            method_metadata={
                "scene_type": _scene_type,
                "solver": cfg.solver.type,
                "train_residual": train_residual,
                "holdout_residual": holdout_residual,
                "measurement_probe_residual": eval_residual,
                "measurement_probe_residual_per_frame": eval_residual_pf,
            },
            execution_policy=execution_policy,
        )
        write_method_child_outputs(Path(out_dir), output, history=history)
        print(f"\nRaw method-child outputs saved to: {out_dir}")
        return None

    # ── 7. Evaluate ───────────────────────────────────────────
    print("\n=== Results ===")
    psnrs, mean_p = evaluate(data.gt_frames, recon_np, cfg.solver.type)
    _write_metrics_json(
        os.path.join(out_dir, "metrics.json"), data.gt_frames, recon_np
    )

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
        "est_accel": me.get('accel'),
        "est_beta": me.get('beta'),
        "est_lin": me.get('lin'),
        "motion_type": data.motion_type,
        "holdout_residual": holdout_residual,
        "train_residual": train_residual,
        "eval_residual": eval_residual,
        "eval_residual_pf": eval_residual_pf,
    })
    if truth_path is not None:
        results.update({
            "execution_class": execution_policy.execution_class,
            "truth_access": execution_policy.truth_access,
            "promotion_eligible": execution_policy.promotion_eligible,
        })
    _write_results_json(os.path.join(out_dir, "results.json"), results, history)

    # Checkpoint
    torch.save({"scene": scene.state_dict(), "motion": motion.state_dict(),
                "config": raw}, os.path.join(out_dir, "checkpoint.pt"))

    print(f"\nAll results saved to: {out_dir}")
    return mean_p


def legacy_main(argv=None):
    """Run the explicitly requested legacy compatibility workflow."""
    return _run_legacy_compatibility(argv)


def main(argv=None):
    """Dispatch strict by default.

    ``_run_legacy_compatibility`` remains reachable only through the explicit
    ``legacy_main`` boundary below.
    """
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
