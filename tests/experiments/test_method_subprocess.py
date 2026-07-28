from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from gsdiff.data._artifact_dataset import (
    blind_acquisition_spec,
    load_acquisition_data,
    save_acquisition_data,
)
from gsdiff.data._artifact_identity import array_descriptor
from gsdiff.data._artifact_io import artifact_sha256
from gsdiff.data._artifact_models import SPIAcquisitionData
from gsdiff.experiments.audit import validate_audit_log
from gsdiff.experiments.child_outputs import (
    validate_method_child_outputs_v2,
)
from gsdiff.experiments.execution import materialize_method_execution
from gsdiff.experiments.methods import (
    MethodResolutionRequest,
    derive_algorithm_seed,
    resolve_method_semantics,
    thaw_json,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path(r"D:\conda\envs\spi\python.exe")
TRUTH_SEEKING_CHILD = (
    REPO_ROOT
    / "tests"
    / "experiments"
    / "fixtures"
    / "truth_seeking_child.py"
)


def _tiny_immutable_acquisition() -> SPIAcquisitionData:
    rng = np.random.default_rng(2026072801)
    T, H, W, rows = 4, 8, 8, 3
    patterns = rng.integers(
        0, 2, size=(T * rows, H, W), dtype=np.int8
    ).astype(np.float32)
    frame_indices = np.repeat(np.arange(T, dtype=np.int64), rows)
    source = rng.random((T, H, W), dtype=np.float32)
    measurements = np.einsum(
        "khw,khw->k",
        patterns,
        source[frame_indices],
    ).astype(np.float32)
    holdout_patterns = rng.integers(
        0, 2, size=(T, H, W), dtype=np.int8
    ).astype(np.float32)
    holdout_frame_indices = np.arange(T, dtype=np.int64)
    holdout_measurements = np.einsum(
        "khw,khw->k",
        holdout_patterns,
        source[holdout_frame_indices],
    ).astype(np.float32)
    arrays = {
        "patterns": patterns,
        "measurements": measurements,
        "frame_indices": frame_indices,
        "time_grid": np.linspace(0.0, 1.0, T, dtype=np.float64),
        "holdout_patterns": holdout_patterns,
        "holdout_measurements": holdout_measurements,
        "holdout_frame_indices": holdout_frame_indices,
    }
    return SPIAcquisitionData(
        dataset_identity_sha256="b" * 64,
        **arrays,
        H=H,
        W=W,
        T=T,
        K=patterns.shape[0],
        holdout_K=holdout_patterns.shape[0],
        acquisition={
            "pattern_family": "bernoulli",
            "pattern_values": [0, 1],
            "pattern_order": "sequential",
            "time_assignment": "uniform",
            "holdout_pattern_family": "uniform-random",
            "noise_convention": "detector-absolute",
            "noise_sigma_absolute": 0.0,
        },
        array_descriptors={
            name: array_descriptor(value)
            for name, value in arrays.items()
        },
    )


def _resolve_smoke(method_id: str, acquisition: SPIAcquisitionData):
    return resolve_method_semantics(
        method_id,
        method_config_id="smoke-default-v1",
        base_config=(
            {"gaussian_count": 1000}
            if method_id.startswith("gsdiff_")
            else {}
        ),
        measurements_metadata={
            "H": acquisition.H,
            "W": acquisition.W,
            "T": acquisition.T,
            "K": acquisition.K,
            "holdout_K": acquisition.holdout_K,
        },
        execution_profile="controller-cpu-smoke-v1",
    )


def _materialize_real(
    tmp_path: Path,
    *,
    method_id: str,
    stage_name: str,
):
    acquisition = _tiny_immutable_acquisition()
    upstream_root = tmp_path / "upstream-dataset"
    upstream_root.mkdir(exist_ok=True)
    measurements_source = upstream_root / "measurements.npz"
    if not measurements_source.exists():
        save_acquisition_data(acquisition, measurements_source)
    method = _resolve_smoke(method_id, acquisition)
    base_config = (
        {"gaussian_count": 1000}
        if method_id.startswith("gsdiff_")
        else {}
    )
    measurements_metadata = {
        "H": acquisition.H,
        "W": acquisition.W,
        "T": acquisition.T,
        "K": acquisition.K,
        "holdout_K": acquisition.holdout_K,
    }
    algorithm_seed = derive_algorithm_seed(
        cell_seed=29,
        dataset_identity_sha256=acquisition.dataset_identity_sha256,
        method_id=method.method_id,
        method_config_sha256=method.method_config_sha256,
    )
    execution = materialize_method_execution(
        method,
        resolution_request=MethodResolutionRequest(
            requested_method_id=method_id,
            requested_method_config_id="smoke-default-v1",
            base_config=base_config,
            measurements_metadata=measurements_metadata,
            requested_execution_profile="controller-cpu-smoke-v1",
        ),
        stage_root=tmp_path / stage_name,
        measurements_source=measurements_source,
        measurements_file_sha256=artifact_sha256(
            measurements_source
        ),
        dataset_identity_sha256=acquisition.dataset_identity_sha256,
        expected_acquisition_spec=blind_acquisition_spec(acquisition),
        algorithm_seed=algorithm_seed,
        checkpoint_store={},
        python_executable=PYTHON,
        source_root=REPO_ROOT,
        requested_runtime_device="cpu",
    )
    return (
        execution,
        acquisition,
        method,
        algorithm_seed,
        upstream_root,
        measurements_source,
    )


def _run_materialized(execution, argv=None, *, timeout=120):
    with (
        execution.stdout_path.open("wb") as stdout,
        execution.stderr_path.open("wb") as stderr,
    ):
        return subprocess.run(
            execution.argv if argv is None else argv,
            cwd=execution.cwd,
            env=dict(execution.env),
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            shell=False,
            check=False,
            timeout=timeout,
        )


def _audit_events(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8", errors="strict"
        ).splitlines()
    ]


def _string_leaves(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _string_leaves(key)
            yield from _string_leaves(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _string_leaves(item)


def _normalized_path_text(value: str) -> str:
    return os.path.normcase(value).replace("\\", "/")


def _assert_path_not_exposed(value, forbidden: Path) -> None:
    forbidden_text = _normalized_path_text(str(forbidden))
    for leaf in _string_leaves(value):
        assert forbidden_text not in _normalized_path_text(leaf), leaf


def test_path_exposure_guard_checks_unescaped_nested_windows_strings():
    marker = Path(r"D:\Research\gsdiff_spi")
    with pytest.raises(AssertionError):
        _assert_path_not_exposed(
            {"nested": [{"cwd": str(marker / "stage" / "code")}]},
            marker,
        )


def _create_directory_reparse(link: Path, target: Path) -> None:
    if os.name != "nt":
        try:
            link.symlink_to(target, target_is_directory=True)
        except (OSError, NotImplementedError) as error:
            pytest.skip(f"directory symlink unavailable: {error}")
        return
    completed = subprocess.run(
        [
            "cmd.exe",
            "/d",
            "/c",
            "mklink",
            "/J",
            str(link),
            str(target),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=False,
        check=False,
        timeout=10,
    )
    if completed.returncode != 0 or not os.path.lexists(link):
        pytest.skip(
            "directory junction unavailable: "
            + completed.stdout.decode("utf-8", errors="replace")
        )
    assert getattr(os.lstat(link), "st_file_attributes", 0) & 0x400


@pytest.mark.parametrize("method_id", ["dgi", "gsdiff_tv"])
def test_real_strict_method_subprocess_is_blind_and_writes_v2(
    tmp_path,
    method_id,
):
    (
        execution,
        acquisition,
        method,
        algorithm_seed,
        upstream_root,
        _measurements_source,
    ) = _materialize_real(
        tmp_path,
        method_id=method_id,
        stage_name=f"stage-{method_id}",
    )

    completed = _run_materialized(execution)

    assert completed.returncode == 0, execution.stderr_path.read_text(
        encoding="utf-8", errors="strict"
    )
    assert {path.name for path in execution.child_output_dir.iterdir()} == {
        "reconstruction.npz",
        "method-info.json",
    }
    staged_acquisition = load_acquisition_data(
        execution.measurements_path,
        expected_dataset_identity_sha256=(
            acquisition.dataset_identity_sha256
        ),
        expected_acquisition_spec=execution.expected_acquisition_spec,
    )
    validate_method_child_outputs_v2(
        execution.child_output_dir,
        expected_method=method,
        expected_acquisition=staged_acquisition,
        expected_dataset_identity_sha256=(
            acquisition.dataset_identity_sha256
        ),
        expected_measurements_file_sha256=artifact_sha256(
            execution.measurements_path
        ),
        expected_algorithm_seed=algorithm_seed,
    )
    summary = validate_audit_log(
        execution.audit_log_path,
        expected_policy_sha256=execution.audit_policy_sha256,
    )
    assert summary["terminal_status"] == "success"
    events = _audit_events(execution.audit_log_path)
    assert [event["sequence"] for event in events] == list(
        range(len(events))
    )
    assert not any(
        event.get("decision") == "deny" for event in events
    )
    assert not any(
        str(event.get("operation", "")).startswith("socket.")
        or event.get("operation")
        in {"socket-unknown", "audit-socket-poisoned"}
        for event in events
    )
    assert execution.stdout_path.parent != execution.child_output_dir
    assert execution.stderr_path.parent != execution.child_output_dir
    assert method.publication_eligible is False
    assert method.selection_eligible is False
    assert method.promotion_eligible is False
    assert method.convergence_status == (
        "smoke-only/not-convergence-assessed"
    )
    stdout_lines = execution.stdout_path.read_text(
        encoding="utf-8", errors="strict"
    ).splitlines()
    assert stdout_lines == [
        f"completed method child: {method.method_id}",
        str(execution.child_output_dir / "reconstruction.npz"),
        str(execution.child_output_dir / "method-info.json"),
    ]
    assert execution.stderr_path.read_text(
        encoding="utf-8", errors="strict"
    ) == ""

    child_visible = {
        "argv": list(execution.argv),
        "env": dict(execution.env),
        "cwd": str(execution.cwd),
        "semantic_config": thaw_json(method.semantic_config),
    }
    assert all(
        "truth" not in leaf.lower()
        for leaf in _string_leaves(child_visible)
    )
    for forbidden_root in (upstream_root, REPO_ROOT):
        _assert_path_not_exposed(child_visible, forbidden_root)
        _assert_path_not_exposed(events, forbidden_root)
    forbidden_paths = (
        str(upstream_root).lower(),
        "gsdiff\\evaluation",
        "gsdiff/evaluation",
        "_artifact_truth.py",
        "_evaluation.py",
    )
    for event in events:
        accessed = str(event.get("resolved_path", "")).lower()
        assert not any(value in accessed for value in forbidden_paths)


def _attack_argv(
    execution,
    *,
    action: str,
    target: Path | None = None,
) -> tuple[str, ...]:
    entrypoint = execution.cwd / "scripts" / "run_baselines.py"
    entrypoint.write_bytes(TRUTH_SEEKING_CHILD.read_bytes())
    delimiter = execution.argv.index("--")
    child_arguments = ["--audit-action", action]
    if target is not None:
        child_arguments.extend(["--target", str(target)])
    return (
        *execution.argv[: delimiter + 1],
        *child_arguments,
    )


@pytest.mark.parametrize(
    ("attack", "action", "expected_operation"),
    [
        ("sibling-read", "read", "open"),
        ("absolute-upstream-read", "read", "open"),
        ("directory-list", "listdir", "os.listdir"),
        ("directory-scan", "scandir", "os.scandir"),
        ("nested-subprocess", "subprocess", "subprocess.Popen"),
    ],
)
def test_truth_seeking_method_subprocess_is_denied_exactly(
    tmp_path,
    attack,
    action,
    expected_operation,
):
    (
        execution,
        _acquisition,
        _method,
        _algorithm_seed,
        upstream_root,
        measurements_source,
    ) = _materialize_real(
        tmp_path,
        method_id="dgi",
        stage_name=f"attack-{attack}",
    )
    if attack == "sibling-read":
        sibling_secret = execution.cwd.parent / "sibling-secret.txt"
        sibling_secret.write_text("secret", encoding="utf-8")
        target = Path("..") / sibling_secret.name
        expected_resolved_path = sibling_secret.resolve()
        assert not target.is_absolute()
    elif attack == "absolute-upstream-read":
        target = measurements_source.resolve()
        expected_resolved_path = target
    elif attack in {"directory-list", "directory-scan"}:
        target = upstream_root.resolve()
        expected_resolved_path = target
    elif attack == "nested-subprocess":
        target = None
        expected_resolved_path = None
    else:
        raise AssertionError(attack)

    completed = _run_materialized(
        execution,
        _attack_argv(execution, action=action, target=target),
    )

    assert completed.returncode != 0
    events = _audit_events(execution.audit_log_path)
    assert events[0] == {
        **events[0],
        "operation": "hook-installed",
        "decision": "allow",
        "policy_sha256": execution.audit_policy_sha256,
    }
    assert [event["sequence"] for event in events] == list(
        range(len(events))
    )
    denied = [
        event
        for event in events
        if event.get("decision") == "deny"
    ]
    assert len(denied) == 1
    assert denied[0]["operation"] == expected_operation
    if expected_resolved_path is not None:
        assert denied[0]["resolved_path"] == str(expected_resolved_path)
    assert events[-1]["operation"] == "bootstrap-finished"
    assert events[-1]["status"] == "error"
    with pytest.raises(
        ValueError,
        match="denied|unsuccessful|terminal",
    ):
        validate_audit_log(
            execution.audit_log_path,
            expected_policy_sha256=execution.audit_policy_sha256,
        )


def test_truth_seeking_method_subprocess_reparse_escape_is_denied(
    tmp_path,
):
    (
        execution,
        _acquisition,
        _method,
        _algorithm_seed,
        _upstream_root,
        measurements_source,
    ) = _materialize_real(
        tmp_path,
        method_id="dgi",
        stage_name="attack-reparse",
    )
    link = execution.cwd / "reparse-escape.npz"
    try:
        os.symlink(measurements_source.resolve(), link)
    except (OSError, NotImplementedError) as error:
        if os.path.lexists(link):
            link.unlink()
        junction = execution.cwd / "reparse-escape"
        _create_directory_reparse(junction, measurements_source.parent)
        link = junction / measurements_source.name

    completed = _run_materialized(
        execution,
        _attack_argv(execution, action="read", target=link),
    )

    assert completed.returncode != 0
    events = _audit_events(execution.audit_log_path)
    denied = [
        event
        for event in events
        if event.get("decision") == "deny"
    ]
    assert len(denied) == 1
    assert denied[0]["operation"] == "open"
    assert events[-1]["operation"] == "bootstrap-finished"
    assert events[-1]["status"] == "error"
