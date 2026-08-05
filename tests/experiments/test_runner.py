from __future__ import annotations

import contextlib
from dataclasses import fields, replace
import ctypes
import hashlib
import importlib.util
import inspect
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import time
from types import MappingProxyType
import uuid

import numpy as np
import pytest

import gsdiff.experiments.child_outputs as child_outputs_module
import gsdiff.experiments.manifest as manifest_module
import gsdiff.experiments._owned_tree as owned_tree_module
import gsdiff.experiments.runner as runner_module
import gsdiff.data._artifact_persistence as artifact_persistence_module

from gsdiff.data._artifact_identity import array_descriptor
from gsdiff.data._artifact_models import SPIAcquisitionData
from gsdiff.data.artifacts import (
    blind_acquisition_spec,
    build_dataset_manifest,
    build_dataset_payloads,
    dataset_manifest_bytes,
    publish_dataset,
    verify_dataset_directory,
)
from gsdiff.evaluation.metrics import evaluate_video_global_affine
from gsdiff.experiments.dataset_binding import dataset_measurement_record
from gsdiff.experiments.execution import _materialization_identity_documents

from gsdiff.experiments.identity import (
    build_run_identity,
    canonical_json_bytes,
    requirements_dependencies_sha256,
    resolved_config_sha256,
)
from gsdiff.experiments.manifest import (
    build_aggregate_index,
    build_manifest,
    load_complete_manifest,
    validate_manifest,
)
from gsdiff.experiments.methods import (
    CheckpointRequirement,
    MethodResolutionRequest,
    derive_algorithm_seed,
    resolve_method_semantics,
)
from gsdiff.experiments.protocol import ExperimentCell
from gsdiff.experiments.protocol import expand_cells, load_protocol
from gsdiff.experiments.runner import (
    RunOutcome,
    RunExecutionPlan,
    RunRequest,
    _claim_identity,
    _preflight_execution,
    _remove_owned_work,
    _require_same_filesystem,
    _validate_authoritative_request,
    _verify_claim,
    reusable_run,
    run_request,
)
from gsdiff.experiments.source_snapshot import (
    materialize_source_snapshot,
    selected_source_evidence,
)


ROOT = Path(__file__).resolve().parents[2]
HASH = "a" * 64
_ENVIRONMENT_LOCK = json.loads(
    (ROOT / "docs/reproducibility/environment-lock.json").read_text("utf-8")
)
ENVIRONMENT_HASH = _ENVIRONMENT_LOCK["fingerprint_sha256"]
_COMPLETE_GENERATED_DATASET = None
_FAILURE_GENERATED_DATASET = None
SOURCE_ROOTS = (
    Path("gsdiff"),
    Path("scripts"),
    Path("configs"),
    Path("schemas"),
    Path("assets"),
    Path("train.py"),
    Path("requirements-lock.txt"),
    Path("docs/reproducibility/environment-lock.json"),
)
_COMPLETE_RUNTIME = {
    "python": _ENVIRONMENT_LOCK["fingerprint"]["python"]["version"],
    "pytorch": _ENVIRONMENT_LOCK["fingerprint"]["pytorch"]["version"],
    "cuda": _ENVIRONMENT_LOCK["fingerprint"]["pytorch"]["cuda_build"] or "",
    "gpu": "",
    "os": _ENVIRONMENT_LOCK["fingerprint"]["platform"]["platform"],
}
_PYTHON_EXECUTABLE_SHA256 = hashlib.sha256(
    Path(__import__("sys").executable).read_bytes()
).hexdigest()


def _request_plan_with_repo_snapshot(request, plan, artifact_root: Path):
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()
    snapshot = materialize_source_snapshot(
        ROOT,
        artifact_root,
        commit,
        SOURCE_ROOTS,
    )
    inventory, digest = selected_source_evidence(snapshot)
    base_config = dict(plan.config_resolved)
    base_config.pop("runner_execution", None)
    verified = verify_dataset_directory(request.dataset_dir)
    algorithm_seed = derive_algorithm_seed(
        cell_seed=request.cell.seed,
        dataset_identity_sha256=verified.dataset_identity_sha256,
        method_id=request.method.method_id,
        method_config_sha256=request.method.method_config_sha256,
    )
    _materialized_config, predicted_logical = _materialization_identity_documents(
        method=request.method,
        dataset_identity_sha256=verified.dataset_identity_sha256,
        measurements_file_sha256=verified.payload_evidence["measurements.npz"].sha256,
        expected_acquisition_spec=blind_acquisition_spec(verified.acquisition),
        algorithm_seed=algorithm_seed,
        source_inventory=[dict(item) for item in inventory],
        requested_runtime_device=plan.requested_runtime_device,
    )
    config_resolved = runner_module._identity_bound_config(
        base_config,
        requested_runtime_device=plan.requested_runtime_device,
        source_snapshot_sha256=snapshot.snapshot_sha256,
        source_projection_sha256=digest,
        compute_cap=request.method.semantic_config["compute_cap"],
        materialization_logical_sha256=hashlib.sha256(
            canonical_json_bytes(predicted_logical)
        ).hexdigest(),
        method_info_contract=(
            child_outputs_module.build_method_info_contract_v1(
                request.method,
                blind_acquisition_spec(verified.acquisition),
            )
        ),
        dataset_input_contract=runner_module.build_dataset_input_contract(
            verified
        ),
        runtime_contract=plan.runtime_metadata,
        python_executable_sha256=_PYTHON_EXECUTABLE_SHA256,
    )
    payload = dict(request.identity.payload())
    payload["code_commit"] = commit
    payload["config_sha256"] = resolved_config_sha256(config_resolved)
    identity = build_run_identity(**payload)
    updated_plan = replace(
        plan,
        identity=identity,
        registry_path=snapshot.root / "configs/protocols/methods-v1.yaml",
        source_root=snapshot.root,
        code_commit=commit,
        config_resolved=config_resolved,
        source_snapshot=snapshot,
        expected_source_inventory=inventory,
        expected_source_snapshot_sha256=digest,
    )
    return replace(
        request,
        identity=identity,
        execution_plan=updated_plan,
    ), updated_plan


def _claim_once_worker(claims_dir, identity, execution_marker, queue):
    with _claim_identity(claims_dir, identity):
        if execution_marker.exists():
            queue.put("cached")
            return
        execution_marker.write_text(str(__import__("os").getpid()), encoding="ascii")
        time.sleep(0.4)
        queue.put("complete")


def _hold_claim_worker(claims_dir, identity, ready):
    with _claim_identity(claims_dir, identity):
        ready.set()
        time.sleep(60)


def _hold_claim_for_worker(claims_dir, identity, ready, hold_seconds):
    with _claim_identity(claims_dir, identity):
        ready.set()
        time.sleep(hold_seconds)


def _hard_kill_run_worker(tmp_root, artifact_root, started, survived, queue):
    request, plan = _real_request_and_plan(Path(tmp_root))
    real_materialize = runner_module.materialize_method_execution

    def delayed_materialize(*args, **kwargs):
        materialized = real_materialize(*args, **kwargs)
        code = (
            "import sys,time\n"
            "from pathlib import Path\n"
            "Path(sys.argv[1]).write_text('started', encoding='ascii')\n"
            "time.sleep(5)\n"
            "Path(sys.argv[2]).write_text('survived', encoding='ascii')\n"
        )
        return replace(
            materialized,
            argv=(
                str(Path(__import__("sys").executable)),
                "-c",
                code,
                str(started),
                str(survived),
            ),
        )

    runner_module.materialize_method_execution = delayed_materialize
    queue.put(request.identity.identity_sha256)
    run_request(request, Path(artifact_root))


def _hard_kill_two_jobs_worker(
    tmp_root,
    method_pid_marker,
    sampler_pid_marker,
):
    sampler_path = str(sampler_pid_marker).replace("'", "''")
    runner_module._POWERSHELL_SAMPLER_SCRIPT = (
        "$targetPid=[Int32]$args[0];"
        f"[IO.File]::WriteAllText('{sampler_path}',[String]$PID);"
        "[Console]::Out.WriteLine('READY');[Console]::Out.Flush();"
        "while($true){"
        "[Console]::Out.WriteLine((\"R`tpid_{0}_luid_0x0_phys_0`t0`t0\" -f $targetPid));"
        "[Console]::Out.WriteLine('E');[Console]::Out.Flush();"
        "Start-Sleep -Milliseconds 100}"
    )
    code = (
        "import os,sys,time\n"
        "from pathlib import Path\n"
        "Path(sys.argv[1]).write_text(str(os.getpid()),encoding='ascii')\n"
        "time.sleep(60)\n"
    )
    root = Path(tmp_root)
    with (root / "hard-kill-stdout.log").open("wb") as stdout, (
        root / "hard-kill-stderr.log"
    ).open("wb") as stderr:
        runner_module._run_method_child(
            argv=(
                str(Path(__import__("sys").executable)),
                "-c",
                code,
                str(method_pid_marker),
            ),
            cwd=root,
            env=dict(os.environ),
            stdout_stream=stdout,
            stderr_stream=stderr,
            requested_runtime_device="cuda:0",
            compute_cap=runner_module._ComputeCap(
                120,
                1024,
                "ineligible-retain-artifacts",
            ),
        )


def _windows_pid_is_running(process_id: int) -> bool:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    open_process = kernel32.OpenProcess
    open_process.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    open_process.restype = ctypes.c_void_p
    handle = open_process(0x1000, 0, process_id)
    if not handle:
        return False
    try:
        exit_code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            raise ctypes.WinError(ctypes.get_last_error())
        return exit_code.value == 259
    finally:
        kernel32.CloseHandle(handle)


def _identity(**overrides: object):
    values: dict[str, object] = {
        "execution_class": "blind_method_child",
        "scientific_contract_id": "gsdiff-sim-v1",
        "scientific_contract_sha256": HASH,
        "method_id": "gsdiff_tv",
        "target_id": "tank",
        "motion_id": "trans",
        "seed": 7,
        "config_sha256": hashlib.sha256(b"{}").hexdigest(),
        "dataset_identity_sha256": "c" * 64,
        "assets_sha256": {"target": "d" * 64},
        "checkpoints_sha256": {},
        "code_commit": "b" * 40,
        "dirty_worktree": False,
        "source_tree_hash": None,
        "dependencies_sha256": requirements_dependencies_sha256(
            ROOT / "requirements-lock.txt"
        ),
        "environment_lock_sha256": ENVIRONMENT_HASH,
        "metric_version": "metrics-v1",
    }
    values.update(overrides)
    return build_run_identity(**values)


def _synthetic_source_hashes() -> tuple[str, str]:
    inventory = [
        {
            "path": "gsdiff/module.py",
            "mode": "100644",
            "git_blob": "1" * 40,
            "sha256": "2" * 64,
            "size_bytes": 1,
        }
    ]
    snapshot = hashlib.sha256(
        canonical_json_bytes(
            {
                "schema": "source-snapshot-identity-v1",
                "commit": "b" * 40,
                "inventory": inventory,
            }
        )
    ).hexdigest()
    projection = hashlib.sha256(
        canonical_json_bytes(
            [{"path": "gsdiff/module.py", "sha256": "2" * 64}]
        )
    ).hexdigest()
    return snapshot, projection


def _synthetic_complete_config(method=None) -> dict[str, object]:
    snapshot, projection = _synthetic_source_hashes()
    method = method or _complete_fixture_method()
    compute_cap = dict(method.semantic_config["compute_cap"])
    logical = _synthetic_materialization_logical(method)
    return runner_module._identity_bound_config(
        {"method_config_sha256": method.method_config_sha256},
        requested_runtime_device="cpu",
        source_snapshot_sha256=snapshot,
        source_projection_sha256=projection,
        compute_cap=compute_cap,
        materialization_logical_sha256=hashlib.sha256(
            canonical_json_bytes(logical)
        ).hexdigest(),
        method_info_contract=_synthetic_method_info_contract(method),
        dataset_input_contract=_complete_dataset_input_contract(),
        runtime_contract=_COMPLETE_RUNTIME,
        python_executable_sha256=_PYTHON_EXECUTABLE_SHA256,
    )


def _synthetic_method_info_contract(method=None) -> dict[str, object]:
    method = method or _complete_fixture_method()
    return child_outputs_module.build_method_info_contract_v1(
        method,
        blind_acquisition_spec(_complete_fixture_acquisition()),
    )


def _synthetic_materialization_logical(
    method=None,
    requested_runtime_device="cpu",
) -> dict[str, object]:
    method = method or _complete_fixture_method()
    acquisition = _complete_fixture_acquisition()
    seed = derive_algorithm_seed(
        cell_seed=7,
        dataset_identity_sha256=acquisition.dataset_identity_sha256,
        method_id=method.method_id,
        method_config_sha256=method.method_config_sha256,
    )
    _config, logical = _materialization_identity_documents(
        method=method,
        dataset_identity_sha256=acquisition.dataset_identity_sha256,
        measurements_file_sha256=_complete_dataset_payload_hashes()[
            "measurements.npz"
        ],
        expected_acquisition_spec=blind_acquisition_spec(acquisition),
        algorithm_seed=seed,
        source_inventory=[{"path": "gsdiff/module.py", "sha256": "2" * 64}],
        requested_runtime_device=requested_runtime_device,
    )
    return logical


def _complete_identity(**overrides: object):
    dataset_spec = _complete_generated_dataset().dataset_identity_spec
    method_id = str(overrides.get("method_id", "dgi"))
    method = _complete_fixture_method(method_id)
    overrides.setdefault("method_id", method_id)
    overrides.setdefault(
        "scientific_contract_id",
        dataset_spec["scientific_contract"]["id"],
    )
    overrides.setdefault(
        "scientific_contract_sha256",
        dataset_spec["scientific_contract"]["sha256"],
    )
    overrides.setdefault("target_id", dataset_spec["target"]["id"])
    overrides.setdefault("motion_id", dataset_spec["motion"]["id"])
    overrides.setdefault("seed", dataset_spec["seed"])
    overrides.setdefault(
        "dataset_identity_sha256",
        _complete_generated_dataset().dataset_identity_sha256,
    )
    overrides.setdefault(
        "assets_sha256",
        runner_module._identity_asset_mapping(
            _complete_generated_dataset().resolved_generator_config["target"]
        ),
    )
    overrides.setdefault(
        "checkpoints_sha256",
        {
            item.logical_id: item.sha256
            for item in method.checkpoint_requirements
        },
    )
    overrides.setdefault(
        "config_sha256",
        resolved_config_sha256(_synthetic_complete_config(method)),
    )
    return _identity(**overrides)


def _complete_fixture_method(method_id="dgi"):
    return resolve_method_semantics(
        method_id,
        method_config_id="default",
        base_config=(
            {"gaussian_count": 1000}
            if method_id.startswith("gsdiff_")
            else {}
        ),
        measurements_metadata={},
        execution_profile="publication-v1",
    )


def _complete_fixture_acquisition() -> SPIAcquisitionData:
    return _complete_generated_dataset().acquisition


def _complete_generated_dataset():
    global _COMPLETE_GENERATED_DATASET
    if _COMPLETE_GENERATED_DATASET is None:
        builder = _load_dataset_builder()
        plan = builder.plan_campaign_datasets(
            repo_root=ROOT,
            protocol_path=ROOT / "configs/protocols/pilot-v1.yaml",
            runtime={
                "dependencies_sha256": "1" * 64,
                "environment_lock_sha256": "2" * 64,
            },
            generator_commit="3" * 40,
        )
        _COMPLETE_GENERATED_DATASET = builder.generate_corrected_dataset(
            **plan.requests[0].generation_arguments()
        )
    return _COMPLETE_GENERATED_DATASET


def _complete_dataset_payload_hashes() -> dict[str, str]:
    return {
        name: hashlib.sha256(payload).hexdigest()
        for name, payload in build_dataset_payloads(
            _complete_generated_dataset()
        ).items()
    }


def _complete_dataset_input_contract() -> dict[str, object]:
    generated = _complete_generated_dataset()
    payloads = build_dataset_payloads(generated)
    manifest = build_dataset_manifest(generated, payloads)
    return {
        "dataset_manifest_sha256": hashlib.sha256(
            dataset_manifest_bytes(manifest)
        ).hexdigest(),
        "measurements_file_sha256": hashlib.sha256(
            payloads["measurements.npz"]
        ).hexdigest(),
        "evaluation_truth_file_sha256": hashlib.sha256(
            payloads["evaluation-truth.npz"]
        ).hexdigest(),
        "measurement": dataset_measurement_record(manifest),
    }


def _fixture_history_metrics(method, index: int, budget: int) -> dict[str, float]:
    value = float(budget - index + 1)
    if method.method_id in {"static_cs", "perframe_cs", "monin"}:
        return {
            "data_fidelity": value,
            "primal_residual": value,
            "dual_residual": value,
        }
    if method.method_id == "tv3d":
        return {"data_fidelity": value}
    if method.method_id == "gidc3dtv":
        return {
            "loss": value,
            "data_fidelity": value,
            "learning_rate": 0.001,
        }
    if method.method_id == "recinr":
        metrics = {"loss": value, "learning_rate": 0.001}
        if index > method.semantic_config["solver"]["warm_steps"]:
            metrics["data_fidelity"] = value
        return metrics
    if method.method_id in {"siren", "recinr_se2"}:
        metrics = {"loss": value, "data_fidelity": value}
    elif method.method_id in {"gsdiff_tv", "gsdiff_diffusion"}:
        metrics = {
            "loss": value,
            "primal_residual": value,
            "dual_residual": value,
        }
    else:
        return {}
    if index == budget:
        metrics["objective"] = value
    return metrics


def _write_complete_run(artifact_root: Path, identity=None) -> Path:
    identity = identity or _complete_identity()
    publication = publish_dataset(
        artifact_root,
        _complete_generated_dataset(),
    )
    verified_dataset = publication.verified
    assert (
        identity.payload()["dataset_identity_sha256"]
        == verified_dataset.dataset_identity_sha256
    )
    run_dir = artifact_root / "runs" / identity.identity_sha256
    outputs = run_dir / "outputs"
    evidence = run_dir / "evidence"
    outputs.mkdir(parents=True)
    evidence.mkdir()
    method = _complete_fixture_method(identity.payload()["method_id"])
    acquisition = verified_dataset.acquisition
    method_info_contract = _synthetic_method_info_contract(method)
    algorithm_seed = derive_algorithm_seed(
        cell_seed=7,
        dataset_identity_sha256=acquisition.dataset_identity_sha256,
        method_id=method.method_id,
        method_config_sha256=method.method_config_sha256,
    )
    selection_contract = method_info_contract["selection"]
    candidate_grid = selection_contract["candidate_grid"]
    if candidate_grid is None:
        selected_hyperparameters = None
        selection = None
    else:
        selection = {
            "formula_id": "heldout-normalized-l2-v1",
            "candidate_grid": candidate_grid,
            "selected_candidate": candidate_grid[0],
            "rows": [
                {
                    "candidate": candidate,
                    "formula_id": "heldout-normalized-l2-v1",
                    "numerator": float(index + 1),
                    "denominator": 1.0,
                    "value": float(index + 1),
                }
                for index, candidate in enumerate(candidate_grid)
            ],
        }
        keys = selection_contract["selected_hyperparameter_keys"]
        assert keys == ["lambda"]
        selected_hyperparameters = {"lambda": candidate_grid[0]}
    history_contract = method_info_contract["history"]
    history = tuple(
        {
            "kind": history_contract["kind"],
            history_contract["index_field"]: index + 1,
            **_fixture_history_metrics(
                method,
                index + 1,
                history_contract["observed_count"],
            ),
        }
        for index in range(history_contract["observed_count"])
    )
    motion_contract = method_info_contract["motion_estimate"]
    native_iteration = method_info_contract["native_iteration"]
    result = child_outputs_module.MethodChildResult(
        method_id=method.method_id,
        reconstruction=np.ones(
            (acquisition.T, acquisition.H, acquisition.W),
            dtype=np.float32,
        ),
        estimated_motion_trajectory=(
            np.zeros((4, 3), dtype=np.float32)
            if motion_contract["presence"] == "required"
            else None
        ),
        dgi=(
            np.ones((acquisition.H, acquisition.W), dtype=np.float32)
            if method_info_contract["auxiliary_arrays"]["dgi"] == "required"
            else None
        ),
        info={
            "parameter_count": method_info_contract[
                "expected_parameter_count"
            ],
            "native_iteration_unit": native_iteration["unit"],
            "native_iteration_budget": native_iteration["budget"],
            "convergence_status": method_info_contract[
                "convergence_status"
            ],
            "selected_hyperparameters": selected_hyperparameters,
            "selection": selection,
            "checkpoint_hashes": method_info_contract["checkpoints"],
            "native_motion_model": motion_contract["native_model"],
        },
        history=history,
    )
    child_outputs_module.write_method_child_outputs_v2(
        outputs,
        method=method,
        acquisition=acquisition,
        measurements_file_sha256=verified_dataset.payload_evidence[
            "measurements.npz"
        ].sha256,
        algorithm_seed=algorithm_seed,
        result=result,
        child_started_at_utc="2026-08-04T00:00:00Z",
        child_finished_at_utc="2026-08-04T00:00:01Z",
    )
    audit_policy_sha256 = "5" * 64
    audit_log = b"\n".join(
        (
            canonical_json_bytes(
                {
                    "sequence": 0,
                    "timestamp_utc": "2026-08-04T00:00:00.000000Z",
                    "operation": "hook-installed",
                    "decision": "allow",
                    "policy_sha256": audit_policy_sha256,
                }
            ),
            canonical_json_bytes(
                {
                    "sequence": 1,
                    "timestamp_utc": "2026-08-04T00:00:01.000000Z",
                    "operation": "bootstrap-finished",
                    "decision": "allow",
                    "status": "success",
                }
            ),
            b"",
        )
    )
    source_inventory = [
        {
            "path": "gsdiff/module.py",
            "mode": "100644",
            "git_blob": "1" * 40,
            "sha256": "2" * 64,
            "size_bytes": 1,
        }
    ]
    source_snapshot_sha256 = hashlib.sha256(
        canonical_json_bytes(
            {
                "schema": "source-snapshot-identity-v1",
                "commit": "b" * 40,
                "inventory": source_inventory,
            }
        )
    ).hexdigest()
    selected_source = [
        {"path": "gsdiff/module.py", "sha256": "2" * 64}
    ]
    selected_source_sha256 = hashlib.sha256(
        canonical_json_bytes(selected_source)
    ).hexdigest()
    payloads = {
        "outputs/reconstruction.npz": (outputs / "reconstruction.npz").read_bytes(),
        "outputs/method-info.json": (outputs / "method-info.json").read_bytes(),
        "outputs/stdout.log": b"stdout\n",
        "outputs/stderr.log": b"",
        "outputs/metrics.json": canonical_json_bytes(
            evaluate_video_global_affine(
                verified_dataset.truth.gt_frames,
                result.reconstruction,
            )
        ),
        "resolved-config.json": canonical_json_bytes(
            _synthetic_complete_config(method)
        ),
        "lifecycle.json": canonical_json_bytes(
            {
                "schema": "run-lifecycle-v1",
                "state": "complete",
                "identity_sha256": identity.identity_sha256,
                "owner_token": "6" * 32,
                "fence": 1,
            }
        ),
        "evidence/audit.jsonl": audit_log,
        "evidence/audit-validation.json": canonical_json_bytes(
            {
                "schema": "validated-method-audit-log-v1",
                "policy_sha256": audit_policy_sha256,
                "audit_log_sha256": hashlib.sha256(audit_log).hexdigest(),
                "event_count": 2,
                "terminal_status": "success",
            }
        ),
        "evidence/materialization-logical.json": canonical_json_bytes(
            _synthetic_materialization_logical(method)
        ),
        "evidence/resource-sampling.json": canonical_json_bytes(
            {
                "schema": "run-resource-sampling-v1",
                "status": "complete",
                "backend": "cpu-no-vram-sampling-v1",
                "sampling_interval_ms": 0,
                "sample_count": 0,
                "peak_vram_bytes": 0,
                "runtime_seconds": 1.0,
                "requested_runtime_device": "cpu",
                "compute_cap": dict(method.semantic_config["compute_cap"]),
            }
        ),
        "evidence/source-snapshot.json": canonical_json_bytes(
            {
                "schema": "source-snapshot-v1",
                "commit": "b" * 40,
                "snapshot_sha256": source_snapshot_sha256,
                "inventory": source_inventory,
            }
        ),
    }
    for relative, payload in payloads.items():
        path = run_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    artifact_specs = []
    for role, relative, schema in (
        ("reconstruction", "outputs/reconstruction.npz", "reconstruction-v2"),
        ("method-info", "outputs/method-info.json", "method-info-v2"),
        ("stdout", "outputs/stdout.log", "text-v1"),
        ("stderr", "outputs/stderr.log", "text-v1"),
        ("resolved-config", "resolved-config.json", "resolved-run-config-v1"),
        ("lifecycle", "lifecycle.json", "run-lifecycle-v1"),
        ("audit", "evidence/audit.jsonl", "validated-method-audit-log-v1"),
        (
            "audit-validation",
            "evidence/audit-validation.json",
            "audit-validation-v1",
        ),
        (
            "materialization-logical",
            "evidence/materialization-logical.json",
            "materialized-method-execution-v1",
        ),
        (
            "resource-sampling",
            "evidence/resource-sampling.json",
            "run-resource-sampling-v1",
        ),
        (
            "source-snapshot",
            "evidence/source-snapshot.json",
            "source-snapshot-v1",
        ),
    ):
        payload = payloads[relative]
        artifact_specs.append(
            {
                "role": role,
                "path": relative,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
                "schema_version": schema,
                "required": True,
            }
        )
    metrics_payload = payloads["outputs/metrics.json"]
    manifest = build_manifest(
        status="complete",
        identity=identity,
        config_resolved=_synthetic_complete_config(method),
        inputs={
            "measurements_file_sha256": verified_dataset.payload_evidence[
                "measurements.npz"
            ].sha256,
            "evaluation_truth_file_sha256": verified_dataset.payload_evidence[
                "evaluation-truth.npz"
            ].sha256,
            "dataset_manifest_sha256": (
                verified_dataset.dataset_manifest_sha256
            ),
        },
        runtime=_COMPLETE_RUNTIME,
        execution={
            "command": list(method.command_template),
            "started_at_utc": "2026-08-04T00:00:00Z",
            "ended_at_utc": "2026-08-04T00:00:01Z",
            "return_code": 0,
            "runtime_seconds": 1.0,
            "peak_vram_bytes": 0,
        },
        measurement=dataset_measurement_record(verified_dataset.manifest),
        metrics={
            "version": "metrics-v1",
            "path": "outputs/metrics.json",
            "sha256": hashlib.sha256(metrics_payload).hexdigest(),
        },
        artifacts=artifact_specs,
    )
    (run_dir / "manifest.json").write_bytes(canonical_json_bytes(manifest))
    return run_dir


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _plan(identity=None) -> RunExecutionPlan:
    identity = identity or _identity(method_id="dgi", assets_sha256={})
    return RunExecutionPlan(
        identity=identity,
        resolution_request=MethodResolutionRequest(
            requested_method_id="dgi",
            requested_method_config_id="smoke-default-v1",
            base_config={},
            measurements_metadata={},
            requested_execution_profile="controller-cpu-smoke-v1",
        ),
        registry_path=ROOT / "configs/protocols/methods-v1.yaml",
        checkpoint_store=MappingProxyType({}),
        python_executable=Path(__import__("sys").executable),
        source_root=ROOT,
        requested_runtime_device="cpu",
        config_resolved={
            "runner_execution": {
                "python_executable_sha256": _PYTHON_EXECUTABLE_SHA256,
            }
        },
        assets_sha256=MappingProxyType({}),
        code_commit="b" * 40,
        dirty_worktree=False,
        source_tree_hash=None,
        dependencies_sha256=identity.payload()["dependencies_sha256"],
        environment_lock_sha256=identity.payload()["environment_lock_sha256"],
        metric_version="metrics-v1",
        runtime_metadata=dict(_COMPLETE_RUNTIME),
        expected_dataset_manifest_sha256="3" * 64,
        minimum_free_bytes=1,
    )


def _request(identity=None, execution_plan=None) -> RunRequest:
    identity = identity or _identity(method_id="dgi", assets_sha256={})
    execution_plan = execution_plan or _plan(identity)
    method = resolve_method_semantics(
        "dgi",
        method_config_id="smoke-default-v1",
        base_config={},
        measurements_metadata={},
        execution_profile="controller-cpu-smoke-v1",
        registry_path=ROOT / "configs/protocols/methods-v1.yaml",
    )
    return RunRequest(
        cell=ExperimentCell(
            scientific_contract_id="gsdiff-sim-v1",
            scientific_contract_sha256=HASH,
            campaign_id="test-v1",
            target="tank",
            motion="trans",
            seed=7,
            method="dgi",
        ),
        dataset_dir=Path("must-not-be-read"),
        method=method,
        identity=identity,
        execution_plan=execution_plan,
    )


def _pilot_smoke_policy_fixture():
    resolution_request = MethodResolutionRequest(
        requested_method_id="dgi",
        requested_method_config_id="default",
        base_config={},
        measurements_metadata={},
        requested_execution_profile="pilot-smoke-v1",
    )
    method = resolve_method_semantics(
        "dgi",
        method_config_id="default",
        base_config={},
        measurements_metadata={},
        execution_profile="pilot-smoke-v1",
        registry_path=ROOT / "configs/protocols/methods-v1.yaml",
    )
    plan = replace(
        _plan(),
        resolution_request=resolution_request,
        requested_runtime_device="cpu",
        config_resolved={
            "phase_id": "pilot-v1",
            "runner_execution": {
                "python_executable_sha256": _PYTHON_EXECUTABLE_SHA256,
            },
        },
    )
    return method, plan


def _load_dataset_builder():
    path = ROOT / "scripts/experiments/build_datasets.py"
    spec = importlib.util.spec_from_file_location("runner_dataset_builder", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    __import__("sys").modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_campaign_runner():
    path = ROOT / "scripts/experiments/run_campaign.py"
    spec = importlib.util.spec_from_file_location("runner_campaign_cli", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    __import__("sys").modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _real_request_and_plan(
    tmp_path: Path,
    *,
    requested_execution_profile: str | None = None,
    metric_version: str = "metrics-v1",
):
    global _FAILURE_GENERATED_DATASET

    from gsdiff.data.artifacts import publish_dataset

    builder = _load_dataset_builder()
    protocol_path = ROOT / "configs/protocols/failure-v1.yaml"
    campaign = load_protocol(protocol_path)
    cell = expand_cells(campaign)[0]
    execution_profile = (
        campaign["execution_profile"]
        if requested_execution_profile is None
        else requested_execution_profile
    )
    dataset_plan = builder.plan_campaign_datasets(
        repo_root=ROOT,
        protocol_path=protocol_path,
        runtime={
            "dependencies_sha256": "1" * 64,
            "environment_lock_sha256": "2" * 64,
        },
        generator_commit="3" * 40,
    )
    dataset_requests = [
        candidate
        for candidate in dataset_plan.requests
        if candidate.target_snapshot.target_id == cell.target
        and candidate.seed == cell.seed
        and candidate.acquisition_config
        == campaign["acquisition_configs"][cell.acquisition_config_id]
    ]
    assert len(dataset_requests) == 1
    if _FAILURE_GENERATED_DATASET is None:
        _FAILURE_GENERATED_DATASET = builder.generate_corrected_dataset(
            **dataset_requests[0].generation_arguments()
        )
    generated = _FAILURE_GENERATED_DATASET
    publication = publish_dataset(tmp_path / "run-artifacts", generated)
    verified = publication.verified
    resolution_request = MethodResolutionRequest(
        requested_method_id="dgi",
        requested_method_config_id="default",
        base_config={},
        measurements_metadata=verified.acquisition.acquisition,
        requested_execution_profile=execution_profile,
    )
    method = resolve_method_semantics(
        "dgi",
        method_config_id="default",
        base_config={},
        measurements_metadata=verified.acquisition.acquisition,
        execution_profile=execution_profile,
        registry_path=ROOT / "configs/protocols/methods-v1.yaml",
    )
    config_resolved = {"method_config_sha256": method.method_config_sha256}
    dataset_target = verified.manifest["resolved_generator_config"]["target"]
    assets_sha256 = runner_module._identity_asset_mapping(dataset_target)
    dependencies_sha256 = requirements_dependencies_sha256(
        ROOT / "requirements-lock.txt"
    )
    identity = build_run_identity(
        execution_class="blind_method_child",
        scientific_contract_id=cell.scientific_contract_id,
        scientific_contract_sha256=cell.scientific_contract_sha256,
        method_id=cell.method,
        target_id=cell.target,
        motion_id=cell.motion,
        seed=cell.seed,
        config_sha256=resolved_config_sha256(config_resolved),
        dataset_identity_sha256=verified.dataset_identity_sha256,
        assets_sha256=assets_sha256,
        checkpoints_sha256={},
        code_commit="b" * 40,
        dirty_worktree=False,
        source_tree_hash=None,
        dependencies_sha256=dependencies_sha256,
        environment_lock_sha256=ENVIRONMENT_HASH,
        metric_version=metric_version,
    )
    plan = RunExecutionPlan(
        identity=identity,
        resolution_request=resolution_request,
        registry_path=ROOT / "configs/protocols/methods-v1.yaml",
        checkpoint_store=MappingProxyType({}),
        python_executable=Path(__import__("sys").executable),
        source_root=ROOT,
        requested_runtime_device="cpu",
        config_resolved=config_resolved,
        assets_sha256=MappingProxyType(dict(assets_sha256)),
        code_commit="b" * 40,
        dirty_worktree=False,
        source_tree_hash=None,
        dependencies_sha256=dependencies_sha256,
        environment_lock_sha256=ENVIRONMENT_HASH,
        metric_version=metric_version,
        runtime_metadata=dict(_COMPLETE_RUNTIME),
        expected_dataset_manifest_sha256=verified.dataset_manifest_sha256,
        minimum_free_bytes=1,
    )
    request = RunRequest(cell, verified.dataset_dir, method, identity, plan)
    return _request_plan_with_repo_snapshot(
        request,
        plan,
        tmp_path / "s",
    )


def test_runner_public_api_keeps_two_argument_execution_with_explicit_evidence():
    assert [field.name for field in fields(RunRequest)] == [
        "cell",
        "dataset_dir",
        "method",
        "identity",
        "execution_plan",
    ]
    assert [field.name for field in fields(RunOutcome)] == [
        "status",
        "run_dir",
        "diagnostic_dir",
        "return_code",
    ]
    assert list(inspect.signature(run_request).parameters) == [
        "request",
        "artifact_root",
    ]


def test_nonpromotable_pilot_smoke_policy_accepts_only_the_declared_alias():
    method, plan = _pilot_smoke_policy_fixture()

    assert runner_module._is_exact_cpu_pilot_smoke(method, plan) is True


@pytest.mark.parametrize(
    "mutation",
    [
        "direct-normalized-config",
        "direct-normalized-profile",
        "normalized-config",
        "normalized-profile",
        "phase",
        "cuda",
        "publication-eligible",
        "selection-eligible",
        "promotion-eligible",
        "convergence",
    ],
)
def test_nonpromotable_pilot_smoke_policy_rejects_near_misses(
    mutation: str,
):
    method, plan = _pilot_smoke_policy_fixture()
    if mutation == "direct-normalized-config":
        resolution = replace(
            plan.resolution_request,
            requested_method_config_id="smoke-default-v1",
        )
        method = resolve_method_semantics(
            "dgi",
            method_config_id="smoke-default-v1",
            base_config={},
            measurements_metadata={},
            execution_profile="pilot-smoke-v1",
            registry_path=ROOT / "configs/protocols/methods-v1.yaml",
        )
        plan = replace(plan, resolution_request=resolution)
    elif mutation == "direct-normalized-profile":
        resolution = replace(
            plan.resolution_request,
            requested_method_config_id="smoke-default-v1",
            requested_execution_profile="controller-cpu-smoke-v1",
        )
        method = resolve_method_semantics(
            "dgi",
            method_config_id="smoke-default-v1",
            base_config={},
            measurements_metadata={},
            execution_profile="controller-cpu-smoke-v1",
            registry_path=ROOT / "configs/protocols/methods-v1.yaml",
        )
        plan = replace(plan, resolution_request=resolution)
    elif mutation == "normalized-config":
        method = replace(method, method_config_id="default")
    elif mutation == "normalized-profile":
        method = replace(method, execution_profile="publication-v1")
    elif mutation == "phase":
        plan = replace(
            plan,
            config_resolved={
                **plan.config_resolved,
                "phase_id": "primary-selection-v1",
            },
        )
    elif mutation == "cuda":
        plan = replace(plan, requested_runtime_device="cuda:0")
    elif mutation == "publication-eligible":
        method = replace(method, publication_eligible=True)
    elif mutation == "selection-eligible":
        method = replace(method, selection_eligible=True)
    elif mutation == "promotion-eligible":
        method = replace(method, promotion_eligible=True)
    else:
        assert mutation == "convergence"
        method = replace(method, convergence_status="convergence-required")
    assert runner_module._is_exact_cpu_pilot_smoke(method, plan) is False


def test_authoritative_request_allows_exact_pilot_alias_to_reach_compute_cap(
    monkeypatch,
):
    method, plan = _pilot_smoke_policy_fixture()
    request = _request(execution_plan=plan)
    request = replace(
        request,
        cell=replace(
            request.cell,
            campaign_id="pilot-v1",
            method_config_id="default",
        ),
        method=method,
    )

    def reached_compute_cap(_method):
        raise RuntimeError("reached authoritative compute cap")

    monkeypatch.setattr(runner_module, "_compute_cap", reached_compute_cap)

    with pytest.raises(RuntimeError, match="reached authoritative compute cap"):
        _validate_authoritative_request(request, plan)


def test_authoritative_request_rejects_campaign_execution_profile_mismatch(
    tmp_path: Path,
):
    request, plan = _real_request_and_plan(
        tmp_path,
        requested_execution_profile="primary-full-v1",
    )

    with pytest.raises(ValueError, match="execution_profile"):
        _validate_authoritative_request(request, plan)


def test_authoritative_request_rejects_campaign_metric_version_mismatch(
    tmp_path: Path,
):
    request, plan = _real_request_and_plan(
        tmp_path,
        metric_version="metrics-v2",
    )

    with pytest.raises(ValueError, match="metric_version"):
        _validate_authoritative_request(request, plan)


def test_runner_identity_config_binds_device_snapshot_and_projection_digests():
    base = {"method_config_sha256": "1" * 64}
    compute_cap = {
        "wall_time_seconds": 10,
        "peak_vram_bytes": 1,
        "on_exceed": "ineligible-retain-artifacts",
    }
    cpu = runner_module._identity_bound_config(
        base,
        requested_runtime_device="cpu",
        source_snapshot_sha256="2" * 64,
        source_projection_sha256="3" * 64,
        compute_cap=compute_cap,
        materialization_logical_sha256="5" * 64,
        method_info_contract=_synthetic_method_info_contract(),
        dataset_input_contract=_complete_dataset_input_contract(),
        runtime_contract=_COMPLETE_RUNTIME,
        python_executable_sha256=_PYTHON_EXECUTABLE_SHA256,
    )
    cuda = runner_module._identity_bound_config(
        base,
        requested_runtime_device="cuda:0",
        source_snapshot_sha256="2" * 64,
        source_projection_sha256="3" * 64,
        compute_cap=compute_cap,
        materialization_logical_sha256="5" * 64,
        method_info_contract=_synthetic_method_info_contract(),
        dataset_input_contract=_complete_dataset_input_contract(),
        runtime_contract=_COMPLETE_RUNTIME,
        python_executable_sha256=_PYTHON_EXECUTABLE_SHA256,
    )
    changed_snapshot = runner_module._identity_bound_config(
        base,
        requested_runtime_device="cpu",
        source_snapshot_sha256="4" * 64,
        source_projection_sha256="3" * 64,
        compute_cap=compute_cap,
        materialization_logical_sha256="5" * 64,
        method_info_contract=_synthetic_method_info_contract(),
        dataset_input_contract=_complete_dataset_input_contract(),
        runtime_contract=_COMPLETE_RUNTIME,
        python_executable_sha256=_PYTHON_EXECUTABLE_SHA256,
    )
    changed_projection = runner_module._identity_bound_config(
        base,
        requested_runtime_device="cpu",
        source_snapshot_sha256="2" * 64,
        source_projection_sha256="4" * 64,
        compute_cap=compute_cap,
        materialization_logical_sha256="5" * 64,
        method_info_contract=_synthetic_method_info_contract(),
        dataset_input_contract=_complete_dataset_input_contract(),
        runtime_contract=_COMPLETE_RUNTIME,
        python_executable_sha256=_PYTHON_EXECUTABLE_SHA256,
    )
    changed_cap = runner_module._identity_bound_config(
        base,
        requested_runtime_device="cpu",
        source_snapshot_sha256="2" * 64,
        source_projection_sha256="3" * 64,
        compute_cap={**compute_cap, "wall_time_seconds": 11},
        materialization_logical_sha256="5" * 64,
        method_info_contract=_synthetic_method_info_contract(),
        dataset_input_contract=_complete_dataset_input_contract(),
        runtime_contract=_COMPLETE_RUNTIME,
        python_executable_sha256=_PYTHON_EXECUTABLE_SHA256,
    )

    assert cpu["runner_execution"]["compute_cap"] == compute_cap
    assert len(
        {
            resolved_config_sha256(config)
            for config in (
                cpu,
                cuda,
                changed_snapshot,
                changed_projection,
                changed_cap,
            )
        }
    ) == 5


def test_cuda_request_cannot_reuse_cpu_identity_directory(tmp_path: Path):
    snapshot_sha256, projection_sha256 = _synthetic_source_hashes()
    method = _complete_fixture_method()
    cpu_config = _synthetic_complete_config(method)
    cuda_logical = _synthetic_materialization_logical(method, "cuda:0")
    cuda_config = runner_module._identity_bound_config(
        {"method_config_sha256": method.method_config_sha256},
        requested_runtime_device="cuda:0",
        source_snapshot_sha256=snapshot_sha256,
        source_projection_sha256=projection_sha256,
        compute_cap=method.semantic_config["compute_cap"],
        materialization_logical_sha256=hashlib.sha256(
            canonical_json_bytes(cuda_logical)
        ).hexdigest(),
        method_info_contract=_synthetic_method_info_contract(method),
        dataset_input_contract=_complete_dataset_input_contract(),
        runtime_contract=_COMPLETE_RUNTIME,
        python_executable_sha256=_PYTHON_EXECUTABLE_SHA256,
    )
    cpu_identity = _complete_identity(
        config_sha256=resolved_config_sha256(cpu_config)
    )
    cuda_identity = _complete_identity(
        config_sha256=resolved_config_sha256(cuda_config)
    )
    cpu_run = _write_complete_run(tmp_path, cpu_identity)

    assert cpu_identity.identity_sha256 != cuda_identity.identity_sha256
    assert reusable_run(tmp_path, cpu_identity) == cpu_run
    assert reusable_run(tmp_path, cuda_identity) is None


def test_run_request_uses_the_explicit_plan_owned_by_the_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    artifact_root = tmp_path / "must-not-be-created"
    request = replace(_request(), execution_plan=_plan())

    class ReachedExplicitRequestValidation(Exception):
        pass

    def reached_explicit_request_validation(*_args, **_kwargs):
        raise ReachedExplicitRequestValidation

    monkeypatch.setattr(
        runner_module,
        "_validate_request_dataset_root",
        reached_explicit_request_validation,
    )

    with pytest.raises(ReachedExplicitRequestValidation):
        run_request(request, artifact_root)

    assert not artifact_root.exists()


def test_embedded_plan_identity_mismatch_fails_before_filesystem_or_child(
    tmp_path: Path,
):
    request = _request()
    unrelated = _plan(_identity(method_id="dgi", assets_sha256={}, seed=11))
    artifact_root = tmp_path / "must-not-be-created"

    with pytest.raises(ValueError, match="identity"):
        run_request(replace(request, execution_plan=unrelated), artifact_root)

    assert not artifact_root.exists()


def test_campaign_barrier_checks_later_plan_before_any_runner_call(
    tmp_path: Path,
    monkeypatch,
):
    cli = _load_campaign_runner()
    first_identity = _identity(method_id="dgi", assets_sha256={})
    second_identity = _identity(
        method_id="dgi",
        assets_sha256={},
        seed=8,
    )
    first_plan = _plan(first_identity)
    second_plan = _plan(second_identity)
    first_request = _request(first_identity, first_plan)
    second_request = replace(
        _request(second_identity, second_plan),
        cell=replace(_request(second_identity).cell, seed=8),
    )
    runner_calls = []

    monkeypatch.setattr(
        cli,
        "_validate_authoritative_request",
        lambda request, plan: (object(), request.method),
    )
    monkeypatch.setattr(cli, "_preflight_disk_space", lambda *args: None)

    def preflight_runtime(plan):
        if plan is second_plan:
            raise ValueError("injected later CUDA preflight failure")

    monkeypatch.setattr(cli, "_preflight_runtime", preflight_runtime)
    monkeypatch.setattr(
        cli,
        "run_request",
        lambda *args: runner_calls.append(args),
    )

    with pytest.raises(ValueError, match="later CUDA"):
        cli._campaign_preflight(
            [first_request, second_request],
            tmp_path,
        )

    assert runner_calls == []


def test_runtime_preflight_accepts_cpu_without_cuda_and_rejects_unavailable_cuda(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(runner_module.torch.cuda, "is_available", lambda: False)

    _preflight_execution(_plan(), tmp_path)
    with pytest.raises(ValueError, match="CUDA"):
        _preflight_execution(
            replace(_plan(), requested_runtime_device="cuda:0"),
            tmp_path,
        )


def test_disk_preflight_uses_absolute_free_space_safety_threshold(
    tmp_path: Path,
    monkeypatch,
):
    usage = __import__("types").SimpleNamespace(free=99)
    monkeypatch.setattr(runner_module.shutil, "disk_usage", lambda path: usage)

    with pytest.raises(ValueError, match="disk"):
        _preflight_execution(
            replace(_plan(), minimum_free_bytes=100),
            tmp_path,
        )


def test_windows_gpu_counter_parser_filters_exact_pid_and_sums_adapters():
    records = [
        ("pid_41_luid_0x0_phys_0", 0, 12),
        ("pid_99_luid_0x0_phys_0", 0, 37),
        ("pid_41_luid_0x1_phys_1", 1, 25),
    ]

    assert runner_module._process_vram_from_counter_records(records, 41) == 37


def test_windows_sampler_does_not_abort_on_unrelated_invalid_counter_samples():
    query = (
        "Get-Counter '\\GPU Process Memory(*)\\Dedicated Usage' "
        "-ErrorAction SilentlyContinue"
    )

    assert query in runner_module._POWERSHELL_SAMPLER_SCRIPT


@pytest.mark.skipif(os.name != "nt", reason="Windows GPU counter contract")
def test_windows_sampler_skips_invalid_status_and_keeps_sampling(
    tmp_path: Path,
    monkeypatch,
):
    fake_counter = r"""
$script:counterCall = 0
function Get-Counter {
    [CmdletBinding()]
    param([Parameter(Position=0)] $Counter)
    $script:counterCall += 1
    if ($script:counterCall -eq 1) {
        Write-Error 'injected unrelated invalid counter sample'
        $status = [UInt64]4294967295
    } else {
        $status = [UInt64]0
    }
    [PSCustomObject]@{
        CounterSamples = @(
            [PSCustomObject]@{
                InstanceName = ('pid_' + $targetPid + '_luid_0x0_phys_0')
                CookedValue = [Double]64
                Status = $status
            }
        )
    }
}
"""
    monkeypatch.setattr(
        runner_module,
        "_POWERSHELL_SAMPLER_SCRIPT",
        fake_counter + runner_module._POWERSHELL_SAMPLER_SCRIPT,
    )

    return_code, evidence = _run_monitored_child(
        tmp_path,
        code="import time; time.sleep(0.75)",
        device="cuda:0",
        cap=runner_module._ComputeCap(
            10,
            1024,
            "ineligible-retain-artifacts",
        ),
    )

    assert return_code == 0
    assert evidence["sample_count"] >= 1
    assert evidence["peak_vram_bytes"] == 64


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object contract")
def test_windows_kill_job_close_terminates_child_and_descendant(tmp_path: Path):
    child_marker = tmp_path / "child-survived"
    descendant_marker = tmp_path / "descendant-survived"
    descendant_script = tmp_path / "descendant.py"
    descendant_script.write_text(
        "import sys,time\n"
        "from pathlib import Path\n"
        "time.sleep(3)\n"
        "Path(sys.argv[1]).write_text('survived', encoding='ascii')\n",
        encoding="utf-8",
    )
    child_script = tmp_path / "child.py"
    child_script.write_text(
        "import subprocess,sys,time\n"
        "from pathlib import Path\n"
        "subprocess.Popen([sys.executable, sys.argv[1], sys.argv[2]])\n"
        "time.sleep(3)\n"
        "Path(sys.argv[3]).write_text('survived', encoding='ascii')\n",
        encoding="utf-8",
    )
    job = runner_module._WindowsKillJob()
    process = runner_module._spawn_suspended_in_job(
        [
            str(Path(__import__("sys").executable)),
            str(child_script),
            str(descendant_script),
            str(descendant_marker),
            str(child_marker),
        ],
        cwd=tmp_path,
        env=dict(os.environ),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        job=job,
    )
    time.sleep(0.5)

    job.close()
    process.wait(timeout=5)
    time.sleep(3)

    assert not child_marker.exists()
    assert not descendant_marker.exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object contract")
def test_method_job_rejects_native_descendant_creation(tmp_path: Path):
    descendant_marker = tmp_path / "descendant-created"
    denied_marker = tmp_path / "descendant-denied"
    code = (
        "import subprocess,sys\n"
        "from pathlib import Path\n"
        "try:\n"
        " p=subprocess.Popen([sys.executable,'-c',"
        "'from pathlib import Path;Path(r\\\"'+sys.argv[1]+'\\\").write_text(\\\"created\\\")'])\n"
        " p.wait()\n"
        "except OSError:\n"
        " Path(sys.argv[2]).write_text('denied',encoding='ascii')\n"
        " raise SystemExit(0)\n"
        "raise SystemExit(9)\n"
    )
    stdout_path = tmp_path / "stdout.log"
    stderr_path = tmp_path / "stderr.log"
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        return_code, _evidence = runner_module._run_method_child(
            argv=(
                str(Path(__import__("sys").executable)),
                "-c",
                code,
                str(descendant_marker),
                str(denied_marker),
            ),
            cwd=tmp_path,
            env=dict(os.environ),
            stdout_stream=stdout,
            stderr_stream=stderr,
            requested_runtime_device="cpu",
            compute_cap=runner_module._ComputeCap(
                10,
                1,
                "ineligible-retain-artifacts",
            ),
        )

    assert return_code == 0
    assert denied_marker.read_text("ascii") == "denied"
    assert not descendant_marker.exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object contract")
def test_parent_hard_kill_closes_method_and_sampler_jobs_without_orphans(
    tmp_path: Path,
):
    context = multiprocessing.get_context("spawn")
    method_pid_marker = tmp_path / "method.pid"
    sampler_pid_marker = tmp_path / "sampler.pid"
    worker = context.Process(
        target=_hard_kill_two_jobs_worker,
        args=(tmp_path, method_pid_marker, sampler_pid_marker),
    )
    worker.start()
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline and not (
        method_pid_marker.exists() and sampler_pid_marker.exists()
    ):
        if worker.exitcode is not None:
            break
        time.sleep(0.05)
    assert worker.is_alive()
    assert method_pid_marker.is_file()
    assert sampler_pid_marker.is_file()
    method_pid = int(method_pid_marker.read_text("ascii"))
    sampler_pid = int(sampler_pid_marker.read_text("ascii"))
    assert _windows_pid_is_running(method_pid)
    assert _windows_pid_is_running(sampler_pid)

    worker.terminate()
    worker.join(10)
    assert worker.exitcode not in (None, 0)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and (
        _windows_pid_is_running(method_pid)
        or _windows_pid_is_running(sampler_pid)
    ):
        time.sleep(0.05)

    assert not _windows_pid_is_running(method_pid)
    assert not _windows_pid_is_running(sampler_pid)


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object contract")
def test_method_child_wall_time_cap_kills_before_survival_marker(tmp_path: Path):
    marker = tmp_path / "child-survived"
    code = (
        "import sys,time\n"
        "from pathlib import Path\n"
        "time.sleep(3)\n"
        "Path(sys.argv[1]).write_text('survived', encoding='ascii')\n"
    )
    stdout_path = tmp_path / "stdout.log"
    stderr_path = tmp_path / "stderr.log"
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        with pytest.raises(runner_module._ComputeCapFailure, match="wall"):
            runner_module._run_method_child(
                argv=(
                    str(Path(__import__("sys").executable)),
                    "-c",
                    code,
                    str(marker),
                ),
                cwd=tmp_path,
                env=dict(os.environ),
                stdout_stream=stdout,
                stderr_stream=stderr,
                requested_runtime_device="cpu",
                compute_cap=runner_module._ComputeCap(
                    1,
                    1,
                    "ineligible-retain-artifacts",
                ),
            )
    time.sleep(2.5)

    assert not marker.exists()


def _run_monitored_child(
    tmp_path: Path,
    *,
    code: str,
    device: str,
    cap: runner_module._ComputeCap,
):
    stdout_path = tmp_path / "monitored-stdout.log"
    stderr_path = tmp_path / "monitored-stderr.log"
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        return runner_module._run_method_child(
            argv=(str(Path(__import__("sys").executable)), "-c", code),
            cwd=tmp_path,
            env=dict(os.environ),
            stdout_stream=stdout,
            stderr_stream=stderr,
            requested_runtime_device=device,
            compute_cap=cap,
        )


@pytest.mark.skipif(os.name != "nt", reason="Windows GPU counter contract")
def test_cuda_sampler_malformed_output_fails_closed(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(
        runner_module,
        "_POWERSHELL_SAMPLER_SCRIPT",
        "[Console]::Out.WriteLine('READY'); "
        "[Console]::Out.WriteLine('malformed'); "
        "[Console]::Out.Flush(); Start-Sleep -Seconds 5",
    )

    with pytest.raises(runner_module._ComputeCapFailure, match="sampler"):
        _run_monitored_child(
            tmp_path,
            code="import time; time.sleep(5)",
            device="cuda:0",
            cap=runner_module._ComputeCap(
                10,
                1024,
                "ineligible-retain-artifacts",
            ),
        )


@pytest.mark.skipif(os.name != "nt", reason="Windows GPU counter contract")
def test_cuda_sampler_no_sample_fails_closed(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        runner_module,
        "_POWERSHELL_SAMPLER_SCRIPT",
        "[Console]::Out.WriteLine('READY'); [Console]::Out.Flush(); "
        "Start-Sleep -Seconds 5",
    )

    with pytest.raises(runner_module._ComputeCapFailure, match="no valid sample"):
        _run_monitored_child(
            tmp_path,
            code="pass",
            device="cuda:0",
            cap=runner_module._ComputeCap(
                10,
                1024,
                "ineligible-retain-artifacts",
            ),
        )


@pytest.mark.skipif(os.name != "nt", reason="Windows GPU counter contract")
def test_cuda_sampler_process_failure_fails_closed(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        runner_module,
        "_POWERSHELL_SAMPLER_SCRIPT",
        "[Console]::Out.WriteLine('READY'); [Console]::Out.Flush(); exit 9",
    )

    with pytest.raises(runner_module._ComputeCapFailure, match="sampler failed"):
        _run_monitored_child(
            tmp_path,
            code="import time; time.sleep(5)",
            device="cuda:0",
            cap=runner_module._ComputeCap(
                10,
                1024,
                "ineligible-retain-artifacts",
            ),
        )


@pytest.mark.skipif(os.name != "nt", reason="Windows GPU counter contract")
def test_cuda_sampler_eof_after_valid_sample_fails_closed(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(
        runner_module,
        "_POWERSHELL_SAMPLER_SCRIPT",
        "[Console]::Out.WriteLine('READY'); "
        "$pidText=$args[0]; "
        "[Console]::Out.WriteLine((\"R`tpid_{0}_luid_0x0_phys_0`t0`t64\" -f $pidText)); "
        "[Console]::Out.WriteLine('E'); [Console]::Out.Flush(); "
        "Start-Sleep -Milliseconds 100; exit 0",
    )

    with pytest.raises(runner_module._ComputeCapFailure, match="sampler"):
        _run_monitored_child(
            tmp_path,
            code="import time; time.sleep(2)",
            device="cuda:0",
            cap=runner_module._ComputeCap(
                10,
                1024,
                "ineligible-retain-artifacts",
            ),
        )


@pytest.mark.skipif(os.name != "nt", reason="Windows GPU counter contract")
def test_cuda_sampler_eof_after_child_exit_preserves_valid_final_sample(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(
        runner_module,
        "_POWERSHELL_SAMPLER_SCRIPT",
        "[Console]::Out.WriteLine('READY'); "
        "$pidText=$args[0]; "
        "[Console]::Out.WriteLine((\"R`tpid_{0}_luid_0x0_phys_0`t0`t64\" -f $pidText)); "
        "[Console]::Out.WriteLine('E'); [Console]::Out.Flush(); "
        "while (Get-Process -Id ([Int32]$pidText) -ErrorAction SilentlyContinue) { "
        "Start-Sleep -Milliseconds 25 }; exit 0",
    )

    return_code, evidence = _run_monitored_child(
        tmp_path,
        code="import time; time.sleep(0.25)",
        device="cuda:0",
        cap=runner_module._ComputeCap(
            10,
            1024,
            "ineligible-retain-artifacts",
        ),
    )

    assert return_code == 0
    assert evidence["status"] == "complete"
    assert evidence["sample_count"] >= 1
    assert evidence["peak_vram_bytes"] == 64


@pytest.mark.skipif(os.name != "nt", reason="Windows GPU counter contract")
def test_cuda_sampler_eof_rechecks_child_after_poll_drain_interleaving(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(
        runner_module,
        "_POWERSHELL_SAMPLER_SCRIPT",
        "[Console]::Out.WriteLine('READY'); "
        "$pidText=$args[0]; "
        "[Console]::Out.WriteLine((\"R`tpid_{0}_luid_0x0_phys_0`t0`t64\" -f $pidText)); "
        "[Console]::Out.WriteLine('E'); [Console]::Out.Flush(); "
        "while (Get-Process -Id ([Int32]$pidText) -ErrorAction SilentlyContinue) { "
        "Start-Sleep -Milliseconds 1 }; exit 0",
    )
    original_drain = runner_module._drain_sampler_queue
    delayed = False

    def delayed_once(*args, **kwargs):
        nonlocal delayed
        if not delayed:
            delayed = True
            time.sleep(0.2)
        return original_drain(*args, **kwargs)

    monkeypatch.setattr(runner_module, "_drain_sampler_queue", delayed_once)

    return_code, evidence = _run_monitored_child(
        tmp_path,
        code="import time; time.sleep(0.05)",
        device="cuda:0",
        cap=runner_module._ComputeCap(
            10,
            1024,
            "ineligible-retain-artifacts",
        ),
    )

    assert return_code == 0
    assert evidence["status"] == "complete"
    assert evidence["sample_count"] >= 1
    assert evidence["peak_vram_bytes"] == 64


@pytest.mark.skipif(os.name != "nt", reason="Windows GPU counter contract")
def test_cuda_sampler_natural_failure_after_child_exit_fails_closed(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(
        runner_module,
        "_POWERSHELL_SAMPLER_SCRIPT",
        "[Console]::Out.WriteLine('READY'); "
        "$pidText=$args[0]; "
        "[Console]::Out.WriteLine((\"R`tpid_{0}_luid_0x0_phys_0`t0`t64\" -f $pidText)); "
        "[Console]::Out.WriteLine('E'); [Console]::Out.Flush(); "
        "while (Get-Process -Id ([Int32]$pidText) -ErrorAction SilentlyContinue) { "
        "Start-Sleep -Milliseconds 5 }; exit 9",
    )

    with pytest.raises(runner_module._ComputeCapFailure, match="sampler"):
        _run_monitored_child(
            tmp_path,
            code="import time; time.sleep(0.25)",
            device="cuda:0",
            cap=runner_module._ComputeCap(
                10,
                1024,
                "ineligible-retain-artifacts",
            ),
        )


@pytest.mark.parametrize(
    "failure",
    [
        runner_module._ComputeCapFailure(
            "expected failure",
            {"status": "expected failure"},
        ),
        runner_module._ChildFailure(7),
    ],
)
def test_runner_failure_crosses_generator_context_without_masking(failure):

    @contextlib.contextmanager
    def passthrough():
        yield

    with pytest.raises(type(failure)) as caught:
        with passthrough():
            raise failure

    assert caught.value is failure


@pytest.mark.skipif(os.name != "nt", reason="Windows GPU counter contract")
def test_cuda_sampler_pre_ready_failure_has_final_resource_evidence(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(
        runner_module,
        "_POWERSHELL_SAMPLER_SCRIPT",
        "exit 9",
    )

    with pytest.raises(runner_module._ComputeCapFailure) as caught:
        _run_monitored_child(
            tmp_path,
            code="import time; time.sleep(5)",
            device="cuda:0",
            cap=runner_module._ComputeCap(
                10,
                1024,
                "ineligible-retain-artifacts",
            ),
        )
    assert "startup" in caught.value.reason
    assert caught.value.evidence["status"] == caught.value.reason


@pytest.mark.skipif(os.name != "nt", reason="Windows GPU counter contract")
def test_cuda_sampler_final_malformed_output_has_final_resource_evidence(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(
        runner_module,
        "_POWERSHELL_SAMPLER_SCRIPT",
        "[Console]::Out.WriteLine('READY'); [Console]::Out.Flush(); "
        "Start-Sleep -Milliseconds 100; "
        "[Console]::Out.WriteLine('malformed'); [Console]::Out.Flush(); "
        "Start-Sleep -Seconds 5",
    )

    with pytest.raises(runner_module._ComputeCapFailure) as caught:
        _run_monitored_child(
            tmp_path,
            code="pass",
            device="cuda:0",
            cap=runner_module._ComputeCap(
                10,
                1024,
                "ineligible-retain-artifacts",
            ),
        )
    assert "sampler failed" in caught.value.reason
    assert caught.value.evidence["status"] == caught.value.reason


@pytest.mark.skipif(os.name != "nt", reason="Windows GPU counter contract")
def test_cuda_peak_over_cap_kills_job(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        runner_module,
        "_POWERSHELL_SAMPLER_SCRIPT",
        "[Console]::Out.WriteLine('READY'); "
        "$pidText=$args[0]; "
        "[Console]::Out.WriteLine((\"R`tpid_{0}_luid_0x0_phys_0`t0`t999\" -f $pidText)); "
        "[Console]::Out.WriteLine('E'); [Console]::Out.Flush(); "
        "Start-Sleep -Seconds 5",
    )

    with pytest.raises(runner_module._ComputeCapFailure, match="peak VRAM"):
        _run_monitored_child(
            tmp_path,
            code="import time; time.sleep(5)",
            device="cuda:0",
            cap=runner_module._ComputeCap(
                10,
                998,
                "ineligible-retain-artifacts",
            ),
        )


@pytest.mark.skipif(
    os.name != "nt" or not runner_module.torch.cuda.is_available(),
    reason="real Windows CUDA capability",
)
@pytest.mark.cuda
def test_real_cuda_process_memory_sampler_observes_controlled_allocation(
    tmp_path: Path,
):
    return_code, evidence = _run_monitored_child(
        tmp_path,
        code=(
            "import time,torch; "
            "x=torch.empty((16,1024,1024),device='cuda',dtype=torch.float32); "
            "torch.cuda.synchronize(); time.sleep(3)"
        ),
        device="cuda:0",
        cap=runner_module._ComputeCap(
            20,
            2 * 1024**3,
            "ineligible-retain-artifacts",
        ),
    )

    assert return_code == 0
    assert evidence["backend"] == (
        "windows-gpu-process-memory-dedicated-usage-v1"
    )
    assert evidence["sampling_interval_ms"] == 250
    assert evidence["sample_count"] >= 1
    assert evidence["peak_vram_bytes"] >= 64 * 1024**2


@pytest.mark.parametrize(
    "records",
    [
        [("malformed", 0, 1)],
        [("pid_41_luid_0x0_phys_0", 7, 1)],
        [("pid_41_luid_0x0_phys_0", 0, -1)],
    ],
)
def test_windows_gpu_counter_parser_fails_closed_on_malformed_records(records):
    with pytest.raises(ValueError, match="VRAM sampler"):
        runner_module._process_vram_from_counter_records(records, 41)


def test_no_clobber_rename_rejects_unsupported_platform(
    tmp_path: Path,
    monkeypatch,
):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    monkeypatch.setattr(runner_module, "_WINDOWS_ATOMIC_RENAME", False)

    with pytest.raises(RuntimeError, match="Windows"):
        runner_module._rename_no_clobber(source, destination)

    assert source.is_dir()
    assert not destination.exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows claim locking contract")
def test_claim_lock_rejects_hardlink_without_writing_victim(tmp_path: Path):
    claims_dir = tmp_path / "claims"
    claims_dir.mkdir()
    identity = _identity()
    victim = tmp_path / "victim.bin"
    victim.write_bytes(b"")
    lock_path = claims_dir / f"{identity.identity_sha256}.lock"
    os.link(victim, lock_path)

    with pytest.raises(ValueError, match="lock"):
        with _claim_identity(claims_dir, identity):
            pass

    assert victim.read_bytes() == b""


@pytest.mark.skipif(os.name != "nt", reason="Windows claim metadata contract")
def test_claim_metadata_rejects_hardlink_without_modifying_victim(tmp_path: Path):
    claims_dir = tmp_path / "claims"
    claims_dir.mkdir()
    identity = _identity()
    victim = tmp_path / "victim.json"
    payload = canonical_json_bytes(
        {
            "schema": "run-claim-v1",
            "identity_sha256": identity.identity_sha256,
            "owner_token": "8" * 32,
            "fence": 1,
            "pid": 7,
        }
    )
    victim.write_bytes(payload)
    metadata_path = claims_dir / f"{identity.identity_sha256}.json"
    os.link(victim, metadata_path)

    with pytest.raises(ValueError, match="regular|linked|claim"):
        with _claim_identity(claims_dir, identity):
            pass

    assert victim.read_bytes() == payload


def test_claim_metadata_replacement_after_read_is_rejected(
    tmp_path: Path,
    monkeypatch,
):
    claims_dir = tmp_path / "claims"
    claims_dir.mkdir()
    identity = _identity()
    metadata_path = claims_dir / f"{identity.identity_sha256}.json"
    original = {
        "schema": "run-claim-v1",
        "identity_sha256": identity.identity_sha256,
        "owner_token": "9" * 32,
        "fence": 1,
        "pid": 7,
    }
    replacement = {**original, "owner_token": "a" * 32}
    metadata_path.write_bytes(canonical_json_bytes(original))

    def replace_after_read(path):
        preserved = path.with_name(path.name + ".preserved")
        os.rename(path, preserved)
        path.write_bytes(canonical_json_bytes(replacement))

    monkeypatch.setattr(
        runner_module,
        "_claim_metadata_barrier",
        replace_after_read,
        raising=False,
    )

    with pytest.raises(ValueError, match="metadata.*changed|identity"):
        with _claim_identity(claims_dir, identity):
            pass

    assert json.loads(metadata_path.read_text("utf-8")) == replacement


@pytest.mark.skipif(os.name != "nt", reason="Windows reparse contract")
def test_artifact_directory_rejects_linked_ancestor(tmp_path: Path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "artifacts").mkdir()
    linked = tmp_path / "linked"
    try:
        os.symlink(outside, linked, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks unavailable: {error}")

    with pytest.raises(ValueError, match="symlink|junction|reparse"):
        runner_module._ensure_real_directory(linked / "artifacts")


def test_artifact_directory_checks_every_lexical_ancestor(
    tmp_path: Path,
    monkeypatch,
):
    ancestor = tmp_path / "ancestor"
    leaf = ancestor / "artifacts"
    leaf.mkdir(parents=True)
    real_lstat = os.lstat

    class ReparseStat:
        def __init__(self, value):
            self._value = value
            self.st_file_attributes = getattr(value, "st_file_attributes", 0) | 0x400

        def __getattr__(self, name):
            return getattr(self._value, name)

    def mark_ancestor_reparse(path, *args, **kwargs):
        observed = real_lstat(path, *args, **kwargs)
        if Path(path) == ancestor:
            return ReparseStat(observed)
        return observed

    monkeypatch.setattr(os, "lstat", mark_ancestor_reparse)

    with pytest.raises(ValueError, match="symlink|junction|reparse"):
        runner_module._ensure_real_directory(leaf)


def test_artifact_directory_accepts_a_valid_concurrent_creator(
    tmp_path: Path,
    monkeypatch,
):
    target = tmp_path / "artifacts"
    real_mkdir = os.mkdir
    injected = False

    def inject_concurrent_creator(path, mode=0o777, *, dir_fd=None):
        nonlocal injected
        if Path(path) == target and not injected:
            injected = True
            real_mkdir(path, mode, dir_fd=dir_fd)
            raise FileExistsError("injected concurrent directory winner")
        return real_mkdir(path, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "mkdir", inject_concurrent_creator)

    runner_module._ensure_real_directory(target)

    assert injected is True
    assert target.is_dir()


def test_owned_work_swap_after_marker_check_does_not_delete_replacement(
    tmp_path: Path,
    monkeypatch,
):
    artifact_root = tmp_path / "artifacts"
    work_id = "1" * 32
    work = artifact_root / "work" / work_id
    work.mkdir(parents=True)
    expected = {
        "schema": "run-child-work-v1",
        "work_id": work_id,
        "owner_token": "2" * 32,
        "fence": 1,
        "work_device": os.lstat(work).st_dev,
        "work_inode": os.lstat(work).st_ino,
    }
    (work / ".runner-owner.json").write_bytes(canonical_json_bytes(expected))
    (work / "owned.txt").write_text("owned", encoding="ascii")
    preserved = work.with_name("preserved-owned")
    def swap_before_path_delete(path):
        os.rename(path, preserved)
        path.mkdir()
        (path / "replacement-victim.txt").write_text("victim", encoding="ascii")

    monkeypatch.setattr(
        runner_module,
        "_owned_cleanup_barrier",
        swap_before_path_delete,
    )

    with pytest.raises(ValueError, match="identity|changed"):
        _remove_owned_work(artifact_root, work_id, expected)

    assert (work / "replacement-victim.txt").read_text("ascii") == "victim"
    assert (preserved / "owned.txt").read_text("ascii") == "owned"


@pytest.mark.skipif(os.name != "nt", reason="Windows reparse contract")
def test_owned_work_cleanup_rejects_recursive_reparse(tmp_path: Path):
    artifact_root = tmp_path / "artifacts"
    work_id = "3" * 32
    work = artifact_root / "work" / work_id
    work.mkdir(parents=True)
    expected = {
        "schema": "run-child-work-v1",
        "work_id": work_id,
        "owner_token": "4" * 32,
        "fence": 1,
        "work_device": os.lstat(work).st_dev,
        "work_inode": os.lstat(work).st_ino,
    }
    (work / ".runner-owner.json").write_bytes(canonical_json_bytes(expected))
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "victim.txt").write_text("victim", encoding="ascii")
    try:
        os.symlink(outside, work / "linked", target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks unavailable: {error}")

    with pytest.raises(ValueError, match="reparse|linked"):
        _remove_owned_work(artifact_root, work_id, expected)

    assert (outside / "victim.txt").read_text("ascii") == "victim"
    assert work.is_dir()


def test_owned_work_cleanup_rejects_recursive_hardlink(tmp_path: Path):
    artifact_root = tmp_path / "artifacts"
    work_id = "5" * 32
    work = artifact_root / "work" / work_id
    work.mkdir(parents=True)
    expected = {
        "schema": "run-child-work-v1",
        "work_id": work_id,
        "owner_token": "6" * 32,
        "fence": 1,
        "work_device": os.lstat(work).st_dev,
        "work_inode": os.lstat(work).st_ino,
    }
    (work / ".runner-owner.json").write_bytes(canonical_json_bytes(expected))
    victim = tmp_path / "victim.txt"
    victim.write_text("victim", encoding="ascii")
    nested = work / "nested"
    nested.mkdir()
    os.link(victim, nested / "linked-victim.txt")

    with pytest.raises(ValueError, match="multiply-linked"):
        _remove_owned_work(artifact_root, work_id, expected)

    assert victim.read_text("ascii") == "victim"
    assert work.is_dir()


@pytest.mark.skipif(os.name != "nt", reason="Windows pinned cleanup contract")
def test_owned_cleanup_late_entry_swap_preserves_entire_tree(
    tmp_path: Path,
    monkeypatch,
):
    artifact_root = tmp_path / "artifacts"
    work_id = "9" * 32
    work = artifact_root / "work" / work_id
    work.mkdir(parents=True)
    expected = {
        "schema": "run-child-work-v1",
        "work_id": work_id,
        "owner_token": "a" * 32,
        "fence": 1,
        "work_device": os.lstat(work).st_dev,
        "work_inode": os.lstat(work).st_ino,
    }
    (work / ".runner-owner.json").write_bytes(canonical_json_bytes(expected))
    first = work / "a-first.txt"
    late = work / "z-late.txt"
    first.write_text("first", encoding="ascii")
    late.write_text("late", encoding="ascii")
    preserved = tmp_path / "preserved-late.txt"

    def swap_late_entry(entries):
        os.rename(late, preserved)
        late.write_text("replacement", encoding="ascii")

    monkeypatch.setattr(
        owned_tree_module,
        "_pinned_delete_barrier",
        swap_late_entry,
    )

    with pytest.raises((OSError, ValueError)):
        _remove_owned_work(artifact_root, work_id, expected)

    assert first.read_text("ascii") == "first"
    assert late.read_text("ascii") == "late"
    assert not preserved.exists()


def test_final_stage_swap_after_owner_check_does_not_delete_replacement(
    tmp_path: Path,
    monkeypatch,
):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    identity = _identity()
    stage = runs_dir / f"{identity.identity_sha256}.tmp-{__import__('uuid').uuid4()}"
    stage.mkdir()
    claim = {"owner_token": "7" * 32, "fence": 2}
    (stage / ".owner.json").write_bytes(
        canonical_json_bytes(
            {
                "schema": "run-stage-owner-v1",
                "owner_token": claim["owner_token"],
                "fence": claim["fence"],
                "identity_sha256": identity.identity_sha256,
            }
        )
    )
    (stage / "owned.txt").write_text("owned", encoding="ascii")
    stage_identity = runner_module._directory_identity(
        stage,
        noun="test stage",
    )
    preserved = stage.with_name(stage.name + "-preserved")

    def swap_stage(path):
        os.rename(path, preserved)
        path.mkdir()
        (path / "replacement-victim.txt").write_text("victim", encoding="ascii")

    monkeypatch.setattr(runner_module, "_owned_cleanup_barrier", swap_stage)

    with pytest.raises(ValueError, match="identity|changed"):
        runner_module._cleanup_owned_final_stage(
            stage,
            runs_dir,
            claim,
            identity,
            stage_identity,
        )

    assert (stage / "replacement-victim.txt").read_text("ascii") == "victim"
    assert (preserved / "owned.txt").read_text("ascii") == "owned"


def test_publication_compute_cap_is_exactly_parsed_before_child(tmp_path: Path):
    request, _plan = _real_request_and_plan(tmp_path)

    assert runner_module._compute_cap(request.method) == runner_module._ComputeCap(
        wall_time_seconds=1800,
        peak_vram_bytes=15032385536,
        on_exceed="ineligible-retain-artifacts",
    )


@pytest.mark.parametrize(
    "compute_cap",
    [
        None,
        {"wall_time_seconds": 1, "peak_vram_bytes": 2},
        {
            "wall_time_seconds": 1.5,
            "peak_vram_bytes": 2,
            "on_exceed": "ineligible-retain-artifacts",
        },
        {
            "wall_time_seconds": 1,
            "peak_vram_bytes": -1,
            "on_exceed": "ineligible-retain-artifacts",
        },
        {
            "wall_time_seconds": 1,
            "peak_vram_bytes": 2,
            "on_exceed": "continue",
        },
    ],
)
def test_publication_compute_cap_rejects_missing_extra_or_invalid_fields(
    tmp_path: Path,
    compute_cap,
):
    request, _plan = _real_request_and_plan(tmp_path)
    semantics = dict(request.method.semantic_config)
    if compute_cap is None:
        semantics.pop("compute_cap")
    else:
        semantics["compute_cap"] = compute_cap
    method = replace(
        request.method,
        semantic_config=MappingProxyType(semantics),
    )

    with pytest.raises((TypeError, ValueError), match="compute cap"):
        runner_module._compute_cap(method)


def test_campaign_barrier_checks_global_disk_threshold_once(
    tmp_path: Path,
    monkeypatch,
):
    cli = _load_campaign_runner()
    first_identity = _identity(method_id="dgi", assets_sha256={})
    second_identity = _identity(
        method_id="dgi",
        assets_sha256={},
        seed=8,
    )
    first_plan = _plan(first_identity)
    second_plan = _plan(second_identity)
    first_request = _request(first_identity, first_plan)
    second_request = replace(
        _request(second_identity, second_plan),
        cell=replace(_request(second_identity).cell, seed=8),
    )
    disk_calls = []
    monkeypatch.setattr(cli, "_preflight_runtime", lambda *args: None)
    monkeypatch.setattr(
        cli,
        "_validate_authoritative_request",
        lambda request, plan: (object(), request.method),
    )
    monkeypatch.setattr(
        cli,
        "_preflight_disk_space",
        lambda root, threshold: disk_calls.append((root, threshold)),
    )

    cli._campaign_preflight(
        [first_request, second_request],
        tmp_path,
    )

    assert disk_calls == [(tmp_path, 1)]


@pytest.mark.parametrize("duplicate", ["logical-cell", "identity"])
def test_campaign_barrier_rejects_duplicate_cells_and_identities(
    tmp_path: Path,
    monkeypatch,
    duplicate: str,
):
    cli = _load_campaign_runner()
    first_identity = _identity(method_id="dgi", assets_sha256={})
    first_request = _request(first_identity)
    if duplicate == "logical-cell":
        second_identity = _identity(
            method_id="dgi",
            assets_sha256={},
            code_commit="0" * 40,
        )
        second_request = replace(
            first_request,
            identity=second_identity,
            execution_plan=_plan(second_identity),
        )
    else:
        second_identity = first_identity
        second_request = replace(
            first_request,
            cell=replace(first_request.cell, seed=8),
        )
    monkeypatch.setattr(cli, "_preflight_runtime", lambda *args: None)
    monkeypatch.setattr(
        cli,
        "_validate_authoritative_request",
        lambda *args: (object(), first_request.method),
    )

    with pytest.raises(ValueError, match="duplicate"):
        cli._campaign_preflight(
            [first_request, second_request],
            tmp_path,
        )


def test_campaign_barrier_rejects_plan_identity_rebuild_mismatch(
    tmp_path: Path,
    monkeypatch,
):
    cli = _load_campaign_runner()
    request, plan = _real_request_and_plan(tmp_path)
    payload = dict(request.identity.payload())
    payload["code_commit"] = "0" * 40
    unrelated_identity = build_run_identity(**payload)
    monkeypatch.setattr(cli, "_preflight_runtime", lambda *args: None)
    monkeypatch.setattr(cli, "_preflight_disk_space", lambda *args: None)

    with pytest.raises(ValueError, match="authoritative inputs"):
        cli._campaign_preflight(
            [
                replace(
                    request,
                    execution_plan=replace(
                        plan,
                        identity=unrelated_identity,
                    ),
                )
            ],
            tmp_path,
        )


def test_campaign_failed_outcome_is_canonical_and_nonzero(capsys):
    cli = _load_campaign_runner()
    outcome = RunOutcome(
        status="failed",
        run_dir=None,
        diagnostic_dir=Path("opaque-diagnostic"),
        return_code=7,
    )

    assert cli._emit_campaign_outcome("ready-v1", [outcome]) == 1

    captured = capsys.readouterr()
    assert captured.err == ""
    report = json.loads(captured.out)
    assert captured.out.encode("utf-8") == canonical_json_bytes(report) + b"\n"
    assert report == {
        "schema": "campaign-run-report-v1",
        "campaign_id": "ready-v1",
        "requested": 1,
        "complete": 0,
        "cached": 0,
        "failed": 1,
        "status": "failed",
    }


def test_authoritative_assets_require_exact_logical_mapping_and_physical_multiset(
    tmp_path: Path,
):
    request, plan = _real_request_and_plan(tmp_path)
    digest = next(iter(plan.assets_sha256.values()))
    mutations = (
        MappingProxyType({"renamed-logical-id": digest}),
        MappingProxyType({"target": digest, "duplicate": digest}),
        MappingProxyType({}),
        MappingProxyType({"target": "0" * 64}),
    )

    for assets in mutations:
        changed_plan = replace(plan, assets_sha256=assets)
        with pytest.raises(ValueError, match="assets|identity"):
            _validate_authoritative_request(request, changed_plan)


def test_authoritative_assets_reject_resigned_logical_key_substitution(
    tmp_path: Path,
):
    request, plan = _real_request_and_plan(tmp_path)
    digest = next(iter(plan.assets_sha256.values()))
    forged_assets = {"renamed-logical-id": digest}
    payload = dict(request.identity.payload())
    payload["assets_sha256"] = forged_assets
    forged_identity = build_run_identity(**payload)

    with pytest.raises(ValueError, match="assets"):
        _validate_authoritative_request(
            replace(request, identity=forged_identity),
            replace(
                plan,
                identity=forged_identity,
                assets_sha256=MappingProxyType(forged_assets),
            ),
        )


def test_authoritative_request_rejects_dataset_protocol_semantic_mismatch(
    tmp_path: Path,
):
    request, plan = _real_request_and_plan(tmp_path)
    forged_cell = replace(request.cell, motion="forged-motion")
    payload = dict(request.identity.payload())
    payload["motion_id"] = forged_cell.motion
    forged_identity = build_run_identity(**payload)

    with pytest.raises(ValueError, match="dataset|motion|protocol|campaign"):
        _validate_authoritative_request(
            replace(
                request,
                cell=forged_cell,
                identity=forged_identity,
            ),
            replace(plan, identity=forged_identity),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("method_config_id", "forged-config"),
        ("acquisition_config_id", "forged-acquisition"),
    ],
)
def test_authoritative_request_rejects_cell_config_id_relabel_before_claim(
    tmp_path: Path,
    field: str,
    value: str,
):
    request, plan = _real_request_and_plan(tmp_path)
    changed_request = replace(
        request,
        cell=replace(request.cell, **{field: value}),
    )
    artifact_root = tmp_path / "run-artifacts"

    with pytest.raises(
        ValueError,
        match="method_config_id|acquisition_config_id|campaign",
    ):
        _validate_authoritative_request(changed_request, plan)
    with pytest.raises(
        ValueError,
        match="method_config_id|acquisition_config_id|campaign",
    ):
        run_request(changed_request, artifact_root)

    assert not (artifact_root / ".claims").exists()
    assert not (artifact_root / "runs").exists()
    assert not (artifact_root / "failed").exists()


def test_authoritative_request_rejects_resigned_dataset_input_contract(
    tmp_path: Path,
):
    request, plan = _real_request_and_plan(tmp_path)
    config = json.loads(canonical_json_bytes(plan.config_resolved))
    config["runner_execution"]["dataset_input_contract"][
        "evaluation_truth_file_sha256"
    ] = "0" * 64
    payload = dict(request.identity.payload())
    payload["config_sha256"] = resolved_config_sha256(config)
    forged_identity = build_run_identity(**payload)

    with pytest.raises(ValueError, match="dataset inputs|identity config"):
        _validate_authoritative_request(
            replace(request, identity=forged_identity),
            replace(
                plan,
                identity=forged_identity,
                config_resolved=config,
            ),
        )


def test_authoritative_request_rejects_runtime_metadata_relabel(
    tmp_path: Path,
):
    request, plan = _real_request_and_plan(tmp_path)
    relabeled_runtime = {**plan.runtime_metadata, "python": "9.99-forged"}

    with pytest.raises(ValueError, match="runtime metadata|identity config"):
        _validate_authoritative_request(
            request,
            replace(plan, runtime_metadata=relabeled_runtime),
        )


def test_authoritative_request_rejects_fully_resigned_fake_runtime(
    tmp_path: Path,
):
    request, plan = _real_request_and_plan(tmp_path)
    forged_runtime = {name: f"forged-{name}" for name in _COMPLETE_RUNTIME}
    config = json.loads(canonical_json_bytes(plan.config_resolved))
    config["runner_execution"]["runtime_contract"] = forged_runtime
    payload = dict(request.identity.payload())
    payload["config_sha256"] = resolved_config_sha256(config)
    forged_identity = build_run_identity(**payload)

    with pytest.raises(ValueError, match="live authoritative runtime"):
        _validate_authoritative_request(
            replace(request, identity=forged_identity),
            replace(
                plan,
                identity=forged_identity,
                config_resolved=config,
                runtime_metadata=forged_runtime,
            ),
        )


@pytest.mark.parametrize("alternate_kind", ["dummy", "byte-identical-copy"])
def test_child_python_must_be_exact_authoritative_interpreter_before_claim(
    tmp_path: Path,
    monkeypatch,
    alternate_kind: str,
):
    request, plan = _real_request_and_plan(tmp_path)
    artifact_root = tmp_path / "run-artifacts"
    alternate = tmp_path / f"{alternate_kind}.exe"
    if alternate_kind == "dummy":
        alternate.write_bytes(b"not python")
    else:
        shutil.copyfile(Path(__import__("sys").executable), alternate)
    changed_plan = replace(plan, python_executable=alternate)
    child_called = False

    def forbidden_child(**_kwargs):
        nonlocal child_called
        child_called = True
        raise AssertionError("child must not be called")

    monkeypatch.setattr(runner_module, "_run_method_child", forbidden_child)

    with pytest.raises(ValueError, match="Python executable|sys.executable"):
        _validate_authoritative_request(request, changed_plan)
    with pytest.raises(ValueError, match="Python executable|sys.executable"):
        runner_module._preflight_runtime(changed_plan)
    with pytest.raises(ValueError, match="Python executable|sys.executable"):
        run_request(
            replace(request, execution_plan=changed_plan),
            artifact_root,
        )

    assert child_called is False
    assert not (artifact_root / ".claims").exists()
    assert not (artifact_root / "runs").exists()
    assert not (artifact_root / "failed").exists()


@pytest.mark.parametrize(
    ("provenance", "locator_kind", "hash_matches", "message"),
    [
        ("blocked", "file", True, "provenance"),
        ("verified", "file", False, "bytes"),
        ("verified", "directory", True, "regular file"),
    ],
)
def test_authoritative_checkpoint_barrier_rejects_unresolved_unstable_or_wrong_bytes(
    tmp_path: Path,
    monkeypatch,
    provenance: str,
    locator_kind: str,
    hash_matches: bool,
    message: str,
):
    request, plan = _real_request_and_plan(tmp_path)
    checkpoint_bytes = b"authoritative-checkpoint"
    correct_hash = hashlib.sha256(checkpoint_bytes).hexdigest()
    expected_hash = correct_hash if hash_matches else "0" * 64
    locator = tmp_path / "checkpoint.pt"
    if locator_kind == "directory":
        locator.mkdir()
    else:
        locator.write_bytes(checkpoint_bytes)
    requirement = CheckpointRequirement(
        "prior-v1",
        expected_hash,
        provenance,
    )
    method = replace(
        request.method,
        checkpoint_requirements=(requirement,),
    )
    payload = dict(request.identity.payload())
    payload["checkpoints_sha256"] = {"prior-v1": expected_hash}
    identity = build_run_identity(**payload)
    changed_request = replace(request, method=method, identity=identity)
    changed_plan = replace(
        plan,
        identity=identity,
        checkpoint_store=MappingProxyType({"prior-v1": locator}),
    )
    changed_request, changed_plan = _request_plan_with_repo_snapshot(
        changed_request,
        changed_plan,
        tmp_path / "checkpoint-source",
    )
    monkeypatch.setattr(
        runner_module,
        "resolve_method_semantics",
        lambda *args, **kwargs: method,
    )

    with pytest.raises(ValueError, match=message):
        _validate_authoritative_request(changed_request, changed_plan)


def test_authoritative_checkpoint_barrier_rejects_reparse_locator(
    tmp_path: Path,
    monkeypatch,
):
    request, plan = _real_request_and_plan(tmp_path)
    checkpoint_bytes = b"authoritative-checkpoint"
    checkpoint_hash = hashlib.sha256(checkpoint_bytes).hexdigest()
    target = tmp_path / "checkpoint-target.pt"
    target.write_bytes(checkpoint_bytes)
    locator = tmp_path / "checkpoint-link.pt"
    try:
        locator.symlink_to(target)
    except OSError as error:
        pytest.skip(f"filesystem cannot create a test symlink: {error}")
    requirement = CheckpointRequirement(
        "prior-v1",
        checkpoint_hash,
        "verified",
    )
    method = replace(
        request.method,
        checkpoint_requirements=(requirement,),
    )
    payload = dict(request.identity.payload())
    payload["checkpoints_sha256"] = {"prior-v1": checkpoint_hash}
    identity = build_run_identity(**payload)
    changed_request = replace(request, method=method, identity=identity)
    changed_plan = replace(
        plan,
        identity=identity,
        checkpoint_store=MappingProxyType({"prior-v1": locator}),
    )
    changed_request, changed_plan = _request_plan_with_repo_snapshot(
        changed_request,
        changed_plan,
        tmp_path / "checkpoint-source",
    )
    monkeypatch.setattr(
        runner_module,
        "resolve_method_semantics",
        lambda *args, **kwargs: method,
    )

    with pytest.raises(ValueError, match="regular file|linked|reparse"):
        _validate_authoritative_request(changed_request, changed_plan)


def test_successful_request_validates_child_and_promotes_exact_final_inventory(
    tmp_path: Path,
):
    request, plan = _real_request_and_plan(tmp_path)
    artifact_root = tmp_path / "run-artifacts"

    outcome = run_request(request, artifact_root)

    assert outcome.status == "complete"
    assert outcome.run_dir == (
        artifact_root / "runs" / request.identity.identity_sha256
    )
    assert outcome.diagnostic_dir is None
    assert outcome.return_code == 0
    assert sorted(path.name for path in (outcome.run_dir / "outputs").iterdir()) == [
        "method-info.json",
        "metrics.json",
        "reconstruction.npz",
        "stderr.log",
        "stdout.log",
    ]
    assert sorted(
        path.relative_to(outcome.run_dir).as_posix()
        for path in outcome.run_dir.rglob("*")
        if path.is_file()
        ) == [
            "evidence/audit-validation.json",
            "evidence/audit.jsonl",
            "evidence/materialization-logical.json",
            "evidence/resource-sampling.json",
            "evidence/source-snapshot.json",
            "lifecycle.json",
        "manifest.json",
        "outputs/method-info.json",
        "outputs/metrics.json",
        "outputs/reconstruction.npz",
        "outputs/stderr.log",
        "outputs/stdout.log",
        "resolved-config.json",
    ]
    resource_path = outcome.run_dir / "evidence/resource-sampling.json"
    resource = json.loads(resource_path.read_text("utf-8"))
    assert resource_path.read_bytes() == canonical_json_bytes(resource)
    assert resource["backend"] == "cpu-no-vram-sampling-v1"
    assert resource["sampling_interval_ms"] == 0
    assert resource["sample_count"] == 0
    assert resource["peak_vram_bytes"] == 0
    manifest = json.loads((outcome.run_dir / "manifest.json").read_text("utf-8"))
    descriptor = next(
        item
        for item in manifest["artifacts"]
        if item["role"] == "resource-sampling"
    )
    assert descriptor["path"] == "evidence/resource-sampling.json"
    assert descriptor["sha256"] == hashlib.sha256(resource_path.read_bytes()).hexdigest()
    logical_path = outcome.run_dir / "evidence/materialization-logical.json"
    logical = json.loads(logical_path.read_text("utf-8"))
    assert logical_path.read_bytes() == canonical_json_bytes(logical)
    logical_descriptor = next(
        item
        for item in manifest["artifacts"]
        if item["role"] == "materialization-logical"
    )
    assert logical_descriptor["path"] == "evidence/materialization-logical.json"
    assert logical_descriptor["sha256"] == hashlib.sha256(
        logical_path.read_bytes()
    ).hexdigest()
    assert manifest["execution"]["peak_vram_bytes"] == 0
    assert reusable_run(artifact_root, request.identity) == outcome.run_dir

    cached = run_request(request, artifact_root)
    assert cached == RunOutcome("cached", outcome.run_dir, None, 0)


def test_run_request_rejects_foreign_dataset_root_before_artifacts(
    tmp_path: Path,
):
    request, plan = _real_request_and_plan(tmp_path)
    foreign_root = tmp_path / "foreign-run-artifacts"

    with pytest.raises(ValueError, match="canonical run artifact root"):
        run_request(request, foreign_root)

    assert not foreign_root.exists()


def test_child_work_must_share_the_final_stage_filesystem(
    tmp_path: Path, monkeypatch
):
    devices = iter((11, 12))
    monkeypatch.setattr(
        "gsdiff.experiments.runner._filesystem_device",
        lambda path: next(devices),
    )

    with pytest.raises(ValueError, match="filesystem|volume"):
        _require_same_filesystem(tmp_path / "work", tmp_path / "runs")


def test_owned_work_cleanup_rejects_wrong_parent_token_and_fence(tmp_path: Path):
    artifact_root = tmp_path / "artifacts"
    work_id = "a" * 32
    work = artifact_root / "work" / work_id
    work.mkdir(parents=True)
    expected = {
        "schema": "run-child-work-v1",
        "work_id": work_id,
        "owner_token": "b" * 32,
        "fence": 3,
    }
    (work / ".runner-owner.json").write_bytes(
        canonical_json_bytes({**expected, "fence": 2})
    )

    with pytest.raises(ValueError, match="ownership"):
        _remove_owned_work(artifact_root, work_id, expected)
    assert work.is_dir()

    outside = tmp_path / "outside" / work_id
    outside.mkdir(parents=True)
    (outside / ".runner-owner.json").write_bytes(canonical_json_bytes(expected))
    with pytest.raises((TypeError, ValueError), match="work|parent|ownership"):
        _remove_owned_work(tmp_path, f"outside/{work_id}", expected)
    assert outside.is_dir()


@pytest.mark.parametrize(
    "failure_mode",
    ["child-nonzero", "partial-output", "audit", "tamper", "evaluator", "metric-write"],
)
def test_failed_pipeline_never_promotes_and_preserves_prior_valid_run_bytes(
    tmp_path: Path, monkeypatch, failure_mode: str
):
    request, plan = _real_request_and_plan(tmp_path)
    artifact_root = tmp_path / "run-artifacts"
    prior = _write_complete_run(
        artifact_root,
        _complete_identity(method_id="gsdiff_tv", assets_sha256={}),
    )
    prior_before = _tree_bytes(prior)

    if failure_mode in {"child-nonzero", "partial-output"}:
        def fake_run_method_child(*, argv, requested_runtime_device, compute_cap, **kwargs):
            if failure_mode == "partial-output":
                output_dir = Path(argv[argv.index("--output-dir") + 1])
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "reconstruction.npz").write_bytes(b"partial")
            evidence = runner_module._resource_evidence(
                status="complete",
                compute_cap=compute_cap,
                runtime_seconds=0.01,
                sample_count=0,
                peak_vram_bytes=0,
                requested_runtime_device=requested_runtime_device,
            )
            return (7 if failure_mode == "child-nonzero" else 0), evidence

        monkeypatch.setattr(
            runner_module,
            "_run_method_child",
            fake_run_method_child,
        )
        if failure_mode == "partial-output":
            monkeypatch.setattr(
                runner_module,
                "validate_audit_log",
                lambda *args, **kwargs: {
                    "schema": "validated-method-audit-log-v1"
                },
            )
    elif failure_mode == "audit":
        monkeypatch.setattr(
            runner_module,
            "validate_audit_log",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                ValueError("injected audit failure")
            ),
        )
    elif failure_mode == "tamper":
        real_validate = runner_module.validate_method_child_outputs_v2

        def validate_then_tamper(output_dir, **kwargs):
            result = real_validate(output_dir, **kwargs)
            (output_dir / "reconstruction.npz").write_bytes(b"tampered")
            return result

        monkeypatch.setattr(
            runner_module,
            "validate_method_child_outputs_v2",
            validate_then_tamper,
        )
    elif failure_mode == "evaluator":
        monkeypatch.setattr(
            runner_module,
            "evaluate_video_global_affine",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                ValueError("injected evaluator failure")
            ),
        )
    else:
        real_write = runner_module._write_file_durable

        def fail_metric_write(path, payload):
            if path.name == "metrics.json":
                raise OSError("injected metric write failure")
            return real_write(path, payload)

        monkeypatch.setattr(runner_module, "_write_file_durable", fail_metric_write)

    outcome = run_request(request, artifact_root)

    assert outcome.status == "failed"
    assert outcome.run_dir is None
    assert outcome.diagnostic_dir is not None
    assert outcome.diagnostic_dir.is_dir()
    work_pointer = json.loads(
        (outcome.diagnostic_dir / "child-work.json").read_text("utf-8")
    )
    assert set(work_pointer) == {
        "schema",
        "work_id",
        "owner_token",
        "fence",
        "work_device",
        "work_inode",
        "retained",
    }
    assert len(work_pointer["work_id"]) == 32
    assert "/" not in work_pointer["work_id"]
    assert work_pointer["retained"] is False
    assert not (artifact_root / "work" / work_pointer["work_id"]).exists()
    evidence = json.loads(
        (outcome.diagnostic_dir / "diagnostic-artifacts.json").read_text("utf-8")
    )
    assert sorted(evidence["present"]) == [
        "evidence/audit.jsonl",
        "evidence/resource-sampling.json",
        "outputs/stderr.log",
        "outputs/stdout.log",
    ]
    assert evidence["missing_expected"] == []
    for relative, descriptor in evidence["present"].items():
        payload = (outcome.diagnostic_dir / relative).read_bytes()
        assert descriptor == {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
    assert not (
        artifact_root / "runs" / request.identity.identity_sha256
    ).exists()
    assert _tree_bytes(prior) == prior_before


def test_compute_cap_failure_retains_canonical_resource_diagnostic(
    tmp_path: Path,
    monkeypatch,
):
    request, plan = _real_request_and_plan(tmp_path)
    artifact_root = tmp_path / "run-artifacts"
    compute_cap = runner_module._compute_cap(request.method)
    expected_evidence = runner_module._resource_evidence(
        status="wall time compute cap exceeded",
        compute_cap=compute_cap,
        runtime_seconds=1800.25,
        sample_count=0,
        peak_vram_bytes=0,
        requested_runtime_device="cpu",
    )

    def fail_cap(**kwargs):
        raise runner_module._ComputeCapFailure(
            "wall time compute cap exceeded",
            expected_evidence,
        )

    monkeypatch.setattr(runner_module, "_run_method_child", fail_cap)
    outcome = run_request(request, artifact_root)

    assert outcome.status == "failed"
    assert outcome.run_dir is None
    assert outcome.diagnostic_dir is not None
    retained_path = (
        outcome.diagnostic_dir / "evidence/resource-sampling.json"
    )
    retained = json.loads(retained_path.read_text("utf-8"))
    assert retained == expected_evidence
    assert retained_path.read_bytes() == canonical_json_bytes(expected_evidence)
    failure = json.loads(
        (outcome.diagnostic_dir / "failure.json").read_text("utf-8")
    )
    assert failure["error_type"] == "_ComputeCapFailure"
    assert failure["return_code"] == 1
    assert not (
        artifact_root / "runs" / request.identity.identity_sha256
    ).exists()


def test_materialization_failure_records_partial_diagnostic_without_masking_error(
    tmp_path: Path,
    monkeypatch,
):
    request, plan = _real_request_and_plan(tmp_path)
    artifact_root = tmp_path / "run-artifacts"

    def fail_during_materialization(*args, stage_root, **kwargs):
        del args, kwargs
        (stage_root / "parent/logs").mkdir(parents=True)
        (stage_root / "parent/logs/stdout.log").write_bytes(b"partial stdout")
        raise OSError("injected materialization failure")

    monkeypatch.setattr(
        runner_module,
        "materialize_method_execution",
        fail_during_materialization,
    )

    outcome = run_request(request, artifact_root)

    assert outcome.status == "failed"
    assert outcome.diagnostic_dir is not None
    failure = json.loads(
        (outcome.diagnostic_dir / "failure.json").read_text("utf-8")
    )
    assert failure["error_type"] == "OSError"
    assert failure["message"] == "injected materialization failure"
    diagnostic = json.loads(
        (outcome.diagnostic_dir / "diagnostic-artifacts.json").read_text("utf-8")
    )
    assert sorted(diagnostic["present"]) == ["outputs/stdout.log"]
    assert diagnostic["missing_expected"] == [
        "evidence/audit.jsonl",
        "evidence/resource-sampling.json",
        "outputs/stderr.log",
    ]
    pointer = json.loads(
        (outcome.diagnostic_dir / "child-work.json").read_text("utf-8")
    )
    assert pointer["retained"] is True
    assert (artifact_root / "work" / pointer["work_id"]).is_dir()


def test_materialization_failure_records_zero_evidence_without_masking_error(
    tmp_path: Path,
    monkeypatch,
):
    request, plan = _real_request_and_plan(tmp_path)
    artifact_root = tmp_path / "run-artifacts"

    def fail_before_evidence(*args, stage_root, **kwargs):
        del args, kwargs
        stage_root.mkdir()
        raise OSError("injected zero-evidence failure")

    monkeypatch.setattr(
        runner_module,
        "materialize_method_execution",
        fail_before_evidence,
    )

    outcome = run_request(request, artifact_root)

    assert outcome.status == "failed"
    assert outcome.diagnostic_dir is not None
    failure = json.loads(
        (outcome.diagnostic_dir / "failure.json").read_text("utf-8")
    )
    assert failure["message"] == "injected zero-evidence failure"
    diagnostic = json.loads(
        (outcome.diagnostic_dir / "diagnostic-artifacts.json").read_text("utf-8")
    )
    assert diagnostic["present"] == {}
    assert diagnostic["missing_expected"] == [
        "evidence/audit.jsonl",
        "evidence/resource-sampling.json",
        "outputs/stderr.log",
        "outputs/stdout.log",
    ]
    assert diagnostic["retained"] is True


@pytest.mark.parametrize("secondary_fault", ["collect", "write", "rename"])
def test_diagnostic_secondary_failure_raises_original_error_with_note(
    tmp_path: Path,
    monkeypatch,
    secondary_fault: str,
):
    request, plan = _real_request_and_plan(tmp_path)
    artifact_root = tmp_path / "run-artifacts"

    def fail_materialization(*args, **kwargs):
        raise OSError("original materialization failure")

    monkeypatch.setattr(
        runner_module,
        "materialize_method_execution",
        fail_materialization,
    )
    if secondary_fault == "collect":
        monkeypatch.setattr(
            runner_module,
            "_finalize_diagnostic_evidence",
            lambda **kwargs: (_ for _ in ()).throw(
                OSError("diagnostic collection failure")
            ),
        )
    elif secondary_fault == "write":
        real_write = runner_module._write_file_durable

        def fail_failure_write(path, payload):
            if Path(path).name == "failure.json":
                raise OSError("diagnostic write failure")
            return real_write(path, payload)

        monkeypatch.setattr(
            runner_module,
            "_write_file_durable",
            fail_failure_write,
        )
    else:
        real_rename = runner_module._rename_no_clobber

        def fail_diagnostic_rename(source, destination):
            if Path(destination).parent.name == "failed":
                raise OSError("diagnostic rename failure")
            return real_rename(source, destination)

        monkeypatch.setattr(
            runner_module,
            "_rename_no_clobber",
            fail_diagnostic_rename,
        )

    with pytest.raises(
        OSError,
        match="original materialization failure",
    ) as caught:
        run_request(request, artifact_root)

    notes = getattr(caught.value, "__notes__", [])
    assert any("diagnostic" in note for note in notes)


def test_unstable_partial_evidence_is_rejected_without_masking_original_error(
    tmp_path: Path,
    monkeypatch,
):
    request, plan = _real_request_and_plan(tmp_path)
    artifact_root = tmp_path / "run-artifacts"

    def fail_with_evidence(*args, stage_root, **kwargs):
        del args, kwargs
        (stage_root / "parent/logs").mkdir(parents=True)
        (stage_root / "parent/logs/stdout.log").write_bytes(b"unstable")
        raise OSError("original materialization failure")

    real_read = runner_module._read_stable_regular_bytes

    def reject_changed_evidence(path, *, noun):
        if path.name == "stdout.log" and "diagnostic evidence" in noun:
            raise ValueError("diagnostic evidence changed while being read")
        return real_read(path, noun=noun)

    monkeypatch.setattr(
        runner_module,
        "materialize_method_execution",
        fail_with_evidence,
    )
    monkeypatch.setattr(
        runner_module,
        "_read_stable_regular_bytes",
        reject_changed_evidence,
    )

    outcome = run_request(request, artifact_root)

    assert outcome.status == "failed"
    assert outcome.diagnostic_dir is not None
    failure = json.loads(
        (outcome.diagnostic_dir / "failure.json").read_text("utf-8")
    )
    assert failure["message"] == "original materialization failure"
    diagnostic = json.loads(
        (outcome.diagnostic_dir / "diagnostic-artifacts.json").read_text("utf-8")
    )
    assert diagnostic["present"] == {}
    assert "outputs/stdout.log" in diagnostic["missing_expected"]
    assert any("changed while being read" in issue for issue in diagnostic["issues"])


def test_owner_marker_write_failure_retains_work_and_original_error(
    tmp_path: Path,
    monkeypatch,
):
    request, plan = _real_request_and_plan(tmp_path)
    artifact_root = tmp_path / "run-artifacts"
    real_write = runner_module._write_file_durable

    def fail_owner_marker(path, payload):
        if path.name == ".runner-owner.json":
            raise OSError("injected owner marker write failure")
        return real_write(path, payload)

    monkeypatch.setattr(runner_module, "_write_file_durable", fail_owner_marker)

    outcome = run_request(request, artifact_root)

    assert outcome.status == "failed"
    assert outcome.diagnostic_dir is not None
    failure = json.loads(
        (outcome.diagnostic_dir / "failure.json").read_text("utf-8")
    )
    assert failure["message"] == "injected owner marker write failure"
    pointer = json.loads(
        (outcome.diagnostic_dir / "child-work.json").read_text("utf-8")
    )
    assert pointer["retained"] is True
    assert (artifact_root / "work" / pointer["work_id"]).is_dir()


def test_valid_child_output_replacement_after_validation_is_rejected(
    tmp_path: Path,
    monkeypatch,
):
    request, plan = _real_request_and_plan(tmp_path)
    artifact_root = tmp_path / "run-artifacts"
    real_validate = runner_module.validate_method_child_outputs_v2
    evaluation_called = False

    def validate_then_replace_with_other_valid_pair(output_dir, **kwargs):
        original_hashes = real_validate(output_dir, **kwargs)
        reconstruction_path = output_dir / "reconstruction.npz"
        loaded = runner_module.load_reconstruction_v2(reconstruction_path)
        alternate_reconstruction = loaded.reconstruction.copy()
        alternate_dgi = None
        if loaded.dgi is None:
            alternate_reconstruction.flat[0] += 0.125
        else:
            alternate_dgi = loaded.dgi.copy()
            alternate_dgi.flat[0] += 0.125
            alternate_reconstruction[...] = alternate_dgi
        arrays = {
            "reconstruction": alternate_reconstruction,
            "frame_indices": loaded.frame_indices,
            "time_grid": loaded.time_grid,
        }
        if alternate_dgi is not None:
            arrays["dgi"] = alternate_dgi
        if loaded.estimated_motion_trajectory is not None:
            arrays["estimated_motion_trajectory"] = (
                loaded.estimated_motion_trajectory
            )
        metadata = child_outputs_module._reconstruction_metadata(
            method_id=kwargs["expected_method"].method_id,
            acquisition=kwargs["expected_acquisition"],
            arrays=arrays,
        )
        reconstruction_path.write_bytes(
            child_outputs_module.npz_bytes(arrays=arrays, metadata=metadata)
        )
        info_path = output_dir / "method-info.json"
        info = json.loads(info_path.read_text("utf-8"))
        info["reconstruction"]["sha256"] = hashlib.sha256(
            reconstruction_path.read_bytes()
        ).hexdigest()
        info["reconstruction"]["array_descriptors"] = metadata[
            "array_descriptors"
        ]
        info_path.write_bytes(canonical_json_bytes(info))
        alternate_hashes = real_validate(output_dir, **kwargs)
        assert alternate_hashes != original_hashes
        return original_hashes

    monkeypatch.setattr(
        runner_module,
        "validate_method_child_outputs_v2",
        validate_then_replace_with_other_valid_pair,
    )

    real_evaluate = runner_module.evaluate_video_global_affine

    def mark_evaluation(*args, **kwargs):
        nonlocal evaluation_called
        evaluation_called = True
        return real_evaluate(*args, **kwargs)

    monkeypatch.setattr(
        runner_module,
        "evaluate_video_global_affine",
        mark_evaluation,
    )

    outcome = run_request(request, artifact_root)

    assert outcome.status == "failed"
    assert outcome.run_dir is None
    assert outcome.diagnostic_dir is not None
    failure = json.loads(
        (outcome.diagnostic_dir / "failure.json").read_text("utf-8")
    )
    assert "validated child output bytes changed" in failure["message"]
    assert evaluation_called is False


def test_valid_audit_replacement_after_validation_is_rejected(
    tmp_path: Path,
    monkeypatch,
):
    request, plan = _real_request_and_plan(tmp_path)
    artifact_root = tmp_path / "run-artifacts"
    real_validate = runner_module.validate_audit_log

    def validate_then_replace_with_other_valid_audit(path, **kwargs):
        original = real_validate(path, **kwargs)
        events = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
        timestamp = events[0]["timestamp_utc"]
        events[0]["timestamp_utc"] = (
            timestamp[:-2] + ("1Z" if timestamp[-2:] != "1Z" else "2Z")
        )
        path.write_bytes(
            b"".join(
                json.dumps(
                    event,
                    allow_nan=False,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
                + b"\n"
                for event in events
            )
        )
        alternate = real_validate(path, **kwargs)
        assert alternate["audit_log_sha256"] != original["audit_log_sha256"]
        return original

    monkeypatch.setattr(
        runner_module,
        "validate_audit_log",
        validate_then_replace_with_other_valid_audit,
    )

    outcome = run_request(request, artifact_root)

    assert outcome.status == "failed"
    assert outcome.run_dir is None
    assert outcome.diagnostic_dir is not None
    failure = json.loads(
        (outcome.diagnostic_dir / "failure.json").read_text("utf-8")
    )
    assert "validated audit bytes changed" in failure["message"]


def test_staged_validated_snapshot_is_crosschecked_after_write(
    tmp_path: Path,
    monkeypatch,
):
    request, plan = _real_request_and_plan(tmp_path)
    artifact_root = tmp_path / "run-artifacts"
    real_write = runner_module._write_file_durable

    def corrupt_staged_snapshot(path, payload):
        result = real_write(path, payload)
        if path.name == "reconstruction.npz" and path.parent.name == "outputs":
            path.write_bytes(b"replacement after staged write")
        return result

    monkeypatch.setattr(
        runner_module,
        "_write_file_durable",
        corrupt_staged_snapshot,
    )

    outcome = run_request(request, artifact_root)

    assert outcome.status == "failed"
    assert outcome.diagnostic_dir is not None
    failure = json.loads(
        (outcome.diagnostic_dir / "failure.json").read_text("utf-8")
    )
    assert "snapshot changed during staged write" in failure["message"]


def test_materialized_source_projection_mismatch_fails_before_child(
    tmp_path: Path,
    monkeypatch,
):
    request, plan = _real_request_and_plan(tmp_path)
    request, plan = _request_plan_with_repo_snapshot(
        request,
        plan,
        tmp_path / "snapshot-artifacts",
    )
    artifact_root = tmp_path / "run-artifacts"
    real_materialize = runner_module.materialize_method_execution

    def tamper_logical_projection(*args, **kwargs):
        materialized = real_materialize(*args, **kwargs)
        record = dict(materialized.materialization_record)
        logical = dict(record["logical"])
        logical["source_snapshot_sha256"] = "f" * 64
        record["logical"] = logical
        return replace(materialized, materialization_record=record)

    def forbidden_child(*args, **kwargs):
        raise AssertionError("source projection mismatch reached child")

    monkeypatch.setattr(
        runner_module,
        "materialize_method_execution",
        tamper_logical_projection,
    )
    monkeypatch.setattr(runner_module, "_run_method_child", forbidden_child)

    outcome = run_request(request, artifact_root)

    assert outcome.status == "failed"
    assert outcome.diagnostic_dir is not None
    failure = json.loads(
        (outcome.diagnostic_dir / "failure.json").read_text("utf-8")
    )
    assert "materialized source inventory" in failure["message"]


def test_identity_hash_rejects_plausible_materializer_logical_mutation_before_child(
    tmp_path: Path,
    monkeypatch,
):
    request, plan = _real_request_and_plan(tmp_path)
    request, plan = _request_plan_with_repo_snapshot(
        request,
        plan,
        tmp_path / "snapshot-artifacts",
    )
    artifact_root = tmp_path / "run-artifacts"
    real_materialize = runner_module.materialize_method_execution

    def tamper_entrypoint(*args, **kwargs):
        materialized = real_materialize(*args, **kwargs)
        record = dict(materialized.materialization_record)
        logical = dict(record["logical"])
        logical["entrypoint"] = "scripts/experiments/method_child.py"
        record["logical"] = logical
        return replace(materialized, materialization_record=record)

    monkeypatch.setattr(
        runner_module,
        "materialize_method_execution",
        tamper_entrypoint,
    )
    monkeypatch.setattr(
        runner_module,
        "_run_method_child",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("tampered logical record reached child")
        ),
    )

    outcome = run_request(request, artifact_root)

    assert outcome.status == "failed"
    failure = json.loads(
        (outcome.diagnostic_dir / "failure.json").read_text("utf-8")
    )
    assert "logical record disagrees" in failure["message"]


def test_tampered_source_snapshot_fails_preflight_before_child_or_artifacts(
    tmp_path: Path,
    monkeypatch,
):
    request, plan = _real_request_and_plan(tmp_path)
    request, plan = _request_plan_with_repo_snapshot(
        request,
        plan,
        tmp_path / "snapshot-artifacts",
    )
    source = plan.source_root / "gsdiff/__init__.py"
    source.write_bytes(source.read_bytes() + b"\n# tampered\n")
    artifact_root = tmp_path / "run-artifacts"

    def forbidden_child(*args, **kwargs):
        raise AssertionError("tampered snapshot reached child")

    monkeypatch.setattr(runner_module, "_run_method_child", forbidden_child)

    with pytest.raises(ValueError, match="snapshot.*changed|file bytes"):
        run_request(request, artifact_root)

    assert not (artifact_root / ".claims").exists()
    assert not (artifact_root / "runs").exists()
    assert not (artifact_root / "failed").exists()
    assert not (artifact_root / "work").exists()


def test_expected_source_projection_tamper_fails_before_materialization(
    tmp_path: Path,
    monkeypatch,
):
    request, plan = _real_request_and_plan(tmp_path)
    request, plan = _request_plan_with_repo_snapshot(
        request,
        plan,
        tmp_path / "snapshot-artifacts",
    )
    plan = replace(plan, expected_source_snapshot_sha256="e" * 64)
    request = replace(request, execution_plan=plan)
    artifact_root = tmp_path / "run-artifacts"

    def forbidden(*args, **kwargs):
        raise AssertionError("projection tamper reached materialization")

    monkeypatch.setattr(runner_module, "materialize_method_execution", forbidden)

    with pytest.raises(ValueError, match="projection"):
        run_request(request, artifact_root)

    assert not (artifact_root / ".claims").exists()
    assert not (artifact_root / "runs").exists()
    assert not (artifact_root / "failed").exists()
    assert not (artifact_root / "work").exists()


def test_source_plan_rejects_self_described_snapshot_from_untrusted_repository(
    tmp_path: Path,
):
    attacker_repo = tmp_path / "attacker-repo"
    (attacker_repo / "gsdiff").mkdir(parents=True)
    (attacker_repo / "gsdiff/module.py").write_text(
        "ATTACKER_CONTROLLED = True\n",
        encoding="utf-8",
    )
    (attacker_repo / "scripts/experiments").mkdir(parents=True)
    (attacker_repo / "schemas").mkdir()
    for relative in (
        "train.py",
        "scripts/run_baselines.py",
        "scripts/experiments/method_child_bootstrap.py",
        "schemas/method-info-v2.schema.json",
    ):
        path = attacker_repo / relative
        path.write_text("{}\n" if path.suffix == ".json" else "# attacker\n")
    subprocess.run(["git", "init", "-q"], cwd=attacker_repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "runner-test@example.invalid"],
        cwd=attacker_repo,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Runner Test"],
        cwd=attacker_repo,
        check=True,
    )
    subprocess.run(["git", "add", "--all"], cwd=attacker_repo, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "attacker snapshot"],
        cwd=attacker_repo,
        check=True,
    )
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=attacker_repo,
        text=True,
    ).strip()
    snapshot = materialize_source_snapshot(
        attacker_repo,
        tmp_path / "snapshot-artifacts",
        commit,
        (Path("gsdiff"), Path("scripts"), Path("schemas"), Path("train.py")),
    )
    inventory, digest = selected_source_evidence(snapshot)
    plan = replace(
        _plan(),
        source_root=snapshot.root,
        code_commit=commit,
        source_snapshot=snapshot,
        expected_source_inventory=inventory,
        expected_source_snapshot_sha256=digest,
    )

    with pytest.raises(ValueError, match="trusted|Git|repository"):
        runner_module._validate_source_plan(plan)


@pytest.mark.parametrize("winner_valid", [True, False])
def test_promotion_race_revalidates_winner_and_never_clobbers_it(
    tmp_path: Path, monkeypatch, winner_valid: bool
):
    request, plan = _real_request_and_plan(tmp_path)
    artifact_root = tmp_path / "run-artifacts"
    final_dir = artifact_root / "runs" / request.identity.identity_sha256
    real_promote = runner_module._promote_exact_directory_no_clobber

    def inject_winner(source, destination, **identity):
        if destination == final_dir:
            if winner_valid:
                shutil.copytree(source, destination)
            else:
                destination.mkdir()
                (destination / "results.json").write_text("{}", encoding="utf-8")
            raise FileExistsError("injected promotion race")
        return real_promote(source, destination, **identity)

    monkeypatch.setattr(
        runner_module,
        "_promote_exact_directory_no_clobber",
        inject_winner,
    )

    if winner_valid:
        outcome = run_request(request, artifact_root)
        assert outcome == RunOutcome("cached", final_dir, None, 0)
        assert reusable_run(artifact_root, request.identity) == final_dir
    else:
        with pytest.raises(ValueError, match="winner failed integrity"):
            run_request(request, artifact_root)
        assert (final_dir / "results.json").read_text("utf-8") == "{}"
        stages = list((artifact_root / "runs").glob("*.tmp-*"))
        assert len(stages) == 1
        assert json.loads((stages[0] / "manifest.json").read_text("utf-8"))[
            "status"
        ] == "complete"
        assert not (stages[0] / "failure.json").exists()
        assert not list((artifact_root / "failed").iterdir())
    if winner_valid:
        assert not list((artifact_root / "runs").glob("*.tmp-*"))


@pytest.mark.skipif(os.name != "nt", reason="Windows handle-bound promotion")
def test_run_promotion_renames_pinned_directory_not_swapped_path(
    tmp_path: Path,
    monkeypatch,
):
    request, plan = _real_request_and_plan(tmp_path)
    artifact_root = tmp_path / "run-artifacts"
    swapped_paths: list[Path] = []

    def swap_after_handle_open(source, destination):
        assert destination.name == request.identity.identity_sha256
        displaced = source.with_name(source.name + ".displaced")
        os.rename(source, displaced)
        source.mkdir()
        (source / "poison.txt").write_text("poison", encoding="ascii")
        swapped_paths.append(source)

    monkeypatch.setattr(
        artifact_persistence_module,
        "_handle_bound_promotion_barrier",
        swap_after_handle_open,
    )

    outcome = run_request(request, artifact_root)

    assert outcome.status == "complete"
    assert outcome.run_dir is not None
    assert not (outcome.run_dir / "poison.txt").exists()
    assert reusable_run(artifact_root, request.identity) == outcome.run_dir
    assert len(swapped_paths) == 1
    assert (swapped_paths[0] / "poison.txt").read_text("ascii") == "poison"


def test_running_manifest_and_lifecycle_are_durable_before_child(
    tmp_path: Path,
    monkeypatch,
):
    request, plan = _real_request_and_plan(tmp_path)
    artifact_root = tmp_path / "run-artifacts"
    observed: dict[str, object] = {}

    def inspect_running_anchor(**_kwargs):
        stages = list((artifact_root / "runs").glob("*.tmp-*"))
        assert len(stages) == 1
        stage = stages[0]
        resolved = json.loads(
            (stage / "resolved-config.json").read_text("utf-8")
        )
        lifecycle = json.loads((stage / "lifecycle.json").read_text("utf-8"))
        manifest = json.loads((stage / "manifest.json").read_text("utf-8"))
        assert resolved == plan.config_resolved
        assert lifecycle == {
            "schema": "run-lifecycle-v1",
            "state": "running",
            "identity_sha256": request.identity.identity_sha256,
            "owner_token": lifecycle["owner_token"],
            "fence": lifecycle["fence"],
        }
        assert manifest["status"] == "running"
        assert manifest["config"]["resolved"] == plan.config_resolved
        assert manifest["execution"]["return_code"] is None
        assert manifest["execution"]["ended_at_utc"] is None
        assert manifest["metrics"] is None
        assert manifest["artifacts"] == []
        validate_manifest(manifest, identity=request.identity)
        observed["anchor"] = True
        raise RuntimeError("injected child-boundary stop")

    monkeypatch.setattr(runner_module, "_run_method_child", inspect_running_anchor)

    outcome = run_request(request, artifact_root)

    assert observed == {"anchor": True}
    assert outcome.status == "failed"
    assert outcome.diagnostic_dir is not None


def test_complete_manifest_swap_is_last_stage_mutation_and_never_gets_failure(
    tmp_path: Path,
    monkeypatch,
):
    request, plan = _real_request_and_plan(tmp_path)
    artifact_root = tmp_path / "run-artifacts"
    real_replace = runner_module.os.replace
    observed: dict[str, Path] = {}

    def replace_then_fail(source, destination):
        source_path = Path(source)
        destination_path = Path(destination)
        if destination_path.name == "manifest.json":
            candidate = json.loads(source_path.read_text("utf-8"))
            if candidate.get("status") == "complete":
                stage = destination_path.parent
                assert not (stage / "child-work.json").exists()
                assert not (stage / ".owner.json").exists()
                lifecycle = json.loads(
                    (stage / "lifecycle.json").read_text("utf-8")
                )
                assert lifecycle["state"] == "complete"
                assert not list((artifact_root / "work").iterdir())
                real_replace(source, destination)
                observed["stage"] = stage
                raise OSError("injected error after complete marker")
        return real_replace(source, destination)

    monkeypatch.setattr(runner_module.os, "replace", replace_then_fail)

    with pytest.raises(OSError, match="after complete marker"):
        run_request(request, artifact_root)

    stage = observed["stage"]
    manifest = json.loads((stage / "manifest.json").read_text("utf-8"))
    assert manifest["status"] == "complete"
    assert not (stage / "failure.json").exists()
    assert not list((artifact_root / "failed").iterdir())


@pytest.mark.parametrize(
    ("name", "domain"),
    [("lifecycle.json", "lc"), ("manifest.json", "mf")],
)
def test_durable_replacement_uses_short_collision_safe_sibling(
    tmp_path: Path,
    monkeypatch,
    name: str,
    domain: str,
):
    parent = tmp_path
    while len(str(parent.absolute())) < 225:
        remaining = 225 - len(str(parent.absolute())) - 1
        segment = "p" * min(40, remaining)
        parent = parent / segment
        parent.mkdir()
    destination = parent / name
    destination.write_bytes(b"old")
    legacy_temporary = destination.with_name(
        f".{destination.name}.tmp-{'f' * 32}"
    )
    short_collision = destination.with_name(f".{domain}-{'a' * 12}.tmp")
    assert len(str(legacy_temporary.absolute())) >= 260
    assert len(str(short_collision.absolute())) < 260
    short_collision.write_bytes(b"collision-owner")
    tokens = iter(("a" * 12, "b" * 12))
    monkeypatch.setattr(
        runner_module.secrets,
        "token_hex",
        lambda _size: next(tokens),
    )

    runner_module._replace_file_durable(destination, b"replacement")

    assert destination.read_bytes() == b"replacement"
    assert short_collision.read_bytes() == b"collision-owner"
    assert not destination.with_name(f".{domain}-{'b' * 12}.tmp").exists()


def test_post_rename_sync_failure_is_explicit_durability_error(
    tmp_path: Path,
    monkeypatch,
):
    request, plan = _real_request_and_plan(tmp_path)
    artifact_root = tmp_path / "run-artifacts"
    runs_dir = artifact_root / "runs"
    final_dir = runs_dir / request.identity.identity_sha256
    real_sync = runner_module._sync_directory

    def fail_post_rename_sync(path):
        if Path(path) == runs_dir and final_dir.exists():
            raise OSError("injected post-rename sync failure")
        return real_sync(path)

    monkeypatch.setattr(runner_module, "_sync_directory", fail_post_rename_sync)

    with pytest.raises(RuntimeError, match="durability"):
        run_request(request, artifact_root)

    assert final_dir.is_dir()
    assert reusable_run(artifact_root, request.identity) == final_dir
    assert not list((artifact_root / "failed").iterdir())
    assert run_request(request, artifact_root).status == "cached"


def test_post_rename_transient_validation_failure_returns_complete(
    tmp_path: Path,
    monkeypatch,
):
    request, plan = _real_request_and_plan(tmp_path)
    artifact_root = tmp_path / "run-artifacts"
    final_dir = artifact_root / "runs" / request.identity.identity_sha256
    real_reusable = runner_module.reusable_run
    injected = False

    def fail_first_post_rename_validation(root, identity):
        nonlocal injected
        if final_dir.exists() and not injected:
            injected = True
            raise ValueError("injected transient final validation failure")
        return real_reusable(root, identity)

    monkeypatch.setattr(
        runner_module,
        "reusable_run",
        fail_first_post_rename_validation,
    )

    outcome = run_request(request, artifact_root)

    assert injected is True
    assert outcome == RunOutcome("complete", final_dir, None, 0)
    assert real_reusable(artifact_root, request.identity) == final_dir
    assert not list((artifact_root / "failed").iterdir())
    assert run_request(request, artifact_root).status == "cached"


def test_post_rename_invalid_final_is_explicit_integrity_error_and_retained(
    tmp_path: Path,
    monkeypatch,
):
    request, plan = _real_request_and_plan(tmp_path)
    artifact_root = tmp_path / "run-artifacts"
    runs_dir = artifact_root / "runs"
    final_dir = runs_dir / request.identity.identity_sha256
    real_sync = runner_module._sync_directory
    corrupted = False

    def corrupt_after_rename(path):
        nonlocal corrupted
        if Path(path) == runs_dir and final_dir.exists() and not corrupted:
            corrupted = True
            (final_dir / "outputs/metrics.json").write_bytes(b"{}")
        return real_sync(path)

    monkeypatch.setattr(runner_module, "_sync_directory", corrupt_after_rename)

    with pytest.raises(RuntimeError, match="integrity"):
        run_request(request, artifact_root)

    assert corrupted is True
    assert final_dir.is_dir()
    assert not list((artifact_root / "failed").iterdir())
    with pytest.raises(ValueError):
        reusable_run(artifact_root, request.identity)
    with pytest.raises(ValueError):
        run_request(request, artifact_root)


def test_two_processes_same_identity_execute_once_and_loser_reuses(tmp_path: Path):
    context = multiprocessing.get_context("spawn")
    claims = tmp_path / "claims"
    claims.mkdir()
    marker = tmp_path / "executed"
    queue = context.Queue()
    identity = _identity()
    workers = [
        context.Process(
            target=_claim_once_worker,
            args=(claims, identity, marker, queue),
        )
        for _ in range(2)
    ]

    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(15)

    assert [worker.exitcode for worker in workers] == [0, 0]
    assert sorted([queue.get(timeout=2), queue.get(timeout=2)]) == [
        "cached",
        "complete",
    ]
    assert marker.read_text("ascii").isdigit()


@pytest.mark.skipif(os.name != "nt", reason="Windows claim wait contract")
def test_claim_waits_beyond_the_msvcrt_retry_window(tmp_path: Path):
    context = multiprocessing.get_context("spawn")
    claims = tmp_path / "claims"
    claims.mkdir()
    ready = context.Event()
    identity = _identity()
    worker = context.Process(
        target=_hold_claim_for_worker,
        args=(claims, identity, ready, 11.5),
    )
    worker.start()
    assert ready.wait(10)
    started = time.monotonic()
    try:
        with _claim_identity(claims, identity) as recovered:
            elapsed = time.monotonic() - started
            assert recovered["owner_token"]
    finally:
        worker.join(20)
        if worker.is_alive():
            worker.terminate()
            worker.join(10)

    assert worker.exitcode == 0
    assert elapsed >= 10.5


def test_killed_owner_is_os_released_then_stale_claim_is_fenced(tmp_path: Path):
    context = multiprocessing.get_context("spawn")
    claims = tmp_path / "claims"
    claims.mkdir()
    ready = context.Event()
    identity = _identity()
    worker = context.Process(
        target=_hold_claim_worker,
        args=(claims, identity, ready),
    )
    worker.start()
    assert ready.wait(10)
    worker.terminate()
    worker.join(10)
    assert worker.exitcode not in (None, 0)

    with _claim_identity(claims, identity) as recovered:
        assert recovered["fence"] == 2
        assert recovered["owner_token"]
        assert len(list(claims.glob(f"{identity.identity_sha256}.stale-*.json"))) == 1


@pytest.mark.parametrize("owner_payload", [None, b"{"])
def test_interrupted_stage_owner_publication_is_quarantined(
    tmp_path: Path,
    owner_payload: bytes | None,
):
    claims = tmp_path / ".claims"
    runs = tmp_path / "runs"
    failed = tmp_path / "failed"
    claims.mkdir()
    runs.mkdir()
    failed.mkdir()
    identity = _identity()
    stage = runs / f"{identity.identity_sha256}.tmp-{uuid.uuid4()}"

    with pytest.raises(KeyboardInterrupt):
        with _claim_identity(claims, identity):
            stage.mkdir()
            if owner_payload is not None:
                (stage / ".owner.json").write_bytes(owner_payload)
            raise KeyboardInterrupt("interrupted owner publication")

    with _claim_identity(claims, identity) as recovered:
        assert recovered["fence"] == 2
        runner_module._recover_stale_stages(
            runs_dir=runs,
            failed_dir=failed,
            claims_dir=claims,
            identity=identity,
            current_claim=recovered,
        )

    assert not stage.exists()
    recovered_stages = list(failed.glob(f"{identity.run_id}-recovered-*"))
    assert len(recovered_stages) == 1
    if owner_payload is None:
        assert list(recovered_stages[0].iterdir()) == []
    else:
        assert (recovered_stages[0] / ".owner.json").read_bytes() == owner_payload


@pytest.mark.parametrize("pointer_finalized", [False, True])
def test_recovery_reuses_already_finalized_diagnostic_evidence(
    tmp_path: Path,
    pointer_finalized: bool,
):
    claims = tmp_path / ".claims"
    runs = tmp_path / "runs"
    failed = tmp_path / "failed"
    claims.mkdir()
    runs.mkdir()
    failed.mkdir()
    identity = _identity()
    old_claim = {
        "schema": "run-claim-v1",
        "identity_sha256": identity.identity_sha256,
        "owner_token": "a" * 32,
        "fence": 1,
        "pid": 7,
    }
    stale = claims / f"{identity.identity_sha256}.stale-{uuid.uuid4().hex}.json"
    stale.write_bytes(canonical_json_bytes(old_claim))
    current_claim = {
        "schema": "run-claim-v1",
        "identity_sha256": identity.identity_sha256,
        "owner_token": "b" * 32,
        "fence": 2,
        "pid": 8,
    }
    stage = runs / f"{identity.identity_sha256}.tmp-{uuid.uuid4()}"
    stage.mkdir()
    (stage / ".owner.json").write_bytes(
        canonical_json_bytes(
            {
                "schema": "run-stage-owner-v1",
                "owner_token": old_claim["owner_token"],
                "fence": old_claim["fence"],
                "identity_sha256": identity.identity_sha256,
            }
        )
    )
    pointer = {
        "schema": "run-child-work-v1",
        "work_id": "d" * 32,
        "owner_token": old_claim["owner_token"],
        "fence": old_claim["fence"],
        "work_device": 123,
        "work_inode": 456,
    }
    if pointer_finalized:
        pointer["retained"] = False
    (stage / "child-work.json").write_bytes(canonical_json_bytes(pointer))
    evidence_paths = (
        "outputs/stdout.log",
        "outputs/stderr.log",
        "evidence/audit.jsonl",
        "evidence/resource-sampling.json",
    )
    empty_digest = hashlib.sha256(b"").hexdigest()
    present = {}
    for relative in evidence_paths:
        path = stage / relative
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"")
        present[relative] = {"sha256": empty_digest, "size_bytes": 0}
    (stage / "diagnostic-artifacts.json").write_bytes(
        canonical_json_bytes(
            {
                "schema": "run-diagnostic-artifacts-v1",
                "present": present,
                "missing_expected": [],
                "retained": False,
            }
        )
    )
    evidence_before = {
        relative: (stage / relative).read_bytes() for relative in evidence_paths
    }
    inventory_before = (stage / "diagnostic-artifacts.json").read_bytes()

    runner_module._recover_stale_stages(
        runs_dir=runs,
        failed_dir=failed,
        claims_dir=claims,
        identity=identity,
        current_claim=current_claim,
    )

    assert not stage.exists()
    recovered_stages = list(failed.glob(f"{identity.run_id}-recovered-*"))
    assert len(recovered_stages) == 1
    assert (
        recovered_stages[0] / "diagnostic-artifacts.json"
    ).read_bytes() == inventory_before
    for relative, payload in evidence_before.items():
        assert (recovered_stages[0] / relative).read_bytes() == payload
    recovered_pointer = json.loads(
        (recovered_stages[0] / "child-work.json").read_text("utf-8")
    )
    assert recovered_pointer == {**pointer, "retained": False}


def test_claim_rotation_crash_preserves_monotonic_fence_and_recovers_stage(
    tmp_path: Path,
    monkeypatch,
):
    claims = tmp_path / ".claims"
    runs = tmp_path / "runs"
    failed = tmp_path / "failed"
    claims.mkdir()
    runs.mkdir()
    failed.mkdir()
    identity = _identity()
    old_claim = {
        "schema": "run-claim-v1",
        "identity_sha256": identity.identity_sha256,
        "owner_token": "5" * 32,
        "fence": 5,
        "pid": 7,
    }
    metadata = claims / f"{identity.identity_sha256}.json"
    metadata.write_bytes(canonical_json_bytes(old_claim))
    stage = runs / f"{identity.identity_sha256}.tmp-{uuid.uuid4()}"
    stage.mkdir()
    (stage / ".owner.json").write_bytes(
        canonical_json_bytes(
            {
                "schema": "run-stage-owner-v1",
                "owner_token": old_claim["owner_token"],
                "fence": old_claim["fence"],
                "identity_sha256": identity.identity_sha256,
            }
        )
    )

    def crash_after_rotation(_stale_path, active_path):
        assert not active_path.exists()
        raise KeyboardInterrupt("injected rotation crash")

    monkeypatch.setattr(
        runner_module,
        "_claim_rotation_barrier",
        crash_after_rotation,
    )
    with pytest.raises(KeyboardInterrupt, match="rotation crash"):
        with _claim_identity(claims, identity):
            pass
    assert not metadata.exists()
    stale = list(claims.glob(f"{identity.identity_sha256}.stale-*.json"))
    assert len(stale) == 1
    monkeypatch.setattr(
        runner_module,
        "_claim_rotation_barrier",
        lambda _stale_path, _active_path: None,
    )

    with _claim_identity(claims, identity) as recovered:
        assert recovered["fence"] == 6
        runner_module._recover_stale_stages(
            runs_dir=runs,
            failed_dir=failed,
            claims_dir=claims,
            identity=identity,
            current_claim=recovered,
        )

    assert not stage.exists()
    recovered_stages = list(failed.glob(f"{identity.run_id}-recovered-*"))
    assert len(recovered_stages) == 1
    assert (recovered_stages[0] / ".owner.json").is_file()


@pytest.mark.skipif(os.name != "nt", reason="Windows parent-kill contract")
def test_parent_hard_kill_stops_child_then_recovers_once(tmp_path: Path):
    context = multiprocessing.get_context("spawn")
    artifact_root = tmp_path / "run-artifacts"
    started = tmp_path / "method-started"
    survived = tmp_path / "method-survived"
    queue = context.Queue()
    worker = context.Process(
        target=_hard_kill_run_worker,
        args=(tmp_path, artifact_root, started, survived, queue),
    )
    worker.start()
    identity_sha256 = queue.get(timeout=30)
    deadline = time.monotonic() + 30
    while not started.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert started.is_file()

    worker.terminate()
    worker.join(timeout=10)
    assert worker.exitcode not in (None, 0)
    time.sleep(5.5)
    assert not survived.exists()

    request, plan = _real_request_and_plan(tmp_path)
    assert request.identity.identity_sha256 == identity_sha256
    outcome = run_request(request, artifact_root)
    assert outcome.status == "complete"
    recovered = list((artifact_root / "failed").glob("*-recovered-*"))
    assert len(recovered) == 1
    assert not list((artifact_root / "work").iterdir())

    cached = run_request(request, artifact_root)
    assert cached.status == "cached"
    assert len(list((artifact_root / "failed").glob("*-recovered-*"))) == 1


def test_former_owner_cannot_pass_fence_after_token_replacement(tmp_path: Path):
    claims = tmp_path / "claims"
    claims.mkdir()
    identity = _identity()
    metadata = claims / f"{identity.identity_sha256}.json"

    with pytest.raises(ValueError, match="token|fence"):
        with _claim_identity(claims, identity) as claim:
            replacement = {**claim, "owner_token": "0" * 32, "fence": 2}
            metadata.write_bytes(canonical_json_bytes(replacement))
            _verify_claim(claim, path=metadata)


def test_final_rename_interruption_is_recovered_before_retry_promotes(
    tmp_path: Path, monkeypatch
):
    request, plan = _real_request_and_plan(tmp_path)
    artifact_root = tmp_path / "run-artifacts"
    real_promote = runner_module._promote_exact_directory_no_clobber
    interrupted = False

    def interrupt_final_rename(source, destination, **identity):
        nonlocal interrupted
        if destination.name == request.identity.identity_sha256 and not interrupted:
            interrupted = True
            raise KeyboardInterrupt
        return real_promote(source, destination, **identity)

    monkeypatch.setattr(
        runner_module,
        "_promote_exact_directory_no_clobber",
        interrupt_final_rename,
    )
    with pytest.raises(KeyboardInterrupt):
        run_request(request, artifact_root)
    assert len(list((artifact_root / "runs").glob("*.tmp-*"))) == 1

    outcome = run_request(request, artifact_root)

    assert outcome.status == "complete"
    assert not list((artifact_root / "runs").glob("*.tmp-*"))
    recovered = list((artifact_root / "failed").glob("*-recovered-*"))
    assert len(recovered) == 1


def test_reusable_run_accepts_only_a_fully_valid_canonical_run(tmp_path: Path):
    identity = _complete_identity()
    run_dir = _write_complete_run(tmp_path, identity)

    assert reusable_run(tmp_path, identity) == run_dir


def _rewrite_run_manifest(run_dir: Path, manifest: dict[str, object]) -> None:
    (run_dir / "manifest.json").write_bytes(canonical_json_bytes(manifest))


def _replace_declared_artifact(
    run_dir: Path,
    manifest: dict[str, object],
    *,
    role: str,
    payload: bytes,
) -> None:
    descriptor = next(
        item for item in manifest["artifacts"] if item["role"] == role
    )
    path = run_dir / descriptor["path"]
    path.write_bytes(payload)
    descriptor["sha256"] = hashlib.sha256(payload).hexdigest()
    descriptor["size_bytes"] = len(payload)


def _consume_complete_run(
    consumer: str,
    artifact_root: Path,
    run_dir: Path,
    identity,
):
    if consumer == "direct":
        manifest = json.loads(
            (run_dir / "manifest.json").read_text("utf-8")
        )
        manifest_module._validate_complete_runner_contract(
            run_dir,
            manifest,
            artifact_root=artifact_root,
        )
        return True
    if consumer == "runner":
        return reusable_run(artifact_root, identity)
    if consumer == "loader":
        return load_complete_manifest(
            run_dir / "manifest.json",
            artifact_root=artifact_root,
            expected_identity_sha256=identity.identity_sha256,
        )
    payload = identity.payload()
    return build_aggregate_index(
        campaign_id="test-v1",
        campaign_sha256="7" * 64,
        protocol_sha256="8" * 64,
        scientific_contract_id=payload["scientific_contract_id"],
        scientific_contract_sha256=payload[
            "scientific_contract_sha256"
        ],
        metric_version="metrics-v1",
        expected_identity_sha256s=[identity.identity_sha256],
        manifest_paths={identity.identity_sha256: run_dir / "manifest.json"},
        artifact_root=artifact_root,
    )


@pytest.mark.parametrize(
    "consumer",
    ["direct", "runner", "loader", "aggregate"],
)
def test_complete_consumers_accept_the_exact_runner_contract(
    tmp_path: Path,
    consumer: str,
):
    identity = _complete_identity()
    run_dir = _write_complete_run(tmp_path, identity)

    assert _consume_complete_run(consumer, tmp_path, run_dir, identity) is not None


@pytest.mark.parametrize(
    "consumer",
    ["direct", "runner", "loader", "aggregate"],
)
def test_complete_consumers_reject_resigned_plausible_metrics_forgery(
    tmp_path: Path,
    consumer: str,
):
    identity = _complete_identity()
    run_dir = _write_complete_run(tmp_path, identity)
    manifest = json.loads((run_dir / "manifest.json").read_text("utf-8"))
    metrics_path = run_dir / "outputs/metrics.json"
    metrics = json.loads(metrics_path.read_text("utf-8"))
    metrics.update(
        {
            "psnr_global_affine": 1.2345,
            "ssim_global_affine": 0.0,
            "nrmse_global_affine_l2": 0.5,
            "psnr_legacy_per_frame_minmax": 2.3456,
            "alignment": {"slope": 123.0, "intercept": -99.0},
        }
    )
    payload = canonical_json_bytes(metrics)
    metrics_path.write_bytes(payload)
    manifest["metrics"]["sha256"] = hashlib.sha256(payload).hexdigest()
    _rewrite_run_manifest(run_dir, manifest)

    with pytest.raises(ValueError, match="metrics|dataset|truth"):
        _consume_complete_run(consumer, tmp_path, run_dir, identity)


@pytest.mark.parametrize("consumer", ["runner", "loader", "aggregate"])
def test_complete_consumers_reject_dataset_protocol_semantic_mismatch(
    tmp_path: Path,
    consumer: str,
):
    identity = _complete_identity(motion_id="forged-motion")
    run_dir = _write_complete_run(tmp_path, identity)

    with pytest.raises(ValueError, match="dataset|motion|protocol"):
        _consume_complete_run(consumer, tmp_path, run_dir, identity)


@pytest.mark.parametrize("consumer", ["runner", "loader", "aggregate"])
def test_complete_consumers_reject_resigned_measurement_relabel(
    tmp_path: Path,
    consumer: str,
):
    identity = _complete_identity()
    run_dir = _write_complete_run(tmp_path, identity)
    manifest = json.loads((run_dir / "manifest.json").read_text("utf-8"))
    manifest["measurement"]["requested_snr_db"] = 30
    manifest["measurement"]["realized_train_snr_db"] = 30.125
    _rewrite_run_manifest(run_dir, manifest)

    with pytest.raises(ValueError, match="measurement|dataset input"):
        _consume_complete_run(consumer, tmp_path, run_dir, identity)


@pytest.mark.parametrize(
    "consumer",
    ["direct", "runner", "loader", "aggregate"],
)
def test_complete_consumers_reject_resigned_runtime_relabel(
    tmp_path: Path,
    consumer: str,
):
    identity = _complete_identity()
    run_dir = _write_complete_run(tmp_path, identity)
    manifest = json.loads((run_dir / "manifest.json").read_text("utf-8"))
    manifest["runtime"]["python"] = "9.99-forged"
    _rewrite_run_manifest(run_dir, manifest)

    with pytest.raises(ValueError, match="runtime|identity contract"):
        _consume_complete_run(consumer, tmp_path, run_dir, identity)


@pytest.mark.parametrize(
    "consumer",
    ["direct", "runner", "loader", "aggregate"],
)
def test_complete_consumers_reject_fully_resigned_fake_runtime(
    tmp_path: Path,
    consumer: str,
):
    original_identity = _complete_identity()
    run_dir = _write_complete_run(tmp_path, original_identity)
    old_manifest = json.loads((run_dir / "manifest.json").read_text("utf-8"))
    config = json.loads((run_dir / "resolved-config.json").read_text("utf-8"))
    forged_runtime = {name: f"forged-{name}" for name in _COMPLETE_RUNTIME}
    config["runner_execution"]["runtime_contract"] = forged_runtime
    payload = dict(original_identity.payload())
    payload["config_sha256"] = resolved_config_sha256(config)
    forged_identity = build_run_identity(**payload)
    forged_run_dir = run_dir.parent / forged_identity.identity_sha256
    run_dir.rename(forged_run_dir)
    config_payload = canonical_json_bytes(config)
    (forged_run_dir / "resolved-config.json").write_bytes(config_payload)
    lifecycle = json.loads(
        (forged_run_dir / "lifecycle.json").read_text("utf-8")
    )
    lifecycle["identity_sha256"] = forged_identity.identity_sha256
    lifecycle_payload = canonical_json_bytes(lifecycle)
    (forged_run_dir / "lifecycle.json").write_bytes(lifecycle_payload)
    artifacts = json.loads(canonical_json_bytes(old_manifest["artifacts"]))
    replacements = {
        "resolved-config": config_payload,
        "lifecycle": lifecycle_payload,
    }
    for descriptor in artifacts:
        replacement = replacements.get(descriptor["role"])
        if replacement is not None:
            descriptor["sha256"] = hashlib.sha256(replacement).hexdigest()
            descriptor["size_bytes"] = len(replacement)
    forged_manifest = build_manifest(
        status="complete",
        identity=forged_identity,
        config_resolved=config,
        inputs={
            name: old_manifest["inputs"][name]
            for name in (
                "measurements_file_sha256",
                "evaluation_truth_file_sha256",
                "dataset_manifest_sha256",
            )
        },
        runtime=forged_runtime,
        execution=old_manifest["execution"],
        measurement=old_manifest["measurement"],
        metrics=old_manifest["metrics"],
        artifacts=artifacts,
    )
    _rewrite_run_manifest(forged_run_dir, forged_manifest)

    with pytest.raises(ValueError, match="runtime|authoritative"):
        _consume_complete_run(
            consumer,
            tmp_path,
            forged_run_dir,
            forged_identity,
        )


@pytest.mark.parametrize("consumer", ["runner", "loader", "aggregate"])
def test_complete_consumers_reject_resigned_missing_required_dgi(
    tmp_path: Path,
    consumer: str,
):
    identity = _complete_identity()
    run_dir = _write_complete_run(tmp_path, identity)
    manifest = json.loads((run_dir / "manifest.json").read_text("utf-8"))
    reconstruction_path = run_dir / "outputs/reconstruction.npz"
    loaded = child_outputs_module.load_reconstruction_v2(reconstruction_path)
    arrays = {
        "reconstruction": loaded.reconstruction,
        "frame_indices": loaded.frame_indices,
        "time_grid": loaded.time_grid,
    }
    metadata = child_outputs_module._reconstruction_metadata(
        method_id="dgi",
        acquisition=_complete_fixture_acquisition(),
        arrays=arrays,
    )
    reconstruction_payload = child_outputs_module.npz_bytes(
        arrays=arrays,
        metadata=metadata,
    )
    _replace_declared_artifact(
        run_dir,
        manifest,
        role="reconstruction",
        payload=reconstruction_payload,
    )
    info = json.loads(
        (run_dir / "outputs/method-info.json").read_text("utf-8")
    )
    info["reconstruction"]["sha256"] = hashlib.sha256(
        reconstruction_payload
    ).hexdigest()
    info["reconstruction"]["array_descriptors"] = metadata[
        "array_descriptors"
    ]
    _replace_declared_artifact(
        run_dir,
        manifest,
        role="method-info",
        payload=canonical_json_bytes(info),
    )
    _rewrite_run_manifest(run_dir, manifest)

    with pytest.raises(ValueError, match="dgi|auxiliary"):
        _consume_complete_run(consumer, tmp_path, run_dir, identity)


def test_complete_fixture_uses_exact_production_materialization_logical_shape(
    tmp_path: Path,
):
    run_dir = _write_complete_run(tmp_path, _complete_identity())
    logical = json.loads(
        (run_dir / "evidence/materialization-logical.json").read_text("utf-8")
    )

    assert set(logical) == {
        "schema",
        "method_id",
        "method_config_id",
        "execution_profile",
        "method_config_sha256",
        "methods_registry_protocol_sha256",
        "semantic_sha256",
        "materialized_config_sha256",
        "dataset_identity_sha256",
        "measurements_file_sha256",
        "expected_acquisition_spec",
        "algorithm_seed",
        "checkpoint_sha256",
        "source_inventory",
        "source_snapshot_sha256",
        "requested_runtime_device",
        "child_runtime_device",
        "entrypoint",
        "command_template",
    }


@pytest.mark.parametrize(
    "method_id",
    ["static_cs", "gsdiff_tv", "gsdiff_diffusion"],
)
def test_identity_method_info_contract_accepts_selection_and_neural_controls(
    tmp_path: Path,
    method_id: str,
):
    base_config = (
        {"gaussian_count": 1000}
        if method_id.startswith("gsdiff_")
        else {}
    )
    method = resolve_method_semantics(
        method_id,
        method_config_id="default",
        base_config=base_config,
        measurements_metadata={},
        execution_profile="publication-v1",
    )
    acquisition = _complete_fixture_acquisition()
    contract = child_outputs_module.build_method_info_contract_v1(
        method,
        blind_acquisition_spec(acquisition),
    )
    selection_contract = contract["selection"]
    if selection_contract["candidate_grid"] is None:
        selected_hyperparameters = None
        selection = None
    else:
        grid = selection_contract["candidate_grid"]
        rows = [
            {
                "candidate": candidate,
                "formula_id": "heldout-normalized-l2-v1",
                "numerator": float(index + 1),
                "denominator": 1.0,
                "value": float(index + 1),
            }
            for index, candidate in enumerate(grid)
        ]
        selection = {
            "formula_id": "heldout-normalized-l2-v1",
            "candidate_grid": grid,
            "selected_candidate": grid[0],
            "rows": rows,
        }
        keys = selection_contract["selected_hyperparameter_keys"]
        assert keys == ["lambda"]
        selected_hyperparameters = {"lambda": grid[0]}
    motion_policy = contract["motion_estimate"]
    motion_required = motion_policy["presence"] == "required"
    native_iteration = contract["native_iteration"]
    seed = derive_algorithm_seed(
        cell_seed=7,
        dataset_identity_sha256=acquisition.dataset_identity_sha256,
        method_id=method.method_id,
        method_config_sha256=method.method_config_sha256,
    )
    output_dir = tmp_path / method_id
    history_kind, history_index = (
        ("outer-iteration", "outer_iteration")
        if method_id.startswith("gsdiff_")
        else ("iteration", "iteration")
    )
    history_budget = native_iteration["budget"]
    history = tuple(
        {
            "kind": history_kind,
            history_index: index + 1,
            **_fixture_history_metrics(method, index + 1, history_budget),
        }
        for index in range(history_budget)
    )
    child_outputs_module.write_method_child_outputs_v2(
        output_dir,
        method=method,
        acquisition=acquisition,
        measurements_file_sha256="a" * 64,
        algorithm_seed=seed,
        result=child_outputs_module.MethodChildResult(
            method_id=method.method_id,
            reconstruction=np.ones(
                (acquisition.T, acquisition.H, acquisition.W),
                dtype=np.float32,
            ),
            estimated_motion_trajectory=(
                np.zeros((4, 3), dtype=np.float32)
                if motion_required
                else None
            ),
            dgi=(
                np.ones((acquisition.H, acquisition.W), dtype=np.float32)
                if contract["auxiliary_arrays"]["dgi"] == "required"
                else None
            ),
            info={
                "parameter_count": contract["expected_parameter_count"],
                "native_iteration_unit": native_iteration["unit"],
                "native_iteration_budget": native_iteration["budget"],
                "convergence_status": contract["convergence_status"],
                "selected_hyperparameters": selected_hyperparameters,
                "selection": selection,
                "checkpoint_hashes": contract["checkpoints"],
                "native_motion_model": motion_policy["native_model"],
            },
            history=history,
        ),
        child_started_at_utc="2026-08-04T00:00:00Z",
        child_finished_at_utc="2026-08-04T00:00:01Z",
    )
    info = child_outputs_module.load_method_info_v2(
        output_dir / "method-info.json"
    )

    child_outputs_module.validate_method_info_contract_v1(info, contract)
    forged = json.loads(canonical_json_bytes(info))
    forged["parameter_count"] += 1
    with pytest.raises(ValueError, match="parameter_count|contract"):
        child_outputs_module.validate_method_info_contract_v1(
            forged,
            contract,
        )
    if method_id == "static_cs":
        forged_count = json.loads(canonical_json_bytes(info))
        forged_count["convergence"]["observed_count"] = 999
        with pytest.raises(ValueError, match="history|contract"):
            child_outputs_module.validate_method_info_contract_v1(
                forged_count,
                contract,
            )
        forged_indices = json.loads(canonical_json_bytes(info))
        for row in forged_indices["convergence"]["history"]:
            row["iteration"] = 1
        with pytest.raises(ValueError, match="history|contract"):
            child_outputs_module.validate_method_info_contract_v1(
                forged_indices,
                contract,
            )
        for direct_forgery in (forged_count, forged_indices):
            (output_dir / "method-info.json").write_bytes(
                canonical_json_bytes(direct_forgery)
            )
            with pytest.raises(ValueError, match="history|contract|iteration"):
                child_outputs_module.validate_method_child_outputs_v2(
                    output_dir,
                    expected_method=method,
                    expected_acquisition=acquisition,
                    expected_dataset_identity_sha256=(
                        acquisition.dataset_identity_sha256
                    ),
                    expected_measurements_file_sha256="a" * 64,
                    expected_algorithm_seed=seed,
                )


def test_parameter_count_formulas_match_nondefault_cpu_architectures():
    import torch
    from gsdiff.baselines.gidc import GIDCUNet2D
    from gsdiff.baselines.recinr import ReCINRCanonicalScene
    from gsdiff.experiments.parameter_counts import (
        _gidc_parameter_count,
        _motion_parameter_count,
        _recinr_se2_scene_parameter_count,
    )
    from gsdiff.motion.se2 import SE2Motion

    channels = [3, 5, 7, 9, 11]
    gidc = GIDCUNet2D(in_channels=2, channels=channels)
    assert _gidc_parameter_count(channels) == sum(
        parameter.numel() for parameter in gidc.parameters()
        if parameter.requires_grad
    )
    motion_config = {
        "enable_rotation": False,
        "polynomial_degree": 2,
        "enable_affine": True,
    }
    motion = SE2Motion(
        (31.5, 47.5),
        enable_rotation=False,
        poly_degree=2,
        enable_affine=True,
    )
    assert _motion_parameter_count(motion_config) == sum(
        parameter.numel() for parameter in motion.parameters()
        if parameter.requires_grad
    )
    scene_config = {
        "channels": 7,
        "render_layers": 2,
        "grid_size": 20,
    }
    scene = ReCINRCanonicalScene(
        64,
        96,
        C=7,
        render_layers=2,
        grid_size=20,
    )
    formula = _recinr_se2_scene_parameter_count(scene_config, 64, 96)
    assert formula == sum(
        parameter.numel() for parameter in scene.parameters()
        if parameter.requires_grad
    )
    assert torch.random.get_rng_state().numel() > 0


@pytest.mark.parametrize(
    "mutation",
    ["wrong-kind", "short", "duplicate", "gap", "out-of-order", "index-zero"],
)
def test_method_child_writer_rejects_history_that_does_not_cover_native_budget(
    tmp_path: Path,
    mutation: str,
):
    method = resolve_method_semantics(
        "gsdiff_tv",
        method_config_id="default",
        base_config={"gaussian_count": 1000},
        measurements_metadata={},
        execution_profile="publication-v1",
    )
    acquisition = _complete_fixture_acquisition()
    contract = child_outputs_module.build_method_info_contract_v1(
        method,
        blind_acquisition_spec(acquisition),
    )
    budget = contract["native_iteration"]["budget"]
    rows = [
        {
            "kind": "outer-iteration",
            "outer_iteration": index + 1,
            **_fixture_history_metrics(method, index + 1, budget),
        }
        for index in range(budget)
    ]
    if mutation == "wrong-kind":
        rows[0] = {"kind": "step", "step": 1, "loss": 1.0}
    elif mutation == "short":
        rows = rows[:21]
    elif mutation == "duplicate":
        rows[1]["outer_iteration"] = 1
    elif mutation == "gap":
        rows[1]["outer_iteration"] = 3
    elif mutation == "out-of-order":
        rows[0], rows[1] = rows[1], rows[0]
    else:
        rows[0]["outer_iteration"] = 0
    motion_policy = contract["motion_estimate"]
    native_iteration = contract["native_iteration"]
    seed = derive_algorithm_seed(
        cell_seed=7,
        dataset_identity_sha256=acquisition.dataset_identity_sha256,
        method_id=method.method_id,
        method_config_sha256=method.method_config_sha256,
    )

    with pytest.raises(ValueError, match="history|budget|iteration"):
        child_outputs_module.write_method_child_outputs_v2(
            tmp_path / mutation,
            method=method,
            acquisition=acquisition,
            measurements_file_sha256="a" * 64,
            algorithm_seed=seed,
            result=child_outputs_module.MethodChildResult(
                method_id=method.method_id,
                reconstruction=np.ones(
                    (acquisition.T, acquisition.H, acquisition.W),
                    dtype=np.float32,
                ),
                estimated_motion_trajectory=np.zeros((4, 3), dtype=np.float32),
                dgi=np.ones(
                    (acquisition.H, acquisition.W),
                    dtype=np.float32,
                ),
                info={
                    "parameter_count": contract["expected_parameter_count"],
                    "native_iteration_unit": native_iteration["unit"],
                    "native_iteration_budget": budget,
                    "convergence_status": contract["convergence_status"],
                    "selected_hyperparameters": None,
                    "selection": None,
                    "checkpoint_hashes": contract["checkpoints"],
                    "native_motion_model": motion_policy["native_model"],
                },
                history=tuple(rows),
            ),
            child_started_at_utc="2026-08-04T00:00:00Z",
            child_finished_at_utc="2026-08-04T00:00:01Z",
        )


@pytest.mark.parametrize("consumer", ["runner", "loader", "aggregate"])
def test_complete_consumers_reject_resigned_dgi_history_claim(
    tmp_path: Path,
    consumer: str,
):
    identity = _complete_identity()
    run_dir = _write_complete_run(tmp_path, identity)
    manifest = json.loads((run_dir / "manifest.json").read_text("utf-8"))
    info_path = run_dir / "outputs/method-info.json"
    info = json.loads(info_path.read_text("utf-8"))
    info["convergence"] = {
        "status": "not-applicable",
        "sampling_policy": "all-observations",
        "observed_count": 1,
        "serialized_count": 1,
        "history": [{"kind": "pass", "pass": 1}],
    }
    _replace_declared_artifact(
        run_dir,
        manifest,
        role="method-info",
        payload=canonical_json_bytes(info),
    )
    _rewrite_run_manifest(run_dir, manifest)

    with pytest.raises(ValueError, match="history|contract|iteration"):
        _consume_complete_run(consumer, tmp_path, run_dir, identity)


@pytest.mark.parametrize(
    "mutation",
    ["observed-count", "duplicate-sampled-index", "out-of-order-sampled-index"],
)
@pytest.mark.parametrize("consumer", ["runner", "loader", "aggregate"])
def test_complete_consumers_reject_resigned_static_sampled_history_claim(
    tmp_path: Path,
    mutation: str,
    consumer: str,
):
    identity = _complete_identity(method_id="static_cs")
    run_dir = _write_complete_run(tmp_path, identity)
    manifest = json.loads((run_dir / "manifest.json").read_text("utf-8"))
    info_path = run_dir / "outputs/method-info.json"
    info = json.loads(info_path.read_text("utf-8"))
    if mutation == "observed-count":
        info["convergence"]["observed_count"] = 999
    elif mutation == "duplicate-sampled-index":
        for row in info["convergence"]["history"]:
            row["iteration"] = 1
    else:
        history = info["convergence"]["history"]
        history[0], history[1] = history[1], history[0]
    _replace_declared_artifact(
        run_dir,
        manifest,
        role="method-info",
        payload=canonical_json_bytes(info),
    )
    _rewrite_run_manifest(run_dir, manifest)

    with pytest.raises(ValueError, match="history|contract|iteration"):
        _consume_complete_run(consumer, tmp_path, run_dir, identity)


@pytest.mark.parametrize("method_id", ["static_cs", "gsdiff_tv"])
@pytest.mark.parametrize(
    "consumer",
    ["direct", "runner", "loader", "aggregate"],
)
def test_complete_consumers_reject_resigned_history_without_required_metrics(
    tmp_path: Path,
    method_id: str,
    consumer: str,
):
    identity = _complete_identity(method_id=method_id)
    run_dir = _write_complete_run(tmp_path, identity)
    manifest = json.loads((run_dir / "manifest.json").read_text("utf-8"))
    info_path = run_dir / "outputs/method-info.json"
    info = json.loads(info_path.read_text("utf-8"))
    for row in info["convergence"]["history"]:
        for field in child_outputs_module._HISTORY_METRIC_FIELDS:
            row.pop(field, None)
    _replace_declared_artifact(
        run_dir,
        manifest,
        role="method-info",
        payload=canonical_json_bytes(info),
    )
    _rewrite_run_manifest(run_dir, manifest)

    with pytest.raises(ValueError, match="history|metric|contract"):
        _consume_complete_run(consumer, tmp_path, run_dir, identity)


@pytest.mark.parametrize(
    "mutation",
    [
        "truncated-reconstruction",
        "invalid-method-info",
        "truncated-metrics",
        "invalid-metric-bounds",
    ],
)
@pytest.mark.parametrize("consumer", ["runner", "loader", "aggregate"])
def test_complete_consumers_reject_resigned_invalid_core_outputs(
    tmp_path: Path,
    mutation: str,
    consumer: str,
):
    identity = _complete_identity()
    run_dir = _write_complete_run(tmp_path, identity)
    manifest = json.loads((run_dir / "manifest.json").read_text("utf-8"))
    if mutation == "truncated-reconstruction":
        path = run_dir / "outputs/reconstruction.npz"
        payload = path.read_bytes()[:32]
        _replace_declared_artifact(
            run_dir,
            manifest,
            role="reconstruction",
            payload=payload,
        )
    elif mutation == "invalid-method-info":
        _replace_declared_artifact(
            run_dir,
            manifest,
            role="method-info",
            payload=canonical_json_bytes({"schema": "method-info-v2"}),
        )
    elif mutation == "truncated-metrics":
        payload = canonical_json_bytes({"definition_version": "metrics-v1"})
        (run_dir / "outputs/metrics.json").write_bytes(payload)
        manifest["metrics"]["sha256"] = hashlib.sha256(payload).hexdigest()
    else:
        metrics_path = run_dir / "outputs/metrics.json"
        metrics = json.loads(metrics_path.read_text("utf-8"))
        metrics["alignment"]["slope"] = -1.0
        metrics["nrmse_global_affine_l2"] = -1.0
        payload = canonical_json_bytes(metrics)
        metrics_path.write_bytes(payload)
        manifest["metrics"]["sha256"] = hashlib.sha256(payload).hexdigest()
    _rewrite_run_manifest(run_dir, manifest)

    with pytest.raises((ValueError, OSError), match="reconstruction|method|metrics|ZIP|schema"):
        _consume_complete_run(consumer, tmp_path, run_dir, identity)


@pytest.mark.parametrize(
    "mutation",
    ["execution-family", "native-budget", "parameter-count"],
)
@pytest.mark.parametrize("consumer", ["runner", "loader", "aggregate"])
def test_complete_consumers_reject_resigned_method_info_contract_mutation(
    tmp_path: Path,
    mutation: str,
    consumer: str,
):
    identity = _complete_identity()
    run_dir = _write_complete_run(tmp_path, identity)
    manifest = json.loads((run_dir / "manifest.json").read_text("utf-8"))
    info_path = run_dir / "outputs/method-info.json"
    info = json.loads(info_path.read_text("utf-8"))
    if mutation == "execution-family":
        info["execution_family"] = "gsdiff"
    elif mutation == "native-budget":
        info["native_iteration"]["budget"] += 1
    else:
        info["parameter_count"] += 1
    _replace_declared_artifact(
        run_dir,
        manifest,
        role="method-info",
        payload=canonical_json_bytes(info),
    )
    _rewrite_run_manifest(run_dir, manifest)

    with pytest.raises(ValueError, match="method|contract|identity"):
        _consume_complete_run(consumer, tmp_path, run_dir, identity)


@pytest.mark.parametrize(
    "relative",
    [
        "manifest.json",
        "outputs/metrics.json",
        "outputs/reconstruction.npz",
        "evidence/source-snapshot.json",
    ],
)
@pytest.mark.parametrize("consumer", ["runner", "loader", "aggregate"])
def test_complete_consumers_reject_external_hardlink_alias(
    tmp_path: Path,
    relative: str,
    consumer: str,
):
    identity = _complete_identity()
    run_dir = _write_complete_run(tmp_path, identity)
    source = run_dir / relative
    alias = tmp_path / f"external-{relative.replace('/', '-') }"
    os.link(source, alias)

    with pytest.raises(ValueError, match="link|alias|regular"):
        _consume_complete_run(consumer, tmp_path, run_dir, identity)


@pytest.mark.parametrize(
    "role",
    [
        "reconstruction",
        "method-info",
        "stdout",
        "stderr",
        "resolved-config",
        "lifecycle",
        "audit",
        "audit-validation",
        "materialization-logical",
        "resource-sampling",
        "source-snapshot",
    ],
)
@pytest.mark.parametrize("consumer", ["runner", "loader", "aggregate"])
def test_reusable_run_rejects_self_consistent_required_artifact_omission(
    tmp_path: Path,
    role: str,
    consumer: str,
):
    identity = _complete_identity()
    run_dir = _write_complete_run(tmp_path, identity)
    manifest = json.loads((run_dir / "manifest.json").read_text("utf-8"))
    descriptor = next(
        item for item in manifest["artifacts"] if item["role"] == role
    )
    (run_dir / descriptor["path"]).unlink()
    manifest["artifacts"].remove(descriptor)
    _rewrite_run_manifest(run_dir, manifest)

    with pytest.raises(ValueError, match="artifact|contract|inventory"):
        _consume_complete_run(consumer, tmp_path, run_dir, identity)


@pytest.mark.parametrize(
    "mutation",
    ["role", "path", "schema", "required", "order", "metrics-path"],
)
@pytest.mark.parametrize("consumer", ["runner", "loader", "aggregate"])
def test_reusable_run_rejects_self_consistent_descriptor_contract_mutation(
    tmp_path: Path,
    mutation: str,
    consumer: str,
):
    identity = _complete_identity()
    run_dir = _write_complete_run(tmp_path, identity)
    manifest = json.loads((run_dir / "manifest.json").read_text("utf-8"))
    artifacts = manifest["artifacts"]
    if mutation == "role":
        artifacts[0]["role"] = "forged-reconstruction"
    elif mutation == "path":
        source = run_dir / artifacts[0]["path"]
        destination = run_dir / "outputs/forged-reconstruction.npz"
        source.rename(destination)
        artifacts[0]["path"] = "outputs/forged-reconstruction.npz"
    elif mutation == "schema":
        artifacts[0]["schema_version"] = "forged-v1"
    elif mutation == "required":
        artifacts[0]["required"] = False
    elif mutation == "order":
        manifest["artifacts"] = list(reversed(artifacts))
    else:
        source = run_dir / manifest["metrics"]["path"]
        destination = run_dir / "outputs/forged-metrics.json"
        source.rename(destination)
        manifest["metrics"]["path"] = "outputs/forged-metrics.json"
    _rewrite_run_manifest(run_dir, manifest)

    with pytest.raises(ValueError, match="artifact|contract|metrics"):
        _consume_complete_run(consumer, tmp_path, run_dir, identity)


@pytest.mark.parametrize(
    "mutation",
    [
        "resolved-config",
        "lifecycle",
        "audit-validation",
        "materialization",
        "materialization-device",
        "materialization-method-config",
        "materialization-projection",
        "materialization-delete-field",
        "materialization-extra-field",
        "materialization-plausible-entrypoint",
        "resource",
        "resource-device",
        "resource-raised-cap",
        "resource-runtime-over-cap",
        "resource-peak-over-cap",
        "source-snapshot",
        "metrics-nonfinite",
        "metrics-version",
    ],
)
@pytest.mark.parametrize("consumer", ["runner", "loader", "aggregate"])
def test_reusable_run_rejects_self_consistent_semantic_evidence_mutation(
    tmp_path: Path,
    mutation: str,
    consumer: str,
):
    identity = _complete_identity()
    run_dir = _write_complete_run(tmp_path, identity)
    manifest = json.loads((run_dir / "manifest.json").read_text("utf-8"))
    if mutation == "resolved-config":
        _replace_declared_artifact(
            run_dir,
            manifest,
            role="resolved-config",
            payload=canonical_json_bytes({"forged": True}),
        )
    elif mutation == "lifecycle":
        _replace_declared_artifact(
            run_dir,
            manifest,
            role="lifecycle",
            payload=canonical_json_bytes(
                {
                    "schema": "run-lifecycle-v1",
                    "state": "forged",
                    "identity_sha256": identity.identity_sha256,
                    "owner_token": "6" * 32,
                    "fence": 1,
                }
            ),
        )
    elif mutation == "audit-validation":
        _replace_declared_artifact(
            run_dir,
            manifest,
            role="audit-validation",
            payload=canonical_json_bytes(
                {
                    "schema": "validated-method-audit-log-v1",
                    "policy_sha256": "5" * 64,
                    "audit_log_sha256": "6" * 64,
                    "event_count": 1,
                    "terminal_status": "success",
                }
            ),
        )
    elif mutation in {
        "materialization",
        "materialization-device",
        "materialization-method-config",
        "materialization-projection",
        "materialization-delete-field",
        "materialization-extra-field",
        "materialization-plausible-entrypoint",
    }:
        logical_path = run_dir / "evidence/materialization-logical.json"
        logical = json.loads(logical_path.read_text("utf-8"))
        if mutation == "materialization":
            logical["dataset_identity_sha256"] = "6" * 64
        elif mutation == "materialization-device":
            logical["requested_runtime_device"] = "cuda:0"
        elif mutation == "materialization-method-config":
            logical["method_config_sha256"] = "6" * 64
        elif mutation == "materialization-delete-field":
            del logical["entrypoint"]
        elif mutation == "materialization-extra-field":
            logical["plausible_extra"] = "ignored"
        elif mutation == "materialization-plausible-entrypoint":
            logical["entrypoint"] = "scripts/experiments/method_child.py"
        else:
            logical["source_snapshot_sha256"] = "6" * 64
        _replace_declared_artifact(
            run_dir,
            manifest,
            role="materialization-logical",
            payload=canonical_json_bytes(logical),
        )
    elif mutation.startswith("resource"):
        resource_path = run_dir / "evidence/resource-sampling.json"
        resource = json.loads(resource_path.read_text("utf-8"))
        if mutation == "resource":
            resource["peak_vram_bytes"] = 1
        elif mutation == "resource-device":
            resource["requested_runtime_device"] = "cuda:0"
        elif mutation == "resource-raised-cap":
            resource["compute_cap"] = {
                **resource["compute_cap"],
                "wall_time_seconds": 11,
                "peak_vram_bytes": 2,
            }
        elif mutation == "resource-runtime-over-cap":
            over_cap = float(resource["compute_cap"]["wall_time_seconds"] + 1)
            resource["runtime_seconds"] = over_cap
            manifest["execution"]["runtime_seconds"] = over_cap
        else:
            resource["peak_vram_bytes"] = 2
            manifest["execution"]["peak_vram_bytes"] = 2
        _replace_declared_artifact(
            run_dir,
            manifest,
            role="resource-sampling",
            payload=canonical_json_bytes(resource),
        )
    elif mutation == "source-snapshot":
        source_path = run_dir / "evidence/source-snapshot.json"
        source = json.loads(source_path.read_text("utf-8"))
        source["commit"] = "6" * 40
        source["snapshot_sha256"] = hashlib.sha256(
            canonical_json_bytes(
                {
                    "schema": "source-snapshot-identity-v1",
                    "commit": source["commit"],
                    "inventory": source["inventory"],
                }
            )
        ).hexdigest()
        _replace_declared_artifact(
            run_dir,
            manifest,
            role="source-snapshot",
            payload=canonical_json_bytes(source),
        )
    else:
        metrics_path = run_dir / manifest["metrics"]["path"]
        metrics_payload = (
            b'{"definition_version":"metrics-v1","score":NaN}'
            if mutation == "metrics-nonfinite"
            else canonical_json_bytes(
                {"definition_version": "forged-v1", "score": 1.0}
            )
        )
        metrics_path.write_bytes(metrics_payload)
        manifest["metrics"]["sha256"] = hashlib.sha256(
            metrics_payload
        ).hexdigest()
    _rewrite_run_manifest(run_dir, manifest)

    with pytest.raises(
        ValueError,
        match=(
            "evidence|config|lifecycle|audit|resource|source|metrics|canonical|"
            "non-finite"
        ),
    ):
        _consume_complete_run(consumer, tmp_path, run_dir, identity)


@pytest.mark.parametrize(
    "overrides",
    [
        {"scientific_contract_sha256": "0" * 64},
        {"method_id": "gsdiff_diffusion"},
        {"target_id": "digit5"},
        {"motion_id": "rot"},
        {"seed": 11},
        {"config_sha256": "0" * 64},
        {"dataset_identity_sha256": "0" * 64},
        {"assets_sha256": {"target": "0" * 64}},
        {"checkpoints_sha256": {"prior": "0" * 64}},
        {"code_commit": "0" * 40},
        {
            "dirty_worktree": True,
            "source_tree_hash": "0" * 64,
        },
        {"dependencies_sha256": "0" * 64},
        {"environment_lock_sha256": "0" * 64},
        {"metric_version": "metrics-v2"},
    ],
)
def test_reusable_run_never_reuses_a_different_identity(
    tmp_path: Path, overrides: dict[str, object]
):
    _write_complete_run(tmp_path, _complete_identity())

    assert reusable_run(tmp_path, _complete_identity(**overrides)) is None


@pytest.mark.parametrize("malformation", ["legacy", "partial", "nonzero", "running"])
def test_reusable_run_fails_closed_on_existing_invalid_identity_directory(
    tmp_path: Path, malformation: str
):
    identity = _complete_identity()
    run_dir = tmp_path / "runs" / identity.identity_sha256
    run_dir.mkdir(parents=True)
    if malformation == "legacy":
        (run_dir / "results.json").write_text("{}", encoding="utf-8")
    else:
        valid_dir = _write_complete_run(tmp_path / "source", identity)
        for path in valid_dir.rglob("*"):
            relative = path.relative_to(valid_dir)
            destination = run_dir / relative
            if path.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(path.read_bytes())
        manifest_path = run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text("utf-8"))
        if malformation == "partial":
            (run_dir / "outputs" / "reconstruction.npz").unlink()
        elif malformation == "nonzero":
            manifest["execution"]["return_code"] = 9
            manifest_path.write_bytes(canonical_json_bytes(manifest))
        else:
            manifest["status"] = "running"
            manifest["execution"]["return_code"] = None
            manifest["execution"]["ended_at_utc"] = None
            manifest_path.write_bytes(canonical_json_bytes(manifest))

    with pytest.raises((OSError, ValueError)):
        reusable_run(tmp_path, identity)
