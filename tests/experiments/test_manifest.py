from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest
import numpy as np

import gsdiff.experiments.manifest as manifest_module

from gsdiff.data._artifact_identity import array_descriptor
from gsdiff.data._artifact_models import SPIAcquisitionData
from gsdiff.data.artifacts import (
    blind_acquisition_spec,
    build_dataset_manifest,
    build_dataset_payloads,
    dataset_manifest_bytes,
    publish_dataset,
)
from gsdiff.evaluation.metrics import evaluate_video_global_affine
from gsdiff.experiments.dataset_binding import dataset_measurement_record
from gsdiff.experiments import child_outputs as child_outputs_module
from gsdiff.experiments.execution import _materialization_identity_documents
from gsdiff.experiments.identity import (
    build_run_identity,
    canonical_json_bytes,
    requirements_dependencies_sha256,
)
from gsdiff.experiments.methods import derive_algorithm_seed, resolve_method_semantics
from gsdiff.experiments.runner import _identity_bound_config
from gsdiff.experiments.manifest import (
    build_aggregate_index,
    build_manifest,
    load_complete_manifest,
    validate_aggregate_index,
    validate_manifest,
)


_HASH = "a" * 64
_COMMIT = "b" * 40
_ROOT = Path(__file__).resolve().parents[2]
_ENVIRONMENT_LOCK = json.loads(
    (_ROOT / "docs" / "reproducibility" / "environment-lock.json").read_text("utf-8")
)
_ENVIRONMENT_HASHES = _ENVIRONMENT_LOCK["fingerprint_sha256"]
_PHYSICAL_GENERATED_DATASET = None
_PHYSICAL_RUNTIME = {
    "python": _ENVIRONMENT_LOCK["fingerprint"]["python"]["version"],
    "pytorch": _ENVIRONMENT_LOCK["fingerprint"]["pytorch"]["version"],
    "cuda": _ENVIRONMENT_LOCK["fingerprint"]["pytorch"]["cuda_build"] or "",
    "gpu": "",
    "os": _ENVIRONMENT_LOCK["fingerprint"]["platform"]["platform"],
}
_PYTHON_EXECUTABLE_SHA256 = hashlib.sha256(
    Path(sys.executable).read_bytes()
).hexdigest()


def _make_directory_link_or_skip(link: Path, target: Path) -> None:
    try:
        os.symlink(target, link, target_is_directory=True)
        return
    except OSError:
        if os.name != "nt":
            pytest.skip("filesystem does not permit directory symlinks")
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip("filesystem does not permit directory junctions")


def _identity(*, dirty: bool = False):
    return build_run_identity(
        execution_class="blind_method_child",
        scientific_contract_id="gsdiff-sim-v1",
        scientific_contract_sha256=_HASH,
        method_id="gsdiff_tv",
        target_id="tank",
        motion_id="trans",
        seed=7,
        config_sha256=hashlib.sha256(b"{}").hexdigest(),
        dataset_identity_sha256="c" * 64,
        assets_sha256={"tank": "d" * 64},
        checkpoints_sha256={},
        code_commit=_COMMIT,
        dirty_worktree=dirty,
        source_tree_hash="9" * 64 if dirty else None,
        dependencies_sha256=requirements_dependencies_sha256(_ROOT / "requirements-lock.txt"),
        environment_lock_sha256=_ENVIRONMENT_HASHES,
        metric_version="metrics-v1",
    )


def _complete_manifest(identity=None):
    identity = identity or _identity()
    return build_manifest(
        status="complete",
        identity=identity,
        config_resolved={},
        inputs={
            "measurements_file_sha256": "1" * 64,
            "evaluation_truth_file_sha256": "2" * 64,
            "dataset_manifest_sha256": "3" * 64,
        },
        runtime={"python": "3.12", "pytorch": "2.8", "cuda": "12.8", "gpu": "gpu", "os": "Windows"},
        execution={
            "command": ["python", "run.py"],
            "started_at_utc": "2026-07-27T00:00:00Z",
            "ended_at_utc": "2026-07-27T00:00:01Z",
            "return_code": 0,
            "runtime_seconds": 1.0,
            "peak_vram_bytes": 0,
        },
        measurement={
            "train_count": 2560,
            "holdout_count": 250,
            "pattern_family": "bernoulli",
            "requested_snr_db": 25,
            "noise_calibration_id": "detector-absolute-v1",
            "noise_calibration_sha256": "4" * 64,
            "noise_sigma_absolute": 0.0123,
            "realized_train_snr_db": 24.97,
            "realized_holdout_snr_db": 25.04,
        },
        metrics={"version": "metrics-v1", "path": "metrics.json", "sha256": "5" * 64},
        artifacts=[],
    )


def test_complete_manifest_validates_against_constructed_identity():
    manifest = _complete_manifest()

    validate_manifest(manifest)
    assert manifest["identity_sha256"] == _identity().identity_sha256


def test_manifest_rejects_config_content_that_disagrees_with_identity():
    manifest = _complete_manifest()
    manifest["config"]["resolved"] = {"solver": {"steps": 2}}

    with pytest.raises(ValueError, match="config"):
        validate_manifest(manifest)


def _mutate_contract(manifest):
    manifest["protocol"]["scientific_contract_id"] = "gsdiff-ablation-v1"
    manifest["protocol"]["scientific_contract_sha256"] = "6" * 64


def _mutate_dirty_source_pair(manifest):
    manifest["code"]["dirty_worktree"] = True
    manifest["code"]["source_tree_sha256"] = "6" * 64


def _mutate_resolved_config(manifest):
    resolved = {"solver": {"steps": 11}}
    manifest["config"]["resolved"] = resolved
    manifest["config"]["sha256"] = hashlib.sha256(
        canonical_json_bytes(resolved)
    ).hexdigest()


def _mutate_metric_version(manifest):
    manifest["metric_version"] = "metrics-v2"
    manifest["metrics"]["version"] = "metrics-v2"


@pytest.mark.parametrize(
    "mutate",
    [
        _mutate_contract,
        lambda value: value["protocol"].__setitem__("method", "gsdiff_diffusion"),
        lambda value: value["protocol"].__setitem__("target", "digit5"),
        lambda value: value["protocol"].__setitem__("motion", "rot"),
        lambda value: value["protocol"].__setitem__("seed", 11),
        lambda value: value["code"].__setitem__("git_commit", "c" * 40),
        _mutate_dirty_source_pair,
        _mutate_resolved_config,
        lambda value: value["inputs"].__setitem__(
            "dataset_identity_sha256", "6" * 64
        ),
        lambda value: value["inputs"].__setitem__(
            "assets", {"tank": "6" * 64}
        ),
        lambda value: value["inputs"].__setitem__(
            "checkpoints", {"diffusion": "7" * 64}
        ),
        lambda value: value["runtime"].__setitem__(
            "dependencies_sha256", "8" * 64
        ),
        lambda value: value["runtime"].__setitem__(
            "environment_lock_sha256", "9" * 64
        ),
        _mutate_metric_version,
        lambda value: value.__setitem__(
            "execution_class", "compatibility_unblinded"
        ),
    ],
)
def test_manifest_rejects_valid_format_identity_field_mismatches(mutate):
    manifest = _complete_manifest()
    mutate(manifest)

    with pytest.raises(ValueError, match="identity|invalid manifest|execution_class"):
        validate_manifest(manifest)


def test_dirty_diagnostic_complete_validates_but_is_not_reusable(tmp_path: Path):
    identity = _identity(dirty=True)
    manifest = _complete_manifest(identity)
    validate_manifest(manifest)
    path = tmp_path / "runs" / identity.identity_sha256 / "manifest.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(canonical_json_bytes(manifest))

    with pytest.raises(ValueError, match="dirty"):
        load_complete_manifest(
            path,
            artifact_root=tmp_path,
            expected_identity_sha256=identity.identity_sha256,
        )


@pytest.mark.parametrize("status", ["running", "failed"])
def test_noncomplete_manifest_is_not_reusable(tmp_path: Path, status: str):
    manifest = _complete_manifest()
    manifest["status"] = status
    if status == "running":
        manifest["execution"]["ended_at_utc"] = None
        manifest["execution"]["return_code"] = None
    else:
        manifest["execution"]["return_code"] = 1
        manifest["failure"] = {"stderr_tail": "failure", "diagnostic_paths": []}
    identity = manifest["identity_sha256"]
    path = tmp_path / "runs" / identity / "manifest.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(canonical_json_bytes(manifest))

    assert load_complete_manifest(
        path,
        artifact_root=tmp_path,
        expected_identity_sha256=identity,
    ) is None


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.pop("config"), "config"),
        (lambda value: value.__setitem__("unexpected", True), "unexpected"),
        (lambda value: value["measurement"].__setitem__("noise_sigma_absolute", math.nan), "non-finite"),
        (lambda value: value["measurement"].__setitem__("train_count", True), "True"),
        (lambda value: value["protocol"].__setitem__("scientific_contract_sha256", "short"), "does not match"),
        (lambda value: value["execution"].__setitem__("runtime_seconds", -1.0), "minimum"),
    ],
)
def test_manifest_rejects_malformed_schema_and_native_values(mutate, message: str):
    manifest = _complete_manifest()
    mutate(manifest)

    with pytest.raises((TypeError, ValueError), match=message):
        validate_manifest(manifest)


def test_manifest_rejects_direct_container_and_string_subclasses():
    class DictSubclass(dict):
        pass

    class ListSubclass(list):
        pass

    class StringSubclass(str):
        pass

    with pytest.raises(TypeError, match="unsupported JSON type"):
        validate_manifest(DictSubclass(_complete_manifest()))

    manifest = _complete_manifest()
    manifest["artifacts"] = ListSubclass()
    with pytest.raises(TypeError, match="unsupported JSON type"):
        validate_manifest(manifest)

    manifest = _complete_manifest()
    manifest["protocol"]["method"] = StringSubclass("gsdiff_tv")
    with pytest.raises(TypeError, match="unsupported JSON type"):
        validate_manifest(manifest)


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_loader_rejects_parsed_nonfinite_json_numbers(
    tmp_path: Path,
    literal: str,
):
    path, _ = _write_physical_complete(tmp_path)
    raw = path.read_bytes()
    mutated = raw.replace(b'"runtime_seconds":1.0', f'"runtime_seconds":{literal}'.encode())
    assert mutated != raw
    path.write_bytes(mutated)

    with pytest.raises(ValueError, match="non-finite"):
        load_complete_manifest(
            path,
            artifact_root=tmp_path,
            expected_identity_sha256=path.parent.name,
        )


def _artifact(role: str, path: str) -> dict[str, object]:
    return {
        "role": role,
        "path": path,
        "sha256": "6" * 64,
        "size_bytes": 1,
        "schema_version": "artifact-v1",
        "required": True,
    }


def test_manifest_rejects_duplicate_and_casefold_colliding_output_paths():
    manifest = _complete_manifest()
    manifest["artifacts"] = [
        _artifact("first", "outputs/result.bin"),
        _artifact("second", "OUTPUTS/RESULT.BIN"),
    ]

    with pytest.raises(ValueError, match="collide"):
        validate_manifest(manifest)


def test_manifest_rejects_duplicate_artifact_roles():
    manifest = _complete_manifest()
    manifest["artifacts"] = [
        _artifact("reconstruction", "first.bin"),
        _artifact("reconstruction", "second.bin"),
    ]

    with pytest.raises(ValueError, match="role"):
        validate_manifest(manifest)


@pytest.mark.parametrize(
    "unsafe_path",
    ["../escape", "/absolute", "C:/drive", "folder\\file", "folder/CON", "folder/CON.txt", "folder/CONIN$", "folder/COM¹", "folder/LPT³", "folder/name*", "folder/name\x01", "folder//repeat", "folder/name.", "folder/name:stream"],
)
def test_manifest_rejects_unsafe_output_paths(unsafe_path: str):
    manifest = _complete_manifest()
    manifest["metrics"]["path"] = unsafe_path

    with pytest.raises(ValueError, match="path"):
        validate_manifest(manifest)


def _physical_method():
    return resolve_method_semantics(
        "dgi",
        method_config_id="default",
        base_config={},
        measurements_metadata={},
        execution_profile="publication-v1",
    )


def _physical_acquisition() -> SPIAcquisitionData:
    return _physical_generated_dataset().acquisition


def _physical_generated_dataset():
    global _PHYSICAL_GENERATED_DATASET
    if _PHYSICAL_GENERATED_DATASET is None:
        path = _ROOT / "scripts/experiments/build_datasets.py"
        spec = importlib.util.spec_from_file_location(
            "manifest_physical_dataset_builder",
            path,
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        plan = module.plan_campaign_datasets(
            repo_root=_ROOT,
            protocol_path=_ROOT / "configs/protocols/pilot-v1.yaml",
            runtime={
                "dependencies_sha256": "1" * 64,
                "environment_lock_sha256": "2" * 64,
            },
            generator_commit="3" * 40,
        )
        _PHYSICAL_GENERATED_DATASET = module.generate_corrected_dataset(
            **plan.requests[0].generation_arguments()
        )
    return _PHYSICAL_GENERATED_DATASET


def _physical_payload_hashes() -> dict[str, str]:
    return {
        name: hashlib.sha256(payload).hexdigest()
        for name, payload in build_dataset_payloads(
            _physical_generated_dataset()
        ).items()
    }


def _physical_dataset_input_contract() -> dict[str, object]:
    generated = _physical_generated_dataset()
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


def _physical_source_evidence() -> tuple[list[dict[str, object]], list[dict[str, object]], str, str]:
    source_inventory = [
        {
            "path": "gsdiff/module.py",
            "mode": "100644",
            "git_blob": "1" * 40,
            "sha256": "2" * 64,
            "size_bytes": 1,
        }
    ]
    selected_source = [
        {"path": "gsdiff/module.py", "sha256": "2" * 64}
    ]
    snapshot_sha256 = hashlib.sha256(
        canonical_json_bytes(
            {
                "schema": "source-snapshot-identity-v1",
                "commit": _COMMIT,
                "inventory": source_inventory,
            }
        )
    ).hexdigest()
    projection_sha256 = hashlib.sha256(
        canonical_json_bytes(selected_source)
    ).hexdigest()
    return source_inventory, selected_source, snapshot_sha256, projection_sha256


def _physical_materialization_logical() -> dict[str, object]:
    method = _physical_method()
    acquisition = _physical_acquisition()
    algorithm_seed = derive_algorithm_seed(
        cell_seed=7,
        dataset_identity_sha256=acquisition.dataset_identity_sha256,
        method_id=method.method_id,
        method_config_sha256=method.method_config_sha256,
    )
    _source_inventory, selected_source, _snapshot, projection = (
        _physical_source_evidence()
    )
    _config, logical = _materialization_identity_documents(
        method=method,
        dataset_identity_sha256=acquisition.dataset_identity_sha256,
        measurements_file_sha256=_physical_payload_hashes()[
            "measurements.npz"
        ],
        expected_acquisition_spec=blind_acquisition_spec(acquisition),
        algorithm_seed=algorithm_seed,
        source_inventory=selected_source,
        requested_runtime_device="cpu",
    )
    assert logical["source_snapshot_sha256"] == projection
    return logical


def _physical_runner_config() -> dict[str, object]:
    method = _physical_method()
    _inventory, _selected, snapshot, projection = _physical_source_evidence()
    logical = _physical_materialization_logical()
    return _identity_bound_config(
        {"method_config_sha256": method.method_config_sha256},
        requested_runtime_device="cpu",
        source_snapshot_sha256=snapshot,
        source_projection_sha256=projection,
        compute_cap=method.semantic_config["compute_cap"],
        materialization_logical_sha256=hashlib.sha256(
            canonical_json_bytes(logical)
        ).hexdigest(),
        method_info_contract=(
            child_outputs_module.build_method_info_contract_v1(
                method,
                blind_acquisition_spec(_physical_acquisition()),
            )
        ),
        dataset_input_contract=_physical_dataset_input_contract(),
        runtime_contract=_PHYSICAL_RUNTIME,
        python_executable_sha256=_PYTHON_EXECUTABLE_SHA256,
    )


def _physical_complete_identity():
    payload = dict(_identity().payload())
    dataset_spec = _physical_generated_dataset().dataset_identity_spec
    payload["method_id"] = "dgi"
    payload["dataset_identity_sha256"] = (
        _physical_generated_dataset().dataset_identity_sha256
    )
    target = _physical_generated_dataset().resolved_generator_config["target"]
    assets = target["assets_sha256"]
    descriptor = target["descriptor"]
    payload["assets_sha256"] = (
        {target["id"]: assets[descriptor]}
        if descriptor in assets and len(assets) == 1
        else dict(assets)
    )
    payload["scientific_contract_id"] = dataset_spec[
        "scientific_contract"
    ]["id"]
    payload["scientific_contract_sha256"] = dataset_spec[
        "scientific_contract"
    ]["sha256"]
    payload["target_id"] = dataset_spec["target"]["id"]
    payload["motion_id"] = dataset_spec["motion"]["id"]
    payload["seed"] = dataset_spec["seed"]
    payload["config_sha256"] = hashlib.sha256(
        canonical_json_bytes(_physical_runner_config())
    ).hexdigest()
    return build_run_identity(**payload)


def _write_physical_complete(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    identity = _physical_complete_identity()
    publication = publish_dataset(tmp_path, _physical_generated_dataset())
    verified_dataset = publication.verified
    run_directory = tmp_path / "runs" / identity.identity_sha256
    (run_directory / "outputs").mkdir(parents=True)
    (run_directory / "evidence").mkdir()
    method = _physical_method()
    acquisition = verified_dataset.acquisition
    algorithm_seed = derive_algorithm_seed(
        cell_seed=7,
        dataset_identity_sha256=acquisition.dataset_identity_sha256,
        method_id=method.method_id,
        method_config_sha256=method.method_config_sha256,
    )
    reconstruction = np.ones(
        (acquisition.T, acquisition.H, acquisition.W),
        dtype=np.float32,
    )
    child_outputs_module.write_method_child_outputs_v2(
        run_directory / "outputs",
        method=method,
        acquisition=acquisition,
        measurements_file_sha256=verified_dataset.payload_evidence[
            "measurements.npz"
        ].sha256,
        algorithm_seed=algorithm_seed,
        result=child_outputs_module.MethodChildResult(
            method_id="dgi",
            reconstruction=reconstruction,
            estimated_motion_trajectory=None,
            dgi=np.ones((acquisition.H, acquisition.W), dtype=np.float32),
            info={
                "parameter_count": 0,
                "native_iteration_unit": "pass",
                "native_iteration_budget": 1,
                "convergence_status": "not-applicable",
                "selected_hyperparameters": None,
                "selection": None,
                "checkpoint_hashes": [],
            },
            history=(),
        ),
        child_started_at_utc="2026-07-27T00:00:00Z",
        child_finished_at_utc="2026-07-27T00:00:01Z",
    )
    policy_sha256 = "5" * 64
    audit_log = b"\n".join(
        (
            canonical_json_bytes(
                {
                    "sequence": 0,
                    "timestamp_utc": "2026-07-27T00:00:00.000000Z",
                    "operation": "hook-installed",
                    "decision": "allow",
                    "policy_sha256": policy_sha256,
                }
            ),
            canonical_json_bytes(
                {
                    "sequence": 1,
                    "timestamp_utc": "2026-07-27T00:00:01.000000Z",
                    "operation": "bootstrap-finished",
                    "decision": "allow",
                    "status": "success",
                }
            ),
            b"",
        )
    )
    source_inventory, selected_source, source_snapshot_sha256, _projection = (
        _physical_source_evidence()
    )
    source_identity = {
        "schema": "source-snapshot-identity-v1",
        "commit": _COMMIT,
        "inventory": source_inventory,
    }
    payloads = {
        "outputs/reconstruction.npz": (
            run_directory / "outputs/reconstruction.npz"
        ).read_bytes(),
        "outputs/method-info.json": (
            run_directory / "outputs/method-info.json"
        ).read_bytes(),
        "outputs/stdout.log": b"stdout\n",
        "outputs/stderr.log": b"",
        "resolved-config.json": canonical_json_bytes(
            _physical_runner_config()
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
                "policy_sha256": policy_sha256,
                "audit_log_sha256": hashlib.sha256(audit_log).hexdigest(),
                "event_count": 2,
                "terminal_status": "success",
            }
        ),
        "evidence/materialization-logical.json": canonical_json_bytes(
            _physical_materialization_logical()
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
                "commit": _COMMIT,
                "snapshot_sha256": source_snapshot_sha256,
                "inventory": source_inventory,
            }
        ),
        "outputs/metrics.json": canonical_json_bytes(
            evaluate_video_global_affine(
                verified_dataset.truth.gt_frames,
                reconstruction,
            )
        ),
    }
    for relative, payload in payloads.items():
        (run_directory / relative).write_bytes(payload)
    artifacts = []
    for role, relative, schema in manifest_module._RUNNER_ARTIFACT_CONTRACT:
        payload = payloads[relative]
        artifacts.append(
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
        config_resolved=_physical_runner_config(),
        inputs={
            "measurements_file_sha256": verified_dataset.payload_evidence[
                "measurements.npz"
            ].sha256,
            "evaluation_truth_file_sha256": verified_dataset.payload_evidence[
                "evaluation-truth.npz"
            ].sha256,
            "dataset_manifest_sha256": verified_dataset.dataset_manifest_sha256,
        },
        runtime=dict(_PHYSICAL_RUNTIME),
        execution={
            "command": list(method.command_template),
            "started_at_utc": "2026-07-27T00:00:00Z",
            "ended_at_utc": "2026-07-27T00:00:01Z",
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
        artifacts=artifacts,
    )
    path = run_directory / "manifest.json"
    path.write_bytes(canonical_json_bytes(manifest))
    return path, manifest


def test_load_complete_manifest_verifies_every_declared_regular_file(tmp_path: Path):
    path, manifest = _write_physical_complete(tmp_path)

    loaded = load_complete_manifest(
        path,
        artifact_root=tmp_path,
        expected_identity_sha256=manifest["identity_sha256"],
    )

    assert loaded == manifest


@pytest.mark.parametrize("fault", ["missing", "unlisted", "mutated", "wrong-size"])
def test_load_complete_manifest_rejects_incomplete_or_changed_outputs(tmp_path: Path, fault: str):
    path, manifest = _write_physical_complete(tmp_path)
    artifact = path.parent / "outputs/reconstruction.npz"
    if fault == "missing":
        artifact.unlink()
    elif fault == "unlisted":
        (path.parent / "undeclared.txt").write_text("no", encoding="utf-8")
    elif fault == "mutated":
        artifact.write_bytes(b"changed")
    else:
        manifest["artifacts"][0]["size_bytes"] += 1
        path.write_bytes(canonical_json_bytes(manifest))

    with pytest.raises(ValueError, match="missing|unlisted|mismatch"):
        load_complete_manifest(
            path,
            artifact_root=tmp_path,
            expected_identity_sha256=manifest["identity_sha256"],
        )


def test_load_complete_manifest_rejects_an_undeclared_empty_directory(tmp_path: Path):
    path, _ = _write_physical_complete(tmp_path)
    (path.parent / "empty").mkdir()

    with pytest.raises(ValueError, match="unlisted output directory"):
        load_complete_manifest(
            path,
            artifact_root=tmp_path,
            expected_identity_sha256=path.parent.name,
        )


def test_load_complete_manifest_rejects_a_symlinked_output_leaf(tmp_path: Path):
    path, manifest = _write_physical_complete(tmp_path)
    artifact = path.parent / "outputs/reconstruction.npz"
    target = path.parent / "target.npz"
    target.write_bytes(artifact.read_bytes())
    artifact.unlink()
    try:
        artifact.symlink_to(target)
    except OSError:
        pytest.skip("filesystem does not permit symlink creation")

    with pytest.raises(ValueError, match="symlink|reparse"):
        load_complete_manifest(
            path,
            artifact_root=tmp_path,
            expected_identity_sha256=manifest["identity_sha256"],
        )


def test_load_complete_manifest_rejects_a_symlinked_manifest_leaf(tmp_path: Path):
    path, _ = _write_physical_complete(tmp_path)
    target = path.parent / "manifest-target.json"
    target.write_bytes(path.read_bytes())
    path.unlink()
    try:
        path.symlink_to(target)
    except OSError:
        pytest.skip("filesystem does not permit symlink creation")

    with pytest.raises(ValueError, match="symlink|reparse"):
        load_complete_manifest(
            path,
            artifact_root=tmp_path,
            expected_identity_sha256=path.parent.name,
        )


def test_load_complete_manifest_rejects_a_linked_run_ancestor(tmp_path: Path):
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    path, manifest = _write_physical_complete(real_parent)
    alias = tmp_path / "linked-parent"
    _make_directory_link_or_skip(alias, real_parent)
    linked_manifest = alias / "runs" / manifest["identity_sha256"] / path.name

    with pytest.raises(ValueError, match="symlink|reparse"):
        load_complete_manifest(
            linked_manifest,
            artifact_root=alias,
            expected_identity_sha256=manifest["identity_sha256"],
        )


def test_load_complete_manifest_rejects_a_nested_link_escape(tmp_path: Path):
    path, manifest = _write_physical_complete(tmp_path)
    original = path.parent / "outputs/reconstruction.npz"
    original.unlink()
    outside = tmp_path / "outside"
    outside.mkdir()
    escaped = outside / "reconstruction.npz"
    escaped.write_bytes(b"artifact bytes")
    nested = path.parent / "nested"
    _make_directory_link_or_skip(nested, outside)
    manifest["artifacts"][0]["path"] = "nested/reconstruction.npz"
    path.write_bytes(canonical_json_bytes(manifest))

    with pytest.raises(ValueError, match="symlink|reparse"):
        load_complete_manifest(
            path,
            artifact_root=tmp_path,
            expected_identity_sha256=manifest["identity_sha256"],
        )


def test_single_handle_reader_detects_same_size_in_place_rewrite(tmp_path: Path, monkeypatch):
    path = tmp_path / "output.bin"
    path.write_bytes(b"original")
    original_read = manifest_module.os.read
    rewritten = False

    def rewrite_after_read(descriptor: int, size: int) -> bytes:
        nonlocal rewritten
        block = original_read(descriptor, size)
        if block and not rewritten:
            rewritten = True
            path.write_bytes(b"changed!")
        return block

    monkeypatch.setattr(manifest_module.os, "read", rewrite_after_read)

    with pytest.raises(ValueError, match="changed while being verified"):
        manifest_module._read_regular_file(path, "output")


def test_output_verification_rejects_entries_added_during_hashing(tmp_path: Path, monkeypatch):
    path, _ = _write_physical_complete(tmp_path)
    original_hash = manifest_module._hash_regular_file
    created = False

    def add_entry_during_hashing(output_path: Path):
        nonlocal created
        result = original_hash(output_path)
        if not created:
            created = True
            (path.parent / "raced.txt").write_text("race", encoding="utf-8")
        return result

    monkeypatch.setattr(manifest_module, "_hash_regular_file", add_entry_during_hashing)

    with pytest.raises(ValueError, match="changed while outputs were being verified"):
        load_complete_manifest(
            path,
            artifact_root=tmp_path,
            expected_identity_sha256=path.parent.name,
        )


def test_manifest_snapshot_is_bound_across_environment_verification(
    tmp_path: Path, monkeypatch
):
    path, manifest = _write_physical_complete(tmp_path)
    original_verify = manifest_module.verify_environment_requirements

    def mutate_manifest_during_environment_check(*args, **kwargs):
        hashes = original_verify(*args, **kwargs)
        raw = path.read_bytes()
        python_version = manifest["runtime"]["python"]
        old = f'"python":"{python_version}"'.encode("utf-8")
        new = f'"python":"{"9" * len(python_version)}"'.encode("utf-8")
        mutated = raw.replace(old, new)
        assert len(mutated) == len(raw)
        assert mutated != raw
        path.write_bytes(mutated)
        return hashes

    monkeypatch.setattr(
        manifest_module,
        "verify_environment_requirements",
        mutate_manifest_during_environment_check,
    )

    with pytest.raises(ValueError, match="manifest|directory changed"):
        load_complete_manifest(
            path,
            artifact_root=tmp_path,
            expected_identity_sha256=path.parent.name,
        )


def test_output_hashing_streams_without_buffering_through_manifest_reader(
    tmp_path: Path, monkeypatch
):
    path = tmp_path / "large-output.bin"
    payload = b"a" * (2 * 1024 * 1024 + 17)
    path.write_bytes(payload)

    def forbid_buffered_reader(*args, **kwargs):
        raise AssertionError("output hashing must not buffer via _read_regular_file")

    monkeypatch.setattr(
        manifest_module,
        "_read_regular_file",
        forbid_buffered_reader,
    )

    digest, size = manifest_module._hash_regular_file(path)

    assert digest == hashlib.sha256(payload).hexdigest()
    assert size == len(payload)


def test_safe_reader_rejects_a_nonregular_directory(tmp_path: Path):
    with pytest.raises(ValueError, match="regular|safely open"):
        manifest_module._read_regular_file(tmp_path, "output")


def test_safe_reader_rejects_fifo_without_blocking(tmp_path: Path):
    if not hasattr(os, "mkfifo"):
        pytest.skip("platform has no FIFO support")
    fifo = tmp_path / "output.fifo"
    os.mkfifo(fifo)
    script = (
        "from pathlib import Path;"
        "from gsdiff.experiments.manifest import _read_regular_file;"
        f"_read_regular_file(Path({str(fifo)!r}), 'output')"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        timeout=3,
    )

    assert result.returncode != 0
    assert "regular" in result.stderr or "safely open" in result.stderr


@pytest.mark.parametrize(
    ("requirements_lock", "environment_lock"),
    [
        (_ROOT / "requirements-lock.txt", None),
        (None, _ROOT / "docs" / "reproducibility" / "environment-lock.json"),
    ],
)
def test_load_complete_manifest_rejects_asymmetric_lock_arguments(
    tmp_path: Path,
    requirements_lock: Path | None,
    environment_lock: Path | None,
):
    path, _ = _write_physical_complete(tmp_path)

    with pytest.raises(ValueError, match="provided together"):
        load_complete_manifest(
            path,
            artifact_root=tmp_path,
            expected_identity_sha256=path.parent.name,
            requirements_lock=requirements_lock,
            environment_lock=environment_lock,
        )


def test_load_complete_manifest_rejects_caller_supplied_live_fingerprint(
    tmp_path: Path,
):
    path, _ = _write_physical_complete(tmp_path)

    with pytest.raises(ValueError, match="cannot trust a caller fingerprint"):
        load_complete_manifest(
            path,
            artifact_root=tmp_path,
            expected_identity_sha256=path.parent.name,
            live_fingerprint=_ENVIRONMENT_LOCK["fingerprint"],
        )


def test_load_complete_manifest_rejects_duplicate_json_keys(tmp_path: Path):
    expected_identity = "a" * 64
    path = tmp_path / "runs" / expected_identity / "manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"status":"running","status":"failed"}', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_complete_manifest(
            path,
            artifact_root=tmp_path,
            expected_identity_sha256=expected_identity,
        )


def test_loader_rejects_root_external_path_before_open(tmp_path: Path, monkeypatch):
    expected_identity = "a" * 64
    external = tmp_path / "outside.json"

    def forbidden(*args, **kwargs):
        raise AssertionError("root-external manifest path was opened")

    monkeypatch.setattr(manifest_module, "_load_unique_json", forbidden)
    with pytest.raises(ValueError, match="canonical artifact root"):
        load_complete_manifest(
            external,
            artifact_root=tmp_path,
            expected_identity_sha256=expected_identity,
        )


def test_aggregate_index_rejects_a_dirty_complete_manifest(tmp_path: Path):
    dirty = _complete_manifest(_identity(dirty=True))
    run_identity = dirty["identity_sha256"]
    index = {
        "schema_version": "experiment-aggregate-v1",
        "document_kind": "campaign-index",
        "campaign_id": "primary-v1",
        "campaign_sha256": "1" * 64,
        "protocol_sha256": "2" * 64,
        "scientific_contract_id": "gsdiff-sim-v1",
        "scientific_contract_sha256": _HASH,
        "metric_version": "metrics-v1",
        "expected_identity_sha256s": [run_identity],
        "run_manifests": [{"identity_sha256": run_identity, "manifest_sha256": hashlib.sha256(canonical_json_bytes(dirty)).hexdigest()}],
    }

    run_directory = tmp_path / "runs" / run_identity
    run_directory.mkdir(parents=True)
    path = run_directory / "manifest.json"
    path.write_bytes(canonical_json_bytes(dirty))
    with pytest.raises(ValueError, match="dirty"):
        validate_aggregate_index(
            index,
            manifest_paths={run_identity: path},
            artifact_root=tmp_path,
        )


def test_aggregate_index_binds_record_identity_and_canonical_manifest_hash(tmp_path: Path):
    path, manifest = _write_physical_complete(tmp_path)
    identity = manifest["identity_sha256"]
    index = {
        "schema_version": "experiment-aggregate-v1", "document_kind": "campaign-index",
        "campaign_id": "primary-v1", "campaign_sha256": "1" * 64, "protocol_sha256": "2" * 64,
        "scientific_contract_id": "gsdiff-sim-v1", "scientific_contract_sha256": _HASH,
        "metric_version": "metrics-v1", "expected_identity_sha256s": [identity],
        "run_manifests": [{"identity_sha256": identity, "manifest_sha256": "3" * 64}],
    }

    with pytest.raises(ValueError, match="physical manifest"):
        validate_aggregate_index(
            index,
            manifest_paths={identity: path},
            artifact_root=tmp_path,
        )


def test_primary_and_supplement_indices_share_one_byte_identical_run_manifest(tmp_path: Path):
    manifest_path, manifest = _write_physical_complete(tmp_path)
    identity = manifest["identity_sha256"]
    protocol = manifest["protocol"]
    common = {
        "campaign_sha256": "1" * 64,
        "protocol_sha256": "2" * 64,
        "scientific_contract_id": protocol["scientific_contract_id"],
        "scientific_contract_sha256": protocol[
            "scientific_contract_sha256"
        ],
        "metric_version": "metrics-v1",
    }

    primary = build_aggregate_index(
        campaign_id="primary-v1", expected_identity_sha256s=[identity], manifest_paths={identity: manifest_path}, artifact_root=tmp_path, **common
    )
    supplement = build_aggregate_index(
        campaign_id="supplement-grid-v1", expected_identity_sha256s=[identity], manifest_paths={identity: manifest_path}, artifact_root=tmp_path, **common
    )

    assert manifest_path.read_bytes() == canonical_json_bytes(manifest)
    assert primary["run_manifests"] == supplement["run_manifests"]


def test_build_aggregate_verifies_environment_and_reads_manifest_once(
    tmp_path: Path, monkeypatch
):
    manifest_path, manifest = _write_physical_complete(tmp_path)
    run_identity = manifest["identity_sha256"]
    protocol = manifest["protocol"]
    calls = {"environment": 0, "manifest_read": 0}
    original_environment = manifest_module.verify_environment_requirements
    original_load = manifest_module._load_unique_json

    def count_environment(*args, **kwargs):
        calls["environment"] += 1
        return original_environment(*args, **kwargs)

    def count_manifest_read(*args, **kwargs):
        calls["manifest_read"] += 1
        return original_load(*args, **kwargs)

    monkeypatch.setattr(
        manifest_module,
        "verify_environment_requirements",
        count_environment,
    )
    monkeypatch.setattr(
        manifest_module,
        "_load_unique_json",
        count_manifest_read,
    )

    build_aggregate_index(
        campaign_id="primary-v1",
        campaign_sha256="1" * 64,
        protocol_sha256="2" * 64,
        scientific_contract_id=protocol["scientific_contract_id"],
        scientific_contract_sha256=protocol["scientific_contract_sha256"],
        metric_version="metrics-v1",
        expected_identity_sha256s=[run_identity],
        manifest_paths={run_identity: manifest_path},
        artifact_root=tmp_path,
    )

    assert calls == {"environment": 1, "manifest_read": 1}


def test_validate_aggregate_verifies_environment_and_reads_manifest_once(
    tmp_path: Path, monkeypatch
):
    manifest_path, manifest = _write_physical_complete(tmp_path)
    run_identity = manifest["identity_sha256"]
    protocol = manifest["protocol"]
    index = {
        "schema_version": "experiment-aggregate-v1",
        "document_kind": "campaign-index",
        "campaign_id": "primary-v1",
        "campaign_sha256": "1" * 64,
        "protocol_sha256": "2" * 64,
        "scientific_contract_id": protocol["scientific_contract_id"],
        "scientific_contract_sha256": protocol[
            "scientific_contract_sha256"
        ],
        "metric_version": "metrics-v1",
        "expected_identity_sha256s": [run_identity],
        "run_manifests": [
            {
                "identity_sha256": run_identity,
                "manifest_sha256": hashlib.sha256(
                    manifest_path.read_bytes()
                ).hexdigest(),
            }
        ],
    }
    calls = {"environment": 0, "manifest_read": 0}
    original_environment = manifest_module.verify_environment_requirements
    original_load = manifest_module._load_unique_json

    def count_environment(*args, **kwargs):
        calls["environment"] += 1
        return original_environment(*args, **kwargs)

    def count_manifest_read(*args, **kwargs):
        calls["manifest_read"] += 1
        return original_load(*args, **kwargs)

    monkeypatch.setattr(
        manifest_module,
        "verify_environment_requirements",
        count_environment,
    )
    monkeypatch.setattr(
        manifest_module,
        "_load_unique_json",
        count_manifest_read,
    )

    validate_aggregate_index(
        index,
        manifest_paths={run_identity: manifest_path},
        artifact_root=tmp_path,
    )

    assert calls == {"environment": 1, "manifest_read": 1}


def test_build_aggregate_rejects_extra_manifest_path_before_loading(
    tmp_path: Path, monkeypatch
):
    identity = "1" * 64

    def forbidden(*args, **kwargs):
        raise AssertionError("aggregate attempted to load a manifest")

    monkeypatch.setattr(manifest_module, "_load_complete_manifest_raw", forbidden)

    with pytest.raises(ValueError, match="manifest path identities"):
        build_aggregate_index(
            campaign_id="primary-v1",
            campaign_sha256="2" * 64,
            protocol_sha256="3" * 64,
            scientific_contract_id="gsdiff-sim-v1",
            scientific_contract_sha256="4" * 64,
            metric_version="metrics-v1",
            expected_identity_sha256s=[identity],
            manifest_paths={
                identity: tmp_path / "runs" / identity / "manifest.json",
                "5" * 64: tmp_path / "unexpected.json",
            },
            artifact_root=tmp_path,
        )


def test_validate_aggregate_rejects_extra_manifest_path_before_loading(
    tmp_path: Path, monkeypatch
):
    identity = "1" * 64
    index = {
        "schema_version": "experiment-aggregate-v1",
        "document_kind": "campaign-index",
        "campaign_id": "primary-v1",
        "campaign_sha256": "2" * 64,
        "protocol_sha256": "3" * 64,
        "scientific_contract_id": "gsdiff-sim-v1",
        "scientific_contract_sha256": "4" * 64,
        "metric_version": "metrics-v1",
        "expected_identity_sha256s": [identity],
        "run_manifests": [
            {
                "identity_sha256": identity,
                "manifest_sha256": "6" * 64,
            }
        ],
    }

    def forbidden(*args, **kwargs):
        raise AssertionError("aggregate attempted to load a manifest")

    monkeypatch.setattr(manifest_module, "_load_complete_manifest_raw", forbidden)

    with pytest.raises(ValueError, match="manifest path identities"):
        validate_aggregate_index(
            index,
            manifest_paths={
                identity: tmp_path / "runs" / identity / "manifest.json",
                "5" * 64: tmp_path / "unexpected.json",
            },
            artifact_root=tmp_path,
        )


def test_build_aggregate_cross_checks_contract_during_physical_pass(
    tmp_path: Path,
):
    manifest_path, manifest = _write_physical_complete(tmp_path)
    run_identity = manifest["identity_sha256"]

    with pytest.raises(ValueError, match="protocol"):
        build_aggregate_index(
            campaign_id="primary-v1",
            campaign_sha256="1" * 64,
            protocol_sha256="2" * 64,
            scientific_contract_id="gsdiff-ablation-v1",
            scientific_contract_sha256="6" * 64,
            metric_version="metrics-v1",
            expected_identity_sha256s=[run_identity],
            manifest_paths={run_identity: manifest_path},
            artifact_root=tmp_path,
        )
