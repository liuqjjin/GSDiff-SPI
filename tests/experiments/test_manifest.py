from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

import gsdiff.experiments.manifest as manifest_module

from gsdiff.experiments.identity import (
    build_run_identity,
    canonical_json_bytes,
    requirements_dependencies_sha256,
)
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
_ENVIRONMENT_HASHES = json.loads(
    (_ROOT / "docs" / "reproducibility" / "environment-lock.json").read_text("utf-8")
)["fingerprint_sha256"]


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


def test_dirty_diagnostic_complete_validates_but_is_not_reusable(tmp_path: Path):
    identity = _identity(dirty=True)
    manifest = _complete_manifest(identity)
    validate_manifest(manifest)
    path = tmp_path / "manifest.json"
    path.write_bytes(canonical_json_bytes(manifest))

    with pytest.raises(ValueError, match="dirty"):
        load_complete_manifest(path, expected_identity_sha256=identity.identity_sha256)


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
    path = tmp_path / "manifest.json"
    path.write_bytes(canonical_json_bytes(manifest))

    assert load_complete_manifest(path) is None


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


@pytest.mark.parametrize(
    "unsafe_path",
    ["../escape", "/absolute", "C:/drive", "folder\\file", "folder/CON", "folder/CON.txt", "folder/CONIN$", "folder/COM¹", "folder/LPT³", "folder/name*", "folder/name\x01", "folder//repeat", "folder/name.", "folder/name:stream"],
)
def test_manifest_rejects_unsafe_output_paths(unsafe_path: str):
    manifest = _complete_manifest()
    manifest["metrics"]["path"] = unsafe_path

    with pytest.raises(ValueError, match="path"):
        validate_manifest(manifest)


def _write_physical_complete(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    manifest = _complete_manifest()
    run_directory = tmp_path / manifest["identity_sha256"]
    run_directory.mkdir()
    metrics = run_directory / "metrics.json"
    artifact = run_directory / "reconstruction.npz"
    metrics.write_bytes(b'{"psnr": 30.0}')
    artifact.write_bytes(b"artifact bytes")
    manifest["metrics"]["sha256"] = hashlib.sha256(metrics.read_bytes()).hexdigest()
    manifest["artifacts"] = [
        {
            "role": "reconstruction",
            "path": artifact.name,
            "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            "size_bytes": artifact.stat().st_size,
            "schema_version": "reconstruction-v1",
            "required": True,
        }
    ]
    path = run_directory / "manifest.json"
    path.write_bytes(canonical_json_bytes(manifest))
    return path, manifest


def test_load_complete_manifest_verifies_every_declared_regular_file(tmp_path: Path):
    path, manifest = _write_physical_complete(tmp_path)

    loaded = load_complete_manifest(path, expected_identity_sha256=manifest["identity_sha256"])

    assert loaded == manifest


@pytest.mark.parametrize("fault", ["missing", "unlisted", "mutated", "wrong-size"])
def test_load_complete_manifest_rejects_incomplete_or_changed_outputs(tmp_path: Path, fault: str):
    path, manifest = _write_physical_complete(tmp_path)
    artifact = path.parent / "reconstruction.npz"
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
        load_complete_manifest(path)


def test_load_complete_manifest_rejects_an_undeclared_empty_directory(tmp_path: Path):
    path, _ = _write_physical_complete(tmp_path)
    (path.parent / "empty").mkdir()

    with pytest.raises(ValueError, match="unlisted output directory"):
        load_complete_manifest(path)


def test_load_complete_manifest_rejects_a_symlinked_output_leaf(tmp_path: Path):
    path, manifest = _write_physical_complete(tmp_path)
    artifact = path.parent / "reconstruction.npz"
    target = path.parent / "target.npz"
    target.write_bytes(artifact.read_bytes())
    artifact.unlink()
    try:
        artifact.symlink_to(target)
    except OSError:
        pytest.skip("filesystem does not permit symlink creation")

    with pytest.raises(ValueError, match="symlink|reparse"):
        load_complete_manifest(path, expected_identity_sha256=manifest["identity_sha256"])


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
        load_complete_manifest(path)


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


def test_load_complete_manifest_rejects_duplicate_json_keys(tmp_path: Path):
    path = tmp_path / "manifest.json"
    path.write_text('{"status":"running","status":"failed"}', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_complete_manifest(path)


def test_aggregate_index_rejects_a_dirty_complete_manifest():
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

    with pytest.raises(ValueError, match="clean complete"):
        validate_aggregate_index(index, manifests={run_identity: dirty})


def test_aggregate_index_binds_record_identity_and_canonical_manifest_hash():
    manifest = _complete_manifest()
    identity = manifest["identity_sha256"]
    index = {
        "schema_version": "experiment-aggregate-v1", "document_kind": "campaign-index",
        "campaign_id": "primary-v1", "campaign_sha256": "1" * 64, "protocol_sha256": "2" * 64,
        "scientific_contract_id": "gsdiff-sim-v1", "scientific_contract_sha256": _HASH,
        "metric_version": "metrics-v1", "expected_identity_sha256s": [identity],
        "run_manifests": [{"identity_sha256": identity, "manifest_sha256": "3" * 64}],
    }

    with pytest.raises(ValueError, match="manifest hash"):
        validate_aggregate_index(index, manifests={identity: manifest})


def test_primary_and_supplement_indices_share_one_byte_identical_run_manifest(tmp_path: Path):
    manifest_path, manifest = _write_physical_complete(tmp_path)
    identity = manifest["identity_sha256"]
    common = {
        "campaign_sha256": "1" * 64,
        "protocol_sha256": "2" * 64,
        "scientific_contract_id": "gsdiff-sim-v1",
        "scientific_contract_sha256": _HASH,
        "metric_version": "metrics-v1",
    }

    primary = build_aggregate_index(
        campaign_id="primary-v1", expected_identity_sha256s=[identity], manifest_paths={identity: manifest_path}, **common
    )
    supplement = build_aggregate_index(
        campaign_id="supplement-grid-v1", expected_identity_sha256s=[identity], manifest_paths={identity: manifest_path}, **common
    )

    assert manifest_path.read_bytes() == canonical_json_bytes(manifest)
    assert primary["run_manifests"] == supplement["run_manifests"]
