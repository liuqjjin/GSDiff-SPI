from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import pytest

import gsdiff.experiments.aggregation as aggregation_module
from gsdiff.experiments.aggregation import (
    AggregationIntegrityError,
    IncompletePhaseError,
    LogicalRunKey,
    _merge_record_union,
    build_partial_report,
    load_complete_records,
    merge_aggregate,
    publish_json_atomic,
)
from gsdiff.experiments.identity import canonical_json_bytes
from gsdiff.experiments.contracts import (
    PhaseEvidenceContract,
    StatisticsContract,
)


_PHASE = "primary-selection-v1"
_CONTRACT = "gsdiff-sim-v1"
_CONTRACT_SHA = "a" * 64
_METHOD_CONFIG_SHA = "b" * 64
_CHECKPOINT_SHA_A = "6" * 64
_CHECKPOINT_SHA_B = "7" * 64
_DATASET_SHA = "c" * 64
_DEPENDENCIES_SHA = "d" * 64
_ENVIRONMENT_SHA = "e" * 64
_SOURCE_SNAPSHOT_SHA = "1" * 64
_SOURCE_PROJECTION_SHA = "2" * 64
_COMMIT = "3" * 40


def _key(**changes: object) -> LogicalRunKey:
    values: dict[str, object] = {
        "phase_id": _PHASE,
        "acquisition_config_id": "base",
        "method_config_id": "default",
        "method_id": "gsdiff_tv",
        "target_id": "tank",
        "motion_id": "trans",
        "seed": 7,
    }
    values.update(changes)
    return LogicalRunKey(**values)  # type: ignore[arg-type]


def _metrics(*, definition_version: str = "metrics-v1") -> dict[str, object]:
    return {
        "psnr_global_affine": 31.25,
        "ssim_global_affine": 0.91,
        "nrmse_global_affine_l2": 0.08,
        "psnr_legacy_per_frame_minmax": 29.5,
        "alignment": {"slope": 1.0, "intercept": 0.0},
        "definition_version": definition_version,
        "metric_definition": {},
    }


def _manifest(
    key: LogicalRunKey,
    identity_sha256: str,
    metrics_sha256: str,
    *,
    dataset_sha256: str = _DATASET_SHA,
    method_config_sha256: str = _METHOD_CONFIG_SHA,
    checkpoints_sha256: dict[str, str] | None = None,
) -> dict[str, object]:
    return {
        "schema_version": "experiment-manifest-v1",
        "status": "complete",
        "execution_class": "blind_method_child",
        "metric_version": "metrics-v1",
        "run_id": "display-only",
        "identity_sha256": identity_sha256,
        "protocol": {
            "scientific_contract_id": _CONTRACT,
            "scientific_contract_sha256": _CONTRACT_SHA,
            "target": key.target_id,
            "motion": key.motion_id,
            "seed": key.seed,
            "method": key.method_id,
        },
        "code": {
            "git_commit": _COMMIT,
            "dirty_worktree": False,
            "source_tree_sha256": None,
        },
        "config": {
            "resolved": {
                "phase_id": key.phase_id,
                "acquisition_config_id": key.acquisition_config_id,
                "method_config_sha256": method_config_sha256,
                "runner_execution": {
                    "source_snapshot_sha256": _SOURCE_SNAPSHOT_SHA,
                    "source_projection_sha256": _SOURCE_PROJECTION_SHA,
                    "requested_runtime_device": "cuda:0",
                    "method_info_contract": {
                        "method_id": key.method_id,
                        "method_config_id": key.method_config_id,
                        "execution_profile": "primary-full-v1",
                        "method_config_sha256": method_config_sha256,
                    },
                },
            },
            "sha256": "4" * 64,
        },
        "inputs": {
            "dataset_identity_sha256": dataset_sha256,
            "checkpoints": dict(checkpoints_sha256 or {}),
        },
        "runtime": {
            "dependencies_sha256": _DEPENDENCIES_SHA,
            "environment_lock_sha256": _ENVIRONMENT_SHA,
        },
        "metrics": {
            "version": "metrics-v1",
            "path": "outputs/metrics.json",
            "sha256": metrics_sha256,
        },
    }


def _materialize_run(
    artifact_root: Path,
    manifest: dict[str, object],
    metrics_raw: bytes,
) -> Path:
    identity = manifest["identity_sha256"]
    assert isinstance(identity, str)
    run_dir = artifact_root / "runs" / identity
    (run_dir / "outputs").mkdir(parents=True)
    (run_dir / "outputs" / "metrics.json").write_bytes(metrics_raw)
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    return manifest_path


def _valid_run(
    artifact_root: Path,
    key: LogicalRunKey,
    identity_sha256: str,
    *,
    metrics: dict[str, object] | None = None,
    method_config_sha256: str = _METHOD_CONFIG_SHA,
    checkpoints_sha256: dict[str, str] | None = None,
) -> tuple[Path, dict[str, object], bytes]:
    metric_value = _metrics() if metrics is None else metrics
    metrics_raw = canonical_json_bytes(metric_value)
    manifest = _manifest(
        key,
        identity_sha256,
        hashlib.sha256(metrics_raw).hexdigest(),
        method_config_sha256=method_config_sha256,
        checkpoints_sha256=checkpoints_sha256,
    )
    return _materialize_run(artifact_root, manifest, metrics_raw), manifest, metrics_raw


def _install_loader(monkeypatch, manifests: dict[str, dict[str, object]]) -> None:
    def fake_loader(path: Path, **_kwargs):
        return manifests[path.parent.name]

    monkeypatch.setattr(aggregation_module, "load_complete_manifest", fake_loader)


def _key_document(key: LogicalRunKey) -> dict[str, object]:
    return {
        "phase_id": key.phase_id,
        "acquisition_config_id": key.acquisition_config_id,
        "method_config_id": key.method_config_id,
        "method_id": key.method_id,
        "target_id": key.target_id,
        "motion_id": key.motion_id,
        "seed": key.seed,
    }


def test_load_records_keeps_acquisition_and_method_config_cells_distinct(
    tmp_path: Path, monkeypatch
):
    first = _key()
    second = replace(
        first,
        acquisition_config_id="snr15",
        method_config_id="ablation-j1-v1",
    )
    first_identity = "5" * 64
    second_identity = "6" * 64
    _, first_manifest, _ = _valid_run(tmp_path, first, first_identity)
    _, second_manifest, _ = _valid_run(
        tmp_path,
        second,
        second_identity,
    )
    _install_loader(
        monkeypatch,
        {first_identity: first_manifest, second_identity: second_manifest},
    )

    records = load_complete_records(
        tmp_path,
        phase_id=_PHASE,
        expected_identities={first: first_identity, second: second_identity},
    )

    assert tuple(record.key for record in records) == (first, second)
    assert records[0].metrics == {
        "nrmse_global_affine_l2": 0.08,
        "psnr_global_affine": 31.25,
        "psnr_legacy_per_frame_minmax": 29.5,
        "ssim_global_affine": 0.91,
    }
    assert records[0].manifest_sha256 == hashlib.sha256(
        canonical_json_bytes(first_manifest)
    ).hexdigest()


def test_load_records_preserves_canonical_method_and_checkpoint_provenance(
    tmp_path: Path,
    monkeypatch,
):
    key = _key(method_id="gsdiff_diffusion")
    identity = "5" * 64
    method_config_sha256 = "8" * 64
    checkpoints_sha256 = {
        "prior-b": _CHECKPOINT_SHA_B,
        "prior-a": _CHECKPOINT_SHA_A,
    }
    _, manifest, _ = _valid_run(
        tmp_path,
        key,
        identity,
        method_config_sha256=method_config_sha256,
        checkpoints_sha256=checkpoints_sha256,
    )
    _install_loader(monkeypatch, {identity: manifest})

    record = load_complete_records(
        tmp_path,
        phase_id=_PHASE,
        expected_identities={key: identity},
    )[0]

    assert record.method_config_sha256 == method_config_sha256
    assert tuple(record.checkpoints_sha256.items()) == (
        ("prior-a", _CHECKPOINT_SHA_A),
        ("prior-b", _CHECKPOINT_SHA_B),
    )
    with pytest.raises(TypeError):
        record.checkpoints_sha256["prior-c"] = "9" * 64  # type: ignore[index]


@pytest.mark.parametrize(
    ("mixed_field", "second_method_hash", "second_checkpoints"),
    [
        ("method config", "8" * 64, {"prior": _CHECKPOINT_SHA_A}),
        ("checkpoints", _METHOD_CONFIG_SHA, {"prior": _CHECKPOINT_SHA_B}),
    ],
)
def test_load_records_rejects_mixed_provenance_for_same_method_config(
    tmp_path: Path,
    monkeypatch,
    mixed_field: str,
    second_method_hash: str,
    second_checkpoints: dict[str, str],
):
    first = _key(method_id="gsdiff_diffusion")
    second = replace(first, target_id="digit5")
    first_identity = "5" * 64
    second_identity = "6" * 64
    _, first_manifest, _ = _valid_run(
        tmp_path,
        first,
        first_identity,
        checkpoints_sha256={"prior": _CHECKPOINT_SHA_A},
    )
    _, second_manifest, _ = _valid_run(
        tmp_path,
        second,
        second_identity,
        method_config_sha256=second_method_hash,
        checkpoints_sha256=second_checkpoints,
    )
    _install_loader(
        monkeypatch,
        {first_identity: first_manifest, second_identity: second_manifest},
    )

    with pytest.raises(
        AggregationIntegrityError,
        match=f"mixed {mixed_field} provenance",
    ):
        load_complete_records(
            tmp_path,
            phase_id=_PHASE,
            expected_identities={first: first_identity, second: second_identity},
        )


def test_load_records_rejects_a_key_from_another_phase(tmp_path: Path):
    wrong = _key(phase_id="selection-replay-v1")

    with pytest.raises(AggregationIntegrityError, match="phase"):
        load_complete_records(
            tmp_path,
            phase_id=_PHASE,
            expected_identities={wrong: "5" * 64},
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("phase_id", "selection-replay-v1"),
        ("acquisition_config_id", "stress-snr-db-15-v1"),
    ],
)
def test_load_records_rejects_phase_or_acquisition_relabel_of_complete_run(
    tmp_path: Path,
    monkeypatch,
    field: str,
    replacement: str,
):
    original = _key()
    relabelled = replace(original, **{field: replacement})
    identity = "5" * 64
    _, manifest, _ = _valid_run(tmp_path, original, identity)
    _install_loader(monkeypatch, {identity: manifest})

    with pytest.raises(AggregationIntegrityError, match="phase|acquisition"):
        load_complete_records(
            tmp_path,
            phase_id=relabelled.phase_id,
            expected_identities={relabelled: identity},
        )


def test_load_records_rejects_one_identity_for_two_logical_cells(tmp_path: Path):
    first = _key()
    second = replace(first, acquisition_config_id="snr15")

    with pytest.raises(AggregationIntegrityError, match="duplicate.*identity"):
        load_complete_records(
            tmp_path,
            phase_id=_PHASE,
            expected_identities={first: "5" * 64, second: "5" * 64},
        )


@pytest.mark.parametrize("noncomplete", [False, True])
def test_load_records_reports_every_missing_cell(
    tmp_path: Path, monkeypatch, noncomplete: bool
):
    present = _key()
    missing = replace(present, target_id="digit5")
    present_identity = "5" * 64
    missing_identity = "6" * 64
    _, present_manifest, _ = _valid_run(tmp_path, present, present_identity)
    manifests: dict[str, dict[str, object] | None] = {
        present_identity: present_manifest
    }
    if noncomplete:
        _, missing_manifest, _ = _valid_run(tmp_path, missing, missing_identity)
        manifests[missing_identity] = None

    def fake_loader(path: Path, **_kwargs):
        return manifests[path.parent.name]

    monkeypatch.setattr(aggregation_module, "load_complete_manifest", fake_loader)

    with pytest.raises(IncompletePhaseError) as caught:
        load_complete_records(
            tmp_path,
            phase_id=_PHASE,
            expected_identities={
                present: present_identity,
                missing: missing_identity,
            },
        )

    assert caught.value.missing_keys == (missing,)


def test_load_records_rejects_an_existing_nonfile_manifest_node(
    tmp_path: Path,
):
    key = _key()
    identity = "5" * 64
    manifest_node = tmp_path / "runs" / identity / "manifest.json"
    manifest_node.mkdir(parents=True)

    with pytest.raises(AggregationIntegrityError, match="manifest.*regular file"):
        load_complete_records(
            tmp_path,
            phase_id=_PHASE,
            expected_identities={key: identity},
        )


@pytest.mark.parametrize("message", ["dirty complete manifest", "unblinded evidence"])
def test_load_records_propagates_strict_manifest_refusal(
    tmp_path: Path, monkeypatch, message: str
):
    key = _key()
    identity = "5" * 64
    _valid_run(tmp_path, key, identity)

    def refusing_loader(*_args, **_kwargs):
        raise ValueError(message)

    monkeypatch.setattr(
        aggregation_module,
        "load_complete_manifest",
        refusing_loader,
    )

    with pytest.raises(ValueError, match=message):
        load_complete_records(
            tmp_path,
            phase_id=_PHASE,
            expected_identities={key: identity},
        )


@pytest.mark.parametrize("kind", ["bool", "nan", "duplicate-key"])
def test_load_records_rejects_nonfinite_bool_or_duplicate_metric_evidence(
    tmp_path: Path, monkeypatch, kind: str
):
    key = _key()
    identity = "5" * 64
    metric_value = _metrics()
    if kind == "bool":
        metric_value["psnr_global_affine"] = False
        metrics_raw = canonical_json_bytes(metric_value)
    else:
        metrics_raw = canonical_json_bytes(metric_value)
        if kind == "nan":
            metrics_raw = metrics_raw.replace(
                b'"psnr_global_affine":31.25',
                b'"psnr_global_affine":NaN',
            )
        else:
            metrics_raw = metrics_raw.replace(
                b'"alignment":',
                b'"psnr_global_affine":31.25,"alignment":',
                1,
            )
    manifest = _manifest(
        key,
        identity,
        hashlib.sha256(metrics_raw).hexdigest(),
    )
    _materialize_run(tmp_path, manifest, metrics_raw)
    _install_loader(monkeypatch, {identity: manifest})

    with pytest.raises(AggregationIntegrityError, match="metric|JSON"):
        load_complete_records(
            tmp_path,
            phase_id=_PHASE,
            expected_identities={key: identity},
        )


def test_load_records_rejects_metrics_file_hash_drift(
    tmp_path: Path, monkeypatch
):
    key = _key()
    identity = "5" * 64
    metrics_raw = canonical_json_bytes(_metrics())
    manifest = _manifest(key, identity, "f" * 64)
    _materialize_run(tmp_path, manifest, metrics_raw)
    _install_loader(monkeypatch, {identity: manifest})

    with pytest.raises(AggregationIntegrityError, match="metrics.*hash"):
        load_complete_records(
            tmp_path,
            phase_id=_PHASE,
            expected_identities={key: identity},
        )


@pytest.mark.parametrize(
    "field",
    [
        "contract",
        "metric",
        "code",
        "dependencies",
        "environment",
        "source_snapshot",
        "source_projection",
        "device",
        "profile",
    ],
)
def test_load_records_rejects_mixed_phase_evidence(
    tmp_path: Path, monkeypatch, field: str
):
    first = _key()
    second = replace(first, target_id="digit5")
    first_identity = "5" * 64
    second_identity = "6" * 64
    _, first_manifest, _ = _valid_run(tmp_path, first, first_identity)
    second_metrics = _metrics(
        definition_version="metrics-v2" if field == "metric" else "metrics-v1"
    )
    second_metrics_raw = canonical_json_bytes(second_metrics)
    second_manifest = _manifest(
        second,
        second_identity,
        hashlib.sha256(second_metrics_raw).hexdigest(),
    )
    protocol = second_manifest["protocol"]
    code = second_manifest["code"]
    runtime = second_manifest["runtime"]
    resolved = second_manifest["config"]["resolved"]  # type: ignore[index]
    runner = resolved["runner_execution"]  # type: ignore[index]
    contract = runner["method_info_contract"]  # type: ignore[index]
    if field == "contract":
        protocol["scientific_contract_sha256"] = "7" * 64  # type: ignore[index]
    elif field == "metric":
        second_manifest["metric_version"] = "metrics-v2"
        second_manifest["metrics"]["version"] = "metrics-v2"  # type: ignore[index]
    elif field == "code":
        code["git_commit"] = "7" * 40  # type: ignore[index]
    elif field == "dependencies":
        runtime["dependencies_sha256"] = "7" * 64  # type: ignore[index]
    elif field == "environment":
        runtime["environment_lock_sha256"] = "7" * 64  # type: ignore[index]
    elif field == "source_snapshot":
        runner["source_snapshot_sha256"] = "7" * 64  # type: ignore[index]
    elif field == "source_projection":
        runner["source_projection_sha256"] = "7" * 64  # type: ignore[index]
    elif field == "device":
        runner["requested_runtime_device"] = "cpu"  # type: ignore[index]
    else:
        contract["execution_profile"] = "other-profile-v1"  # type: ignore[index]
    _materialize_run(tmp_path, second_manifest, second_metrics_raw)
    _install_loader(
        monkeypatch,
        {first_identity: first_manifest, second_identity: second_manifest},
    )

    with pytest.raises(AggregationIntegrityError, match="mixed"):
        load_complete_records(
            tmp_path,
            phase_id=_PHASE,
            expected_identities={first: first_identity, second: second_identity},
        )


def test_load_records_rejects_method_config_contract_mismatch(
    tmp_path: Path, monkeypatch
):
    key = _key(method_config_id="ablation-j1-v1")
    identity = "5" * 64
    path, manifest, _ = _valid_run(tmp_path, key, identity)
    contract = manifest["config"]["resolved"]["runner_execution"][  # type: ignore[index]
        "method_info_contract"
    ]
    contract["method_config_id"] = "default"  # type: ignore[index]
    path.write_bytes(canonical_json_bytes(manifest))
    _install_loader(monkeypatch, {identity: manifest})

    with pytest.raises(AggregationIntegrityError, match="method config"):
        load_complete_records(
            tmp_path,
            phase_id=_PHASE,
            expected_identities={key: identity},
        )


def test_load_records_rejects_mixed_datasets_for_one_scientific_cell(
    tmp_path: Path,
    monkeypatch,
):
    first = _key(method_id="gsdiff_tv")
    second = replace(first, method_id="recinr_se2")
    first_identity = "5" * 64
    second_identity = "6" * 64
    _, first_manifest, _ = _valid_run(tmp_path, first, first_identity)
    second_path, second_manifest, _ = _valid_run(
        tmp_path, second, second_identity
    )
    second_manifest["inputs"]["dataset_identity_sha256"] = "7" * 64  # type: ignore[index]
    second_path.write_bytes(canonical_json_bytes(second_manifest))
    _install_loader(
        monkeypatch,
        {first_identity: first_manifest, second_identity: second_manifest},
    )

    with pytest.raises(AggregationIntegrityError, match="mixed dataset"):
        load_complete_records(
            tmp_path,
            phase_id=_PHASE,
            expected_identities={first: first_identity, second: second_identity},
        )


def test_build_partial_report_is_sorted_path_free_and_has_no_statistics():
    later = _key(target_id="usaf", seed=42)
    earlier = _key(target_id="digit5", seed=11)

    report = build_partial_report(
        _PHASE,
        expected_count=3,
        missing_keys=[later, earlier],
    )

    assert report == {
        "schema_version": "experiment-partial-report-v1",
        "status": "partial",
        "phase_id": _PHASE,
        "expected_count": 3,
        "available_complete": 1,
        "missing": [_key_document(earlier), _key_document(later)],
    }


def test_publish_json_atomic_replaces_with_canonical_bytes_and_returns_hash(
    tmp_path: Path,
):
    destination = tmp_path / "partial-report.json"
    destination.write_bytes(b"old")
    document = {"z": 1, "a": [2]}

    digest = publish_json_atomic(destination, document)

    expected = b'{"a":[2],"z":1}'
    assert destination.read_bytes() == expected
    assert digest == hashlib.sha256(expected).hexdigest()
    assert list(tmp_path.glob(f".{destination.name}.*.tmp")) == []


def test_publish_json_atomic_replace_failure_preserves_previous_bytes(
    tmp_path: Path, monkeypatch
):
    destination = tmp_path / "aggregate.json"
    destination.write_bytes(b"old-authority")

    def fail_replace(_source, _destination):
        raise OSError("injected replace failure")

    monkeypatch.setattr(aggregation_module.os, "replace", fail_replace)

    with pytest.raises(OSError, match="injected"):
        publish_json_atomic(destination, {"status": "complete"})

    assert destination.read_bytes() == b"old-authority"
    assert list(tmp_path.glob(f".{destination.name}.*.tmp")) == []


def _aggregate_document(
    records: list[dict[str, object]],
    *,
    phase_id: str = _PHASE,
    status: str = "complete",
    metric_version: str = "metrics-v1",
    summary: object = None,
) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": "experiment-phase-aggregate-v1",
        "status": status,
        "phase_id": phase_id,
        "phase_sha256": "8" * 64,
        "phase_evidence_contract_sha256": "9" * 64,
        "statistics_contract_sha256": "a" * 64,
        "metric_version": metric_version,
        "records": records,
    }
    if summary is not None:
        value["summary"] = summary
    return value


def _aggregate_record(
    key: LogicalRunKey,
    value: float,
    *,
    code_commit: str = _COMMIT,
    environment_lock_sha256: str = _ENVIRONMENT_SHA,
    run_identity_sha256: str | None = None,
    dataset_identity_sha256: str | None = None,
    method_config_sha256: str = _METHOD_CONFIG_SHA,
    checkpoints_sha256: dict[str, str] | None = None,
) -> dict[str, object]:
    run_identity = run_identity_sha256 or hashlib.sha256(
        canonical_json_bytes({"key": _key_document(key), "value": value})
    ).hexdigest()
    return {
        "key": _key_document(key),
        "scientific_contract_id": _CONTRACT,
        "scientific_contract_sha256": _CONTRACT_SHA,
        "method_config_sha256": method_config_sha256,
        "checkpoints_sha256": dict(checkpoints_sha256 or {}),
        "dataset_identity_sha256": dataset_identity_sha256
        or hashlib.sha256(
            canonical_json_bytes(
                {
                    "acquisition_config_id": key.acquisition_config_id,
                    "target_id": key.target_id,
                    "motion_id": key.motion_id,
                    "seed": key.seed,
                }
            )
        ).hexdigest(),
        "run_identity_sha256": run_identity,
        "manifest_sha256": hashlib.sha256(
            f"manifest:{run_identity}".encode("ascii")
        ).hexdigest(),
        "metrics_sha256": hashlib.sha256(
            f"metrics:{run_identity}".encode("ascii")
        ).hexdigest(),
        "metric_version": "metrics-v1",
        "code_commit": code_commit,
        "dependencies_sha256": _DEPENDENCIES_SHA,
        "environment_lock_sha256": environment_lock_sha256,
        "source_snapshot_sha256": _SOURCE_SNAPSHOT_SHA,
        "source_projection_sha256": _SOURCE_PROJECTION_SHA,
        "requested_runtime_device": "cuda:0",
        "execution_profile": "primary-full-v1",
        "metrics": {
            "nrmse_global_affine_l2": 0.1,
            "psnr_global_affine": value,
            "psnr_legacy_per_frame_minmax": value - 1.0,
            "ssim_global_affine": 0.9,
        },
    }


def test_merge_aggregate_is_idempotent_adds_new_records_and_drops_summary():
    first = _key()
    second = replace(first, target_id="digit5")
    first_record = _aggregate_record(first, 31.0)
    existing = _aggregate_document([first_record], summary={"mean": 31.0})
    incoming = _aggregate_document(
        [copy.deepcopy(first_record), _aggregate_record(second, 32.0)],
        summary={"mean": 99.0},
    )

    merged = _merge_record_union(existing, incoming)

    assert merged["records"] == [
        _aggregate_record(second, 32.0),
        first_record,
    ]
    assert "summary" not in merged


def test_public_merge_rebuilds_a_schema_valid_summary_from_bound_contracts():
    first = _key()
    second = replace(first, target_id="digit5")
    first_record = _aggregate_record(first, 31.0)
    second_record = _aggregate_record(second, 32.0)
    existing = _aggregate_document([first_record], summary={"stale": True})
    incoming = _aggregate_document([second_record], summary={"stale": True})
    statistics_contract = StatisticsContract(
        phase_id=_PHASE,
        phase_sha256="8" * 64,
        metric_version="metrics-v1",
        required_seeds=(7,),
        comparisons=(),
        n_bootstrap=16,
        bootstrap_seed=40,
        canonical_sha256="a" * 64,
    )
    phase_evidence_contract = PhaseEvidenceContract(
        phase_id=_PHASE,
        phase_sha256="8" * 64,
        expected_record_count=2,
        statistics_contract_sha256="a" * 64,
        expected_identities=MappingProxyType(
            {
                first: first_record["run_identity_sha256"],
                second: second_record["run_identity_sha256"],
            }
        ),
        expected_scientific_contracts=MappingProxyType(
            {
                first: (_CONTRACT, _CONTRACT_SHA),
                second: (_CONTRACT, _CONTRACT_SHA),
            }
        ),
        canonical_sha256="9" * 64,
    )

    merged = merge_aggregate(
        existing,
        incoming,
        phase_evidence_contract=phase_evidence_contract,
        statistics_contract=statistics_contract,
    )

    assert merged["schema_version"] == "experiment-phase-aggregate-v1"
    assert merged["status"] == "complete"
    assert len(merged["records"]) == 2
    assert merged["summary"]["required_seeds"] == [7]
    assert merged["summary"] != {"stale": True}
    from scripts.experiments.verify_campaign import verify_aggregate_document

    verified = verify_aggregate_document(
        merged,
        statistics_contract=statistics_contract,
        expected_phase_evidence_contract_sha256=(
            phase_evidence_contract.canonical_sha256
        ),
    )
    assert verified["record_count"] == 2

    with pytest.raises(AggregationIntegrityError, match="identities"):
        merge_aggregate(
            existing,
            incoming,
            phase_evidence_contract=replace(
                phase_evidence_contract,
                expected_record_count=3,
            ),
            statistics_contract=statistics_contract,
        )
    wrong_contracts = dict(
        phase_evidence_contract.expected_scientific_contracts
    )
    wrong_contracts[first] = ("wrong-contract-v1", "f" * 64)
    with pytest.raises(AggregationIntegrityError, match="scientific contracts"):
        merge_aggregate(
            existing,
            incoming,
            phase_evidence_contract=replace(
                phase_evidence_contract,
                expected_scientific_contracts=MappingProxyType(
                    wrong_contracts
                ),
            ),
            statistics_contract=statistics_contract,
        )


def test_merge_aggregate_rejects_conflicting_overlap():
    key = _key()
    existing = _aggregate_document([_aggregate_record(key, 31.0)])
    incoming = _aggregate_document([_aggregate_record(key, 32.0)])

    with pytest.raises(AggregationIntegrityError, match="conflicting"):
        _merge_record_union(existing, incoming)


@pytest.mark.parametrize("mixed_field", ["commit", "environment"])
def test_merge_aggregate_rejects_mixed_complete_evidence(mixed_field: str):
    first = _key()
    second = replace(first, target_id="digit5")
    changes = (
        {"code_commit": "9" * 40}
        if mixed_field == "commit"
        else {"environment_lock_sha256": "9" * 64}
    )
    existing = _aggregate_document([_aggregate_record(first, 31.0)])
    incoming = _aggregate_document(
        [_aggregate_record(second, 32.0, **changes)]
    )

    with pytest.raises(AggregationIntegrityError, match="mixed"):
        _merge_record_union(existing, incoming)


@pytest.mark.parametrize(
    ("mixed_field", "second_method_hash", "second_checkpoints"),
    [
        ("method config", "8" * 64, {"prior": _CHECKPOINT_SHA_A}),
        (
            "checkpoints",
            _METHOD_CONFIG_SHA,
            {"prior": _CHECKPOINT_SHA_B},
        ),
    ],
)
def test_merge_aggregate_rejects_mixed_provenance_for_same_method_config(
    mixed_field: str,
    second_method_hash: str,
    second_checkpoints: dict[str, str],
):
    first = _key(method_id="gsdiff_diffusion")
    second = replace(first, target_id="digit5")
    existing = _aggregate_document(
        [
            _aggregate_record(
                first,
                31.0,
                checkpoints_sha256={"prior": _CHECKPOINT_SHA_A},
            )
        ]
    )
    incoming = _aggregate_document(
        [
            _aggregate_record(
                second,
                32.0,
                method_config_sha256=second_method_hash,
                checkpoints_sha256=second_checkpoints,
            )
        ]
    )

    with pytest.raises(
        AggregationIntegrityError,
        match=f"mixed {mixed_field} provenance",
    ):
        _merge_record_union(existing, incoming)


def test_merge_aggregate_rejects_one_run_identity_for_two_logical_keys():
    first = _key()
    second = replace(first, target_id="digit5")
    identity = "7" * 64
    existing = _aggregate_document(
        [_aggregate_record(first, 31.0, run_identity_sha256=identity)]
    )
    incoming = _aggregate_document(
        [_aggregate_record(second, 32.0, run_identity_sha256=identity)]
    )

    with pytest.raises(AggregationIntegrityError, match="run identity"):
        _merge_record_union(existing, incoming)


def test_merge_aggregate_rejects_mixed_datasets_for_one_scientific_cell():
    first = _key(method_id="gsdiff_tv")
    second = replace(first, method_id="recinr_se2")
    existing = _aggregate_document(
        [
            _aggregate_record(
                first,
                31.0,
                dataset_identity_sha256="7" * 64,
            )
        ]
    )
    incoming = _aggregate_document(
        [
            _aggregate_record(
                second,
                32.0,
                dataset_identity_sha256="8" * 64,
            )
        ]
    )

    with pytest.raises(AggregationIntegrityError, match="mixed dataset"):
        _merge_record_union(existing, incoming)


@pytest.mark.parametrize(
    "incoming",
    [
        _aggregate_document([], status="partial"),
        _aggregate_document([], phase_id="primary-confirmatory-v1"),
        _aggregate_document([], metric_version="metrics-v2"),
    ],
)
def test_merge_aggregate_rejects_partial_or_header_mismatch(incoming):
    existing = _aggregate_document([])

    with pytest.raises(AggregationIntegrityError, match="partial|header"):
        _merge_record_union(existing, incoming)


def test_merge_aggregate_rejects_documents_without_the_complete_header():
    minimal = {"status": "complete", "phase_id": _PHASE, "records": []}

    with pytest.raises(AggregationIntegrityError, match="header|schema"):
        _merge_record_union(minimal, copy.deepcopy(minimal))
