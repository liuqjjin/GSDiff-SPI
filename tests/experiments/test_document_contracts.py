from __future__ import annotations

from copy import deepcopy
import importlib
import importlib.util
import json
import os
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from gsdiff.experiments.identity import canonical_json_bytes


ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_SCHEMA_DIRECTORY = ROOT / "schemas"
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
COMMIT = "d" * 40
CONTRACT_SCHEMA_VERSIONS = (
    "phase-evidence-contract-v1",
    "phase-statistics-contract-v1",
    "publication-lock-contract-v1",
    "experiment-statistics-v1",
    "experiment-phase-aggregate-v1",
    "experiment-partial-report-v1",
    "results-lock-v1",
)


def _module(name: str):
    spec = importlib.util.find_spec(name)
    assert spec is not None, f"missing implementation module: {name}"
    return importlib.import_module(name)


def _write_schema(path: Path, document: dict[str, object]) -> None:
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


@pytest.fixture
def schema_directory(tmp_path: Path) -> Path:
    directory = tmp_path / "schemas"
    directory.mkdir()
    _write_schema(
        directory / "sample-child-v1.schema.json",
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "sample-child-v1.schema.json",
            "type": "object",
            "additionalProperties": False,
            "required": ["value"],
            "properties": {
                "value": {"type": "integer", "minimum": 0},
            },
        },
    )
    _write_schema(
        directory / "sample-document-v1.schema.json",
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "sample-document-v1.schema.json",
            "type": "object",
            "additionalProperties": False,
            "required": ["schema_version", "payload"],
            "properties": {
                "schema_version": {"const": "sample-document-v1"},
                "payload": {"$ref": "sample-child-v1.schema.json"},
            },
        },
    )
    return directory


def _sample_document() -> dict[str, object]:
    return {
        "schema_version": "sample-document-v1",
        "payload": {"value": 3},
    }


def test_versioned_json_validates_cross_schema_reference_and_returns_plain_json(
    schema_directory: Path,
):
    module = _module("gsdiff.experiments.versioned_json")
    document = _sample_document()

    validated = module.validate_versioned_json(
        document,
        "sample-document-v1",
        schema_directory=schema_directory,
    )

    assert validated == document
    assert type(validated) is dict
    assert type(validated["payload"]) is dict


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong-version",
        "extra-top-level",
        "extra-nested",
        "bool-as-integer",
    ],
)
def test_versioned_json_rejects_wrong_version_extra_fields_and_native_type_smuggling(
    schema_directory: Path,
    mutation: str,
):
    module = _module("gsdiff.experiments.versioned_json")
    document = _sample_document()
    payload = document["payload"]
    assert type(payload) is dict
    if mutation == "wrong-version":
        document["schema_version"] = "sample-document-v2"
    elif mutation == "extra-top-level":
        document["extra"] = "forbidden"
    elif mutation == "extra-nested":
        payload["extra"] = "forbidden"
    else:
        payload["value"] = True

    with pytest.raises(module.VersionedJSONError):
        module.validate_versioned_json(
            document,
            "sample-document-v1",
            schema_directory=schema_directory,
        )


def test_versioned_json_rejects_invalid_schema_before_document_validation(
    tmp_path: Path,
):
    module = _module("gsdiff.experiments.versioned_json")
    directory = tmp_path / "schemas"
    directory.mkdir()
    _write_schema(
        directory / "invalid-document-v1.schema.json",
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "invalid-document-v1.schema.json",
            "type": "not-a-json-schema-type",
        },
    )

    with pytest.raises(module.VersionedJSONError, match="schema"):
        module.validate_versioned_json(
            {"schema_version": "invalid-document-v1"},
            "invalid-document-v1",
            schema_directory=directory,
        )


def test_versioned_json_registry_accepts_referenced_schema_without_id(
    tmp_path: Path,
):
    module = _module("gsdiff.experiments.versioned_json")
    directory = tmp_path / "schemas"
    directory.mkdir()
    _write_schema(
        directory / "idless-child-v1.schema.json",
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "additionalProperties": False,
            "required": ["value"],
            "properties": {"value": {"type": "integer"}},
        },
    )
    _write_schema(
        directory / "idless-parent-v1.schema.json",
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "idless-parent-v1.schema.json",
            "type": "object",
            "additionalProperties": False,
            "required": ["schema_version", "payload"],
            "properties": {
                "schema_version": {"const": "idless-parent-v1"},
                "payload": {"$ref": "idless-child-v1.schema.json"},
            },
        },
    )

    validated = module.validate_versioned_json(
        {"schema_version": "idless-parent-v1", "payload": {"value": 1}},
        "idless-parent-v1",
        schema_directory=directory,
    )

    assert validated["payload"] == {"value": 1}


def test_versioned_json_loads_only_exact_canonical_bytes(
    tmp_path: Path,
    schema_directory: Path,
):
    module = _module("gsdiff.experiments.versioned_json")
    document = _sample_document()
    path = tmp_path / "document.json"
    payload = canonical_json_bytes(document)
    path.write_bytes(payload)

    loaded, observed = module.load_canonical_versioned_json(
        path,
        "sample-document-v1",
        noun="sample document",
        schema_directory=schema_directory,
    )

    assert loaded == document
    assert observed == payload


@pytest.mark.parametrize(
    "payload",
    [
        b'{"payload":{"value":3},"schema_version":"sample-document-v1"}\n',
        b'{"payload":{"value":3,"value":4},"schema_version":"sample-document-v1"}',
        b'{"payload":{"value":NaN},"schema_version":"sample-document-v1"}',
    ],
)
def test_versioned_json_loader_rejects_noncanonical_duplicate_and_nonfinite_json(
    tmp_path: Path,
    schema_directory: Path,
    payload: bytes,
):
    module = _module("gsdiff.experiments.versioned_json")
    path = tmp_path / "document.json"
    path.write_bytes(payload)

    with pytest.raises(module.VersionedJSONError):
        module.load_canonical_versioned_json(
            path,
            "sample-document-v1",
            noun="sample document",
            schema_directory=schema_directory,
        )


def _path_roles():
    module = _module("gsdiff.experiments.path_roles")
    return module, module.PathRole, module.require_disjoint_path_roles


def test_path_roles_allow_overlapping_read_only_inputs(tmp_path: Path):
    _module_value, PathRole, require_disjoint = _path_roles()
    shared = tmp_path / "shared.json"

    require_disjoint(
        PathRole("aggregate", shared, "read"),
        PathRole("expectations", shared, "read"),
    )


def test_path_roles_reject_exact_read_write_alias(tmp_path: Path):
    module, PathRole, require_disjoint = _path_roles()
    shared = tmp_path / "expectations.json"

    with pytest.raises(module.PathRoleError, match="overlap"):
        require_disjoint(
            PathRole("expectations", shared, "read"),
            PathRole("output", shared, "write"),
        )


def test_path_roles_reject_normalized_dotdot_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module, PathRole, require_disjoint = _path_roles()
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    target = inputs / "expectations.json"
    target.write_bytes(b"authority")
    monkeypatch.chdir(tmp_path)
    alias = Path("inputs") / "unused" / ".." / "expectations.json"

    with pytest.raises(module.PathRoleError, match="overlap"):
        require_disjoint(
            PathRole("expectations", target, "read"),
            PathRole("output", alias, "write"),
        )

    assert target.read_bytes() == b"authority"


def test_path_roles_reject_writer_nested_inside_artifact_root(tmp_path: Path):
    module, PathRole, require_disjoint = _path_roles()
    artifact_root = tmp_path / "artifacts"
    output = artifact_root / "reports" / "aggregate.json"

    with pytest.raises(module.PathRoleError, match="overlap"):
        require_disjoint(
            PathRole("artifact-root", artifact_root, "read"),
            PathRole("output", output, "write"),
        )


def test_path_roles_reject_writer_that_contains_artifact_root(tmp_path: Path):
    module, PathRole, require_disjoint = _path_roles()
    output = tmp_path / "container"
    artifact_root = output / "artifacts"

    with pytest.raises(module.PathRoleError, match="overlap"):
        require_disjoint(
            PathRole("artifact-root", artifact_root, "read"),
            PathRole("output", output, "write"),
        )


def test_path_roles_reject_nested_write_destinations(tmp_path: Path):
    module, PathRole, require_disjoint = _path_roles()
    aggregate = tmp_path / "reports"
    partial = aggregate / "partial.json"

    with pytest.raises(module.PathRoleError, match="overlap"):
        require_disjoint(
            PathRole("output", aggregate, "write"),
            PathRole("partial-report", partial, "write"),
        )


def test_path_roles_reject_existing_hardlink_alias(tmp_path: Path):
    module, PathRole, require_disjoint = _path_roles()
    source = tmp_path / "expectations.json"
    alias = tmp_path / "results-lock.json"
    source.write_bytes(b"authority")
    os.link(source, alias)

    with pytest.raises(module.PathRoleError, match="overlap"):
        require_disjoint(
            PathRole("expectations", source, "read"),
            PathRole("output", alias, "write"),
        )

    assert source.read_bytes() == alias.read_bytes() == b"authority"


def test_path_roles_reject_existing_symlink_alias(tmp_path: Path):
    module, PathRole, require_disjoint = _path_roles()
    source = tmp_path / "expectations.json"
    alias = tmp_path / "results-lock.json"
    source.write_bytes(b"authority")
    try:
        alias.symlink_to(source)
    except OSError as error:
        pytest.skip(f"symlink creation unavailable: {error}")

    with pytest.raises(module.PathRoleError, match="overlap"):
        require_disjoint(
            PathRole("expectations", source, "read"),
            PathRole("output", alias, "write"),
        )


@pytest.mark.skipif(os.name != "nt", reason="Windows case-alias contract")
def test_path_roles_reject_windows_case_alias(tmp_path: Path):
    module, PathRole, require_disjoint = _path_roles()
    source = tmp_path / "Expectations.JSON"
    source.write_bytes(b"authority")
    alias = Path(str(source).swapcase())

    with pytest.raises(module.PathRoleError, match="overlap"):
        require_disjoint(
            PathRole("expectations", source, "read"),
            PathRole("output", alias, "write"),
        )


def test_path_roles_accept_sibling_report_directory(tmp_path: Path):
    _module_value, PathRole, require_disjoint = _path_roles()
    artifact_root = tmp_path / "artifacts"
    expectations = tmp_path / "inputs" / "expectations.json"
    aggregate = tmp_path / "reports" / "aggregate.json"
    partial = tmp_path / "reports" / "partial.json"

    require_disjoint(
        PathRole("artifact-root", artifact_root, "read"),
        PathRole("expectations", expectations, "read"),
        PathRole("output", aggregate, "write"),
        PathRole("partial-report", partial, "write"),
    )


def _logical_key_document(
    *,
    phase_id: str = "selection-replay-v1",
    method_id: str = "method-a",
    seed: int = 7,
) -> dict[str, object]:
    return {
        "phase_id": phase_id,
        "acquisition_config_id": "acquisition-default",
        "method_config_id": "default",
        "method_id": method_id,
        "target_id": "target-a",
        "motion_id": "motion-a",
        "seed": seed,
    }


def _metric_values(*, method_id: str, seed: int) -> dict[str, float]:
    offset = 2.0 if method_id == "method-a" else 0.0
    seed_offset = 1.0 if seed == 11 else 0.0
    return {
        "psnr_global_affine": 30.0 + offset + seed_offset,
        "ssim_global_affine": 0.8 + offset / 100.0,
        "nrmse_global_affine_l2": 0.2 - offset / 100.0,
        "psnr_legacy_per_frame_minmax": 29.0 + offset + seed_offset,
    }


def _summary_row(
    *,
    method_id: str,
    metric: str,
    values: tuple[float, float],
) -> dict[str, object]:
    return {
        "scientific_contract_sha256": SHA_A,
        "acquisition_config_id": "acquisition-default",
        "target_id": "target-a",
        "motion_id": "motion-a",
        "method_id": method_id,
        "method_config_id": "default",
        "metric": metric,
        "per_seed": [
            {"seed": 7, "value": values[0]},
            {"seed": 11, "value": values[1]},
        ],
        "n": 2,
        "mean": (values[0] + values[1]) / 2.0,
        "sample_sd": 0.7071067811865476 if values[0] != values[1] else 0.0,
    }


def _statistics_document() -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for method_id in ("method-a", "method-b"):
        first = _metric_values(method_id=method_id, seed=7)
        second = _metric_values(method_id=method_id, seed=11)
        for metric in (
            "nrmse_global_affine_l2",
            "psnr_global_affine",
            "psnr_legacy_per_frame_minmax",
            "ssim_global_affine",
        ):
            rows.append(
                _summary_row(
                    method_id=method_id,
                    metric=metric,
                    values=(first[metric], second[metric]),
                )
            )
    return {
        "schema_version": "experiment-statistics-v1",
        "phase_id": "selection-replay-v1",
        "required_seeds": [7, 11],
        "n_bootstrap": 10_000,
        "bootstrap_seed": 20260727,
        "summaries": rows,
        "paired_effects": [
            {
                "comparison_id": "method-a-vs-b",
                "scientific_contract_sha256": SHA_A,
                "method_id": "method-a",
                "method_config_id": "default",
                "comparator_id": "method-b",
                "comparator_config_id": "default",
                "metric": "psnr_global_affine",
                "effect_direction": "method_minus_comparator",
                "metric_direction": "higher_is_better",
                "per_seed": [
                    {"seed": 7, "paired_cells": 1, "mean_effect": 2.0},
                    {"seed": 11, "paired_cells": 1, "mean_effect": 2.0},
                ],
                "n": 2,
                "mean": 2.0,
                "sample_sd": 0.0,
                "bootstrap_ci": [2.0, 2.0],
            }
        ],
    }


def _aggregate_record(*, method_id: str, seed: int) -> dict[str, object]:
    return {
        "key": _logical_key_document(method_id=method_id, seed=seed),
        "scientific_contract_id": "scientific-contract-v1",
        "scientific_contract_sha256": SHA_A,
        "method_config_sha256": SHA_B,
        "checkpoints_sha256": {"diffusion-v1": SHA_C},
        "dataset_identity_sha256": SHA_C,
        "run_identity_sha256": SHA_A if method_id == "method-a" else SHA_B,
        "manifest_sha256": SHA_B,
        "metrics_sha256": SHA_C,
        "metric_version": "metrics-v1",
        "code_commit": COMMIT,
        "dependencies_sha256": SHA_A,
        "environment_lock_sha256": SHA_B,
        "source_snapshot_sha256": SHA_C,
        "source_projection_sha256": SHA_A,
        "requested_runtime_device": "cuda:0",
        "execution_profile": "publication-v1",
        "metrics": _metric_values(method_id=method_id, seed=seed),
    }


def _publication_phase_rows() -> list[dict[str, object]]:
    return [
        {"phase_id": "selection-replay-v1", "expected_record_count": 207},
        {"phase_id": "selection-stress-v1", "expected_record_count": 126},
        {"phase_id": "primary-selection-v1", "expected_record_count": 297},
        {"phase_id": "primary-confirmatory-v1", "expected_record_count": 198},
        {"phase_id": "supplement-grid-v1", "expected_record_count": 231},
        {"phase_id": "ood-v1", "expected_record_count": 198},
        {"phase_id": "failure-v1", "expected_record_count": 180},
    ]


def _contract_documents() -> dict[str, dict[str, object]]:
    statistics = _statistics_document()
    publication_phases = _publication_phase_rows()
    return {
        "phase-evidence-contract-v1": {
            "schema_version": "phase-evidence-contract-v1",
            "phase_id": "selection-replay-v1",
            "phase_sha256": SHA_A,
            "expected_record_count": 1,
            "statistics_contract_sha256": SHA_B,
            "expected": [
                {
                    "key": _logical_key_document(),
                    "scientific_contract_id": "gsdiff-sim-v1",
                    "scientific_contract_sha256": SHA_A,
                    "identity_sha256": SHA_C,
                }
            ],
        },
        "phase-statistics-contract-v1": {
            "schema_version": "phase-statistics-contract-v1",
            "phase_id": "primary-confirmatory-v1",
            "phase_sha256": SHA_A,
            "metric_version": "metrics-v1",
            "required_seeds": [73, 101],
            "comparisons": [
                {
                    "comparison_id": "primary-confirmation-v1",
                    "method_id": "gsdiff_tv",
                    "comparator_id": "recinr_se2",
                    "metric": "psnr_global_affine",
                    "method_config_id": "default",
                    "comparator_config_id": "default",
                }
            ],
            "n_bootstrap": 10_000,
            "bootstrap_seed": 20260727,
        },
        "publication-lock-contract-v1": {
            "schema_version": "publication-lock-contract-v1",
            "contract_id": "gsdiff-publication-results-v1",
            "required_phases": publication_phases,
            "required_total_records": 1437,
        },
        "experiment-statistics-v1": statistics,
        "experiment-phase-aggregate-v1": {
            "schema_version": "experiment-phase-aggregate-v1",
            "status": "complete",
            "phase_id": "selection-replay-v1",
            "phase_sha256": SHA_A,
            "metric_version": "metrics-v1",
            "phase_evidence_contract_sha256": SHA_B,
            "statistics_contract_sha256": SHA_C,
            "records": [
                _aggregate_record(method_id=method_id, seed=seed)
                for method_id in ("method-a", "method-b")
                for seed in (7, 11)
            ],
            "summary": statistics,
        },
        "experiment-partial-report-v1": {
            "schema_version": "experiment-partial-report-v1",
            "status": "partial",
            "phase_id": "selection-replay-v1",
            "expected_count": 207,
            "available_complete": 206,
            "missing": [_logical_key_document()],
        },
        "results-lock-v1": {
            "schema_version": "results-lock-v1",
            "status": "complete",
            "publication_lock_contract_sha256": SHA_A,
            "code_commit": COMMIT,
            "metric_version": "metrics-v1",
            "dependencies_sha256": SHA_A,
            "environment_lock_sha256": SHA_B,
            "source_snapshot_sha256": SHA_C,
            "source_projection_sha256": SHA_A,
            "requested_runtime_device": "cuda:0",
            "phases": [
                {
                    "phase_id": row["phase_id"],
                    "phase_sha256": SHA_A,
                    "execution_profile": "publication-v1",
                    "aggregate_sha256": SHA_B,
                    "phase_evidence_contract_sha256": SHA_C,
                    "statistics_contract_sha256": SHA_A,
                    "record_count": row["expected_record_count"],
                }
                for row in publication_phases
            ],
            "total_records": 1437,
            "regeneration_command": [
                "python",
                "scripts/experiments/lock_results.py",
            ],
        },
    }


@pytest.mark.parametrize("schema_version", CONTRACT_SCHEMA_VERSIONS)
def test_repository_contract_schema_exists_is_valid_and_accepts_complete_fixture(
    schema_version: str,
):
    module = _module("gsdiff.experiments.versioned_json")
    path = REPOSITORY_SCHEMA_DIRECTORY / f"{schema_version}.schema.json"

    assert path.is_file(), f"missing repository schema: {path.name}"
    schema = json.loads(path.read_text(encoding="utf-8"))
    assert schema["$id"] == path.name
    Draft202012Validator.check_schema(schema)
    assert module.validate_versioned_json(
        _contract_documents()[schema_version],
        schema_version,
    ) == _contract_documents()[schema_version]


@pytest.mark.parametrize("schema_version", CONTRACT_SCHEMA_VERSIONS)
def test_repository_contract_schemas_reject_undeclared_top_level_fields(
    schema_version: str,
):
    module = _module("gsdiff.experiments.versioned_json")
    document = _contract_documents()[schema_version]
    document["undeclared"] = "must fail closed"

    with pytest.raises(module.VersionedJSONError):
        module.validate_versioned_json(document, schema_version)


@pytest.mark.parametrize(
    ("schema_version", "mutation"),
    [
        ("phase-evidence-contract-v1", "evidence-key-extra"),
        ("phase-statistics-contract-v1", "comparison-extra"),
        ("publication-lock-contract-v1", "publication-wrong-count"),
        ("experiment-statistics-v1", "summary-extra"),
        ("experiment-statistics-v1", "effect-extra"),
        ("experiment-phase-aggregate-v1", "record-extra"),
        ("experiment-phase-aggregate-v1", "checkpoint-id"),
        ("experiment-phase-aggregate-v1", "missing-provenance"),
        ("experiment-partial-report-v1", "missing-key-extra"),
        ("results-lock-v1", "phase-extra"),
        ("results-lock-v1", "missing-contract-hash"),
    ],
)
def test_repository_contract_schemas_reject_nested_shape_and_provenance_drift(
    schema_version: str,
    mutation: str,
):
    module = _module("gsdiff.experiments.versioned_json")
    document = deepcopy(_contract_documents()[schema_version])
    if mutation == "evidence-key-extra":
        document["expected"][0]["key"]["extra"] = 1
    elif mutation == "comparison-extra":
        document["comparisons"][0]["extra"] = 1
    elif mutation == "publication-wrong-count":
        document["required_phases"][0]["expected_record_count"] = 206
    elif mutation == "summary-extra":
        document["summaries"][0]["extra"] = 1
    elif mutation == "effect-extra":
        document["paired_effects"][0]["extra"] = 1
    elif mutation == "record-extra":
        document["records"][0]["extra"] = 1
    elif mutation == "checkpoint-id":
        document["records"][0]["checkpoints_sha256"] = {"Bad ID": SHA_A}
    elif mutation == "missing-provenance":
        del document["records"][0]["method_config_sha256"]
    elif mutation == "missing-key-extra":
        document["missing"][0]["extra"] = 1
    elif mutation == "phase-extra":
        document["phases"][0]["extra"] = 1
    else:
        del document["phases"][0]["statistics_contract_sha256"]

    with pytest.raises(module.VersionedJSONError):
        module.validate_versioned_json(document, schema_version)


@pytest.mark.parametrize(
    ("schema_version", "mutation"),
    [
        ("phase-evidence-contract-v1", "bool-count"),
        ("phase-statistics-contract-v1", "bool-seed"),
        ("publication-lock-contract-v1", "wrong-total"),
        ("experiment-statistics-v1", "bool-bootstrap"),
        ("experiment-phase-aggregate-v1", "bad-sha"),
        ("experiment-phase-aggregate-v1", "bad-commit"),
        ("experiment-phase-aggregate-v1", "bad-device"),
        ("experiment-partial-report-v1", "bool-available"),
        ("results-lock-v1", "bad-phase-sha"),
    ],
)
def test_repository_contract_schemas_reject_native_type_and_digest_smuggling(
    schema_version: str,
    mutation: str,
):
    module = _module("gsdiff.experiments.versioned_json")
    document = deepcopy(_contract_documents()[schema_version])
    if mutation == "bool-count":
        document["expected_record_count"] = True
    elif mutation == "bool-seed":
        document["required_seeds"][0] = True
    elif mutation == "wrong-total":
        document["required_total_records"] = 1436
    elif mutation == "bool-bootstrap":
        document["n_bootstrap"] = True
    elif mutation == "bad-sha":
        document["records"][0]["metrics_sha256"] = "A" * 64
    elif mutation == "bad-commit":
        document["records"][0]["code_commit"] = "d" * 39
    elif mutation == "bad-device":
        document["records"][0]["requested_runtime_device"] = "cuda"
    elif mutation == "bool-available":
        document["available_complete"] = True
    else:
        document["phases"][0]["phase_sha256"] = "0" * 63

    with pytest.raises(module.VersionedJSONError):
        module.validate_versioned_json(document, schema_version)
