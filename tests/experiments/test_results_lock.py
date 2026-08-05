from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import MappingProxyType, ModuleType
import uuid

import pytest

from gsdiff.experiments.aggregation import (
    CompleteMetricRecord,
    IncompletePhaseError,
    LogicalRunKey,
)
from gsdiff.experiments.contracts import (
    PhaseEvidenceContract,
    StatisticsContract,
    build_phase_evidence_contract,
    build_publication_lock_contract_v1,
    build_statistics_contract,
    load_phase_evidence_contract,
    load_publication_lock_contract,
    load_results_lock,
    load_statistics_contract,
)
from gsdiff.experiments.identity import canonical_json_bytes
from gsdiff.experiments.phases import (
    DatasetPlanKey,
    PhasePlan,
    materialize_phase,
    verify_confirmatory_prerequisites,
)
from gsdiff.experiments.protocol import expand_cells, load_protocol
from gsdiff.experiments.statistics import PairedComparison


ROOT = Path(__file__).resolve().parents[2]
ABLATION_PROTOCOL = ROOT / "configs" / "protocols" / "ablations-v1.yaml"
PRIMARY_PROTOCOL = ROOT / "configs" / "protocols" / "primary-v1.yaml"
PHASE_SHA = "a" * 64
CODE_COMMIT = "b" * 40
CONTRACT_SHA = "c" * 64
DEPENDENCIES_SHA = "d" * 64
ENVIRONMENT_SHA = "e" * 64
SOURCE_SNAPSHOT_SHA = "f" * 64
SOURCE_PROJECTION_SHA = "1" * 64
METHOD_CONFIG_SHA = "2" * 64
CHECKPOINT_SHA = "3" * 64
METRICS = {
    "psnr_global_affine": 10.0,
    "ssim_global_affine": 0.5,
    "nrmse_global_affine_l2": 0.25,
    "psnr_legacy_per_frame_minmax": 9.0,
}
PUBLICATION_PHASES = (
    ("selection-replay-v1", 207),
    ("selection-stress-v1", 126),
    ("primary-selection-v1", 297),
    ("primary-confirmatory-v1", 198),
    ("supplement-grid-v1", 231),
    ("ood-v1", 198),
    ("failure-v1", 180),
)


def _load_script(stem: str) -> ModuleType:
    path = ROOT / "scripts" / "experiments" / f"{stem}.py"
    name = f"test_{stem}_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    __import__("sys").modules[name] = module
    spec.loader.exec_module(module)
    return module


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _key(
    phase_id: str,
    method_id: str,
    seed: int,
    *,
    method_config_id: str = "default",
) -> LogicalRunKey:
    return LogicalRunKey(
        phase_id=phase_id,
        acquisition_config_id="base",
        method_config_id=method_config_id,
        method_id=method_id,
        target_id="tank",
        motion_id="trans",
        seed=seed,
    )


def _record(
    phase_id: str,
    method_id: str,
    seed: int,
    *,
    method_config_sha256: str = METHOD_CONFIG_SHA,
    checkpoints_sha256: MappingProxyType[str, str] | None = None,
    environment_lock_sha256: str = ENVIRONMENT_SHA,
) -> CompleteMetricRecord:
    key = _key(phase_id, method_id, seed)
    metrics = dict(METRICS)
    if method_id == "method-a":
        metrics["psnr_global_affine"] = 11.0
    checkpoints = (
        MappingProxyType({"diffusion": CHECKPOINT_SHA})
        if checkpoints_sha256 is None
        else checkpoints_sha256
    )
    return CompleteMetricRecord(
        key=key,
        scientific_contract_id="contract-v1",
        scientific_contract_sha256=CONTRACT_SHA,
        method_config_sha256=method_config_sha256,
        checkpoints_sha256=checkpoints,
        dataset_identity_sha256=_sha(f"dataset:{phase_id}:{seed}"),
        run_identity_sha256=_sha(f"run:{phase_id}:{method_id}:{seed}"),
        manifest_sha256=_sha(f"manifest:{phase_id}:{method_id}:{seed}"),
        metrics_sha256=_sha(f"metrics:{phase_id}:{method_id}:{seed}"),
        metric_version="metrics-v1",
        code_commit=CODE_COMMIT,
        dependencies_sha256=DEPENDENCIES_SHA,
        environment_lock_sha256=environment_lock_sha256,
        source_snapshot_sha256=SOURCE_SNAPSHOT_SHA,
        source_projection_sha256=SOURCE_PROJECTION_SHA,
        requested_runtime_device="cuda:0",
        execution_profile="publication-v1",
        metrics=MappingProxyType(metrics),
    )


def _records(
    phase_id: str = "primary-confirmatory-v1",
) -> tuple[CompleteMetricRecord, ...]:
    return tuple(
        _record(phase_id, method_id, seed)
        for method_id in ("method-a", "method-b")
        for seed in (73, 101)
    )


def _statistics_contract(
    *,
    phase_id: str = "primary-confirmatory-v1",
    phase_sha256: str = PHASE_SHA,
    comparison: bool = True,
) -> StatisticsContract:
    comparisons = (
        (
            PairedComparison(
                comparison_id="method-a-vs-b",
                method_id="method-a",
                comparator_id="method-b",
                metric="psnr_global_affine",
            ),
        )
        if comparison
        else ()
    )
    return StatisticsContract(
        phase_id=phase_id,
        phase_sha256=phase_sha256,
        metric_version="metrics-v1",
        required_seeds=(73, 101),
        comparisons=comparisons,
        n_bootstrap=16,
        bootstrap_seed=40,
        canonical_sha256=_sha(f"statistics:{phase_id}"),
    )


def _phase_evidence_contract(
    records: tuple[CompleteMetricRecord, ...],
    statistics_contract: StatisticsContract,
) -> PhaseEvidenceContract:
    return PhaseEvidenceContract(
        phase_id=statistics_contract.phase_id,
        phase_sha256=statistics_contract.phase_sha256,
        expected_record_count=len(records),
        statistics_contract_sha256=statistics_contract.canonical_sha256,
        expected_identities=MappingProxyType(
            {record.key: record.run_identity_sha256 for record in records}
        ),
        expected_scientific_contracts=MappingProxyType(
            {
                record.key: (
                    record.scientific_contract_id,
                    record.scientific_contract_sha256,
                )
                for record in records
            }
        ),
        canonical_sha256=_sha(f"phase-evidence:{statistics_contract.phase_id}"),
    )


def _aggregate_document(
    aggregate_cli: ModuleType,
    records: tuple[CompleteMetricRecord, ...],
    statistics_contract: StatisticsContract,
    phase_evidence_contract: PhaseEvidenceContract,
) -> dict[str, object]:
    return aggregate_cli.build_aggregate_document(
        records,
        phase_evidence_contract=phase_evidence_contract,
        statistics_contract=statistics_contract,
    )


def _write_canonical(path: Path, document: object) -> None:
    path.write_bytes(canonical_json_bytes(document))


def _selection_plan(phase_id: str = "selection-replay-v1") -> PhasePlan:
    return materialize_phase(
        phase_id,
        ablation_protocol=load_protocol(ABLATION_PROTOCOL),
        primary_protocol=load_protocol(PRIMARY_PROTOCOL),
    )


def _identities_for_plan(plan: PhasePlan) -> dict[LogicalRunKey, str]:
    identities: dict[LogicalRunKey, str] = {}
    for cell in plan.cells:
        key = LogicalRunKey(
            phase_id=plan.phase_id,
            acquisition_config_id=cell.acquisition_config_id,
            method_config_id=cell.method_config_id,
            method_id=cell.method,
            target_id=cell.target,
            motion_id=cell.motion,
            seed=cell.seed,
        )
        identities[key] = _sha(repr(key))
    return identities


def test_scripts_are_import_safe_and_require_new_authority_paths(tmp_path: Path):
    before = set(tmp_path.iterdir())
    aggregate_cli = _load_script("aggregate_campaign")
    verify_cli = _load_script("verify_campaign")
    lock_cli = _load_script("lock_results")
    assert set(tmp_path.iterdir()) == before

    aggregate_actions = {
        action.dest: action
        for action in aggregate_cli._parser()._actions
        if action.dest != "help"
    }
    assert aggregate_actions["phase_id"].required is True
    assert aggregate_actions["phase_evidence_contract"].required is True
    assert aggregate_actions["statistics_contract"].required is True
    for removed in (
        "expectations",
        "required_seed",
        "comparisons",
        "n_bootstrap",
        "bootstrap_seed",
    ):
        assert removed not in aggregate_actions

    verify_actions = {
        action.dest: action
        for action in verify_cli._parser()._actions
        if action.dest != "help"
    }
    assert verify_actions["phase_id"].required is True
    assert verify_actions["phase_evidence_contract"].required is True
    assert verify_actions["statistics_contract"].required is True

    lock_actions = {
        action.dest: action
        for action in lock_cli._parser()._actions
        if action.dest != "help"
    }
    assert lock_actions["publication_contract"].required is True
    assert lock_actions["phase_evidence_contract"].required is True
    assert lock_actions["statistics_contract"].required is True
    assert "required_phase" not in lock_actions


def test_phase_evidence_contract_is_exactly_plan_bound(tmp_path: Path):
    plan = _selection_plan()
    statistics_sha256 = "4" * 64
    document = build_phase_evidence_contract(
        plan,
        expected_identities=_identities_for_plan(plan),
        statistics_contract_sha256=statistics_sha256,
    )
    path = tmp_path / "phase-evidence.json"
    _write_canonical(path, document)

    loaded = load_phase_evidence_contract(
        path,
        expected_plan=plan,
        expected_statistics_contract_sha256=statistics_sha256,
    )
    assert loaded.phase_id == plan.phase_id
    assert loaded.phase_sha256 == plan.phase_sha256
    assert loaded.expected_record_count == 207
    assert loaded.canonical_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert dict(loaded.expected_identities) == _identities_for_plan(plan)
    expected_contracts = {
        LogicalRunKey(
            phase_id=plan.phase_id,
            acquisition_config_id=cell.acquisition_config_id,
            method_config_id=cell.method_config_id,
            method_id=cell.method,
            target_id=cell.target,
            motion_id=cell.motion,
            seed=cell.seed,
        ): (
            cell.scientific_contract_id,
            cell.scientific_contract_sha256,
        )
        for cell in plan.cells
    }
    assert dict(loaded.expected_scientific_contracts) == expected_contracts
    with pytest.raises(TypeError):
        loaded.expected_scientific_contracts[next(iter(expected_contracts))] = (  # type: ignore[index]
            "other-contract",
            "9" * 64,
        )

    mutations: list[dict[str, object]] = []
    wrong_member = deepcopy(document)
    wrong_member["expected"][0]["key"]["seed"] = 999  # type: ignore[index]
    mutations.append(wrong_member)
    wrong_hash = deepcopy(document)
    wrong_hash["phase_sha256"] = "5" * 64
    mutations.append(wrong_hash)
    swapped_statistics = deepcopy(document)
    swapped_statistics["statistics_contract_sha256"] = "5" * 64
    mutations.append(swapped_statistics)
    wrong_scientific_contract = deepcopy(document)
    wrong_scientific_contract["expected"][0][  # type: ignore[index]
        "scientific_contract_sha256"
    ] = "5" * 64
    mutations.append(wrong_scientific_contract)
    missing = deepcopy(document)
    missing["expected"] = missing["expected"][:-1]  # type: ignore[index]
    mutations.append(missing)
    duplicate_identity = deepcopy(document)
    duplicate_identity["expected"][1]["identity_sha256"] = (  # type: ignore[index]
        duplicate_identity["expected"][0]["identity_sha256"]  # type: ignore[index]
    )
    mutations.append(duplicate_identity)
    for index, mutation in enumerate(mutations):
        candidate = tmp_path / f"bad-evidence-{index}.json"
        _write_canonical(candidate, mutation)
        with pytest.raises((TypeError, ValueError)):
            load_phase_evidence_contract(
                candidate,
                expected_plan=plan,
                expected_statistics_contract_sha256=statistics_sha256,
            )


def test_phase_plan_requires_exact_unique_dataset_coverage():
    plan = _selection_plan()
    identities = _identities_for_plan(plan)
    statistics_sha256 = "4" * 64

    invalid_count = replace(
        plan,
        expected_datasets=plan.expected_datasets + 1,
    )
    duplicate_dataset_plans = list(plan.dataset_plans)
    duplicate_dataset_plans[1] = duplicate_dataset_plans[0]
    invalid_duplicate = replace(
        plan,
        dataset_plans=tuple(duplicate_dataset_plans),
    )
    wrong_coverage_plans = list(plan.dataset_plans)
    wrong_coverage_plans[0] = replace(
        wrong_coverage_plans[0],
        target=f"{wrong_coverage_plans[0].target}-other",
    )
    invalid_coverage = replace(
        plan,
        dataset_plans=tuple(wrong_coverage_plans),
    )

    for invalid_plan in (
        invalid_count,
        invalid_duplicate,
        invalid_coverage,
    ):
        with pytest.raises((TypeError, ValueError)):
            build_phase_evidence_contract(
                invalid_plan,
                expected_identities=identities,
                statistics_contract_sha256=statistics_sha256,
            )


def test_statistics_contract_is_derived_from_confirmation_rule(tmp_path: Path):
    protocol = load_protocol(PRIMARY_PROTOCOL)
    cells = tuple(cell for cell in expand_cells(protocol) if cell.seed in {73, 101})
    assert len(cells) == 198
    plan = PhasePlan(
        phase_id="primary-confirmatory-v1",
        phase_sha256="6" * 64,
        cells=cells,
        dataset_plans=tuple(
            sorted(
                {
                    DatasetPlanKey(
                        scientific_contract_id=cell.scientific_contract_id,
                        scientific_contract_sha256=(
                            cell.scientific_contract_sha256
                        ),
                        acquisition_config_id=cell.acquisition_config_id,
                        target=cell.target,
                        motion=cell.motion,
                        seed=cell.seed,
                    )
                    for cell in cells
                }
            )
        ),
        expected_runs=198,
        expected_datasets=18,
    )
    document = build_statistics_contract(plan, source_protocol=protocol)
    assert document["required_seeds"] == [73, 101]
    assert document["comparisons"] == [
        {
            "comparison_id": "primary-confirmation-v1",
            "method_id": "gsdiff_tv",
            "comparator_id": "recinr_se2",
            "metric": "psnr_global_affine",
            "method_config_id": "default",
            "comparator_config_id": "default",
        }
    ]
    assert document["n_bootstrap"] == 10_000
    assert document["bootstrap_seed"] == 20260727
    path = tmp_path / "statistics.json"
    _write_canonical(path, document)
    loaded = load_statistics_contract(
        path,
        expected_plan=plan,
        source_protocol=protocol,
    )
    assert loaded.comparisons[0].comparison_id == "primary-confirmation-v1"

    for field, value in (
        ("required_seeds", [73]),
        ("comparisons", []),
        ("n_bootstrap", 9999),
        ("bootstrap_seed", 1),
    ):
        mutation = deepcopy(document)
        mutation[field] = value
        candidate = tmp_path / f"bad-statistics-{field}.json"
        _write_canonical(candidate, mutation)
        with pytest.raises(ValueError):
            load_statistics_contract(
                candidate,
                expected_plan=plan,
                source_protocol=protocol,
            )


def test_primary_selection_keeps_selection_objective_out_of_paired_statistics():
    protocol = load_protocol(PRIMARY_PROTOCOL)
    plan = materialize_phase(
        "primary-selection-v1",
        ablation_protocol=load_protocol(ABLATION_PROTOCOL),
        primary_protocol=protocol,
    )
    document = build_statistics_contract(plan, source_protocol=protocol)
    assert document["required_seeds"] == [7, 11, 42]
    assert document["comparisons"] == []


def test_publication_lock_contract_is_exactly_seven_phases_and_1437(tmp_path: Path):
    document = build_publication_lock_contract_v1()
    assert [
        (entry["phase_id"], entry["expected_record_count"])
        for entry in document["required_phases"]
    ] == list(PUBLICATION_PHASES)
    assert document["required_total_records"] == 1437
    path = tmp_path / "publication.json"
    _write_canonical(path, document)
    loaded = load_publication_lock_contract(path)
    assert list(loaded.required_phase_counts.items()) == list(PUBLICATION_PHASES)
    assert loaded.required_total_records == 1437

    mutations: list[dict[str, object]] = []
    missing = deepcopy(document)
    missing["required_phases"] = missing["required_phases"][:-1]  # type: ignore[index]
    mutations.append(missing)
    extra = deepcopy(document)
    extra["required_phases"].append(  # type: ignore[union-attr]
        {"phase_id": "pilot-v1", "expected_record_count": 11}
    )
    mutations.append(extra)
    wrong_count = deepcopy(document)
    wrong_count["required_phases"][0]["expected_record_count"] = 206  # type: ignore[index]
    mutations.append(wrong_count)
    duplicate = deepcopy(document)
    duplicate["required_phases"][1] = deepcopy(  # type: ignore[index]
        duplicate["required_phases"][0]  # type: ignore[index]
    )
    mutations.append(duplicate)
    for index, mutation in enumerate(mutations):
        candidate = tmp_path / f"bad-publication-{index}.json"
        _write_canonical(candidate, mutation)
        with pytest.raises(ValueError):
            load_publication_lock_contract(candidate)


def test_schema_aware_atomic_publish_validates_pre_temp_and_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from gsdiff.experiments import aggregation

    destination = tmp_path / "partial.json"
    document = aggregation.build_partial_report(
        "selection-replay-v1",
        2,
        (_key("selection-replay-v1", "method-a", 73),),
    )
    pre_calls: list[str] = []
    byte_calls: list[str] = []
    original_pre = aggregation.validate_versioned_json
    original_bytes = aggregation.validate_canonical_versioned_json_bytes

    def validate_pre(value: object, schema_version: str):
        pre_calls.append(schema_version)
        return original_pre(value, schema_version)

    def validate_bytes(
        payload: bytes,
        schema_version: str,
        *,
        noun: str,
    ):
        byte_calls.append(noun)
        return original_bytes(payload, schema_version, noun=noun)

    monkeypatch.setattr(aggregation, "validate_versioned_json", validate_pre)
    monkeypatch.setattr(
        aggregation,
        "validate_canonical_versioned_json_bytes",
        validate_bytes,
    )
    aggregation.publish_json_atomic(
        destination,
        document,
        schema_version="experiment-partial-report-v1",
    )
    assert pre_calls == ["experiment-partial-report-v1"]
    assert byte_calls == [
        "atomic JSON temporary file",
        "atomic JSON destination",
    ]

    authority = destination.read_bytes()
    with pytest.raises(ValueError):
        aggregation.publish_json_atomic(
            destination,
            {"schema_version": "experiment-partial-report-v1"},
            schema_version="experiment-partial-report-v1",
        )
    assert destination.read_bytes() == authority


def test_aggregate_document_binds_contracts_and_preserves_method_provenance():
    aggregate_cli = _load_script("aggregate_campaign")
    records = _records()
    statistics_contract = _statistics_contract()
    phase_evidence_contract = _phase_evidence_contract(records, statistics_contract)

    document = _aggregate_document(
        aggregate_cli,
        records,
        statistics_contract,
        phase_evidence_contract,
    )

    assert document["phase_evidence_contract_sha256"] == (
        phase_evidence_contract.canonical_sha256
    )
    assert document["statistics_contract_sha256"] == (
        statistics_contract.canonical_sha256
    )
    assert document["summary"]["required_seeds"] == [73, 101]  # type: ignore[index]
    first = document["records"][0]  # type: ignore[index]
    assert first["method_config_sha256"] == METHOD_CONFIG_SHA
    assert first["checkpoints_sha256"] == {"diffusion": CHECKPOINT_SHA}


def test_aggregate_build_rejects_wrong_plan_scientific_contract():
    aggregate_cli = _load_script("aggregate_campaign")
    records = _records()
    statistics_contract = _statistics_contract()
    phase_evidence_contract = _phase_evidence_contract(records, statistics_contract)
    wrong_records = list(records)
    wrong_records[0] = replace(
        wrong_records[0],
        scientific_contract_id="other-contract-v1",
        scientific_contract_sha256="9" * 64,
    )

    with pytest.raises(ValueError, match="scientific contracts"):
        _aggregate_document(
            aggregate_cli,
            tuple(wrong_records),
            statistics_contract,
            phase_evidence_contract,
        )


def test_aggregate_cli_complete_and_partial_paths_are_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    aggregate_cli = _load_script("aggregate_campaign")
    records = _records("selection-replay-v1")
    statistics_contract = _statistics_contract(
        phase_id="selection-replay-v1",
        comparison=False,
    )
    phase_evidence_contract = _phase_evidence_contract(records, statistics_contract)
    plan = PhasePlan(
        phase_id="selection-replay-v1",
        phase_sha256=PHASE_SHA,
        cells=(),
        dataset_plans=(),
        expected_runs=len(records),
        expected_datasets=1,
    )
    monkeypatch.setattr(
        aggregate_cli,
        "materialize_authoritative_phase",
        lambda phase_id: (plan, {}),
    )
    monkeypatch.setattr(
        aggregate_cli,
        "load_statistics_contract",
        lambda *args, **kwargs: statistics_contract,
    )
    monkeypatch.setattr(
        aggregate_cli,
        "load_phase_evidence_contract",
        lambda *args, **kwargs: phase_evidence_contract,
    )
    monkeypatch.setattr(
        aggregate_cli,
        "load_complete_records",
        lambda *args, **kwargs: records,
    )
    output = tmp_path / "aggregate.json"
    partial = tmp_path / "partial.json"
    argv = [
        "--artifact-root",
        str(tmp_path / "artifacts"),
        "--phase-id",
        "selection-replay-v1",
        "--phase-evidence-contract",
        str(tmp_path / "phase-evidence.json"),
        "--statistics-contract",
        str(tmp_path / "statistics.json"),
        "--output",
        str(output),
        "--partial-report",
        str(partial),
    ]
    assert aggregate_cli.main(argv) == 0
    complete = json.loads(output.read_text("utf-8"))
    assert complete["status"] == "complete"
    assert complete["statistics_contract_sha256"] == (
        statistics_contract.canonical_sha256
    )

    authority = output.read_bytes()
    monkeypatch.setattr(
        aggregate_cli,
        "load_complete_records",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            IncompletePhaseError((records[0].key,))
        ),
    )
    assert aggregate_cli.main(argv) != 0
    assert output.read_bytes() == authority
    partial_document = json.loads(partial.read_text("utf-8"))
    assert partial_document["status"] == "partial"
    assert len(partial_document["missing"]) == 1


@pytest.mark.parametrize(
    "mutation",
    (
        "statistics-hash",
        "phase-evidence-hash",
        "required-seeds",
        "bootstrap-count",
        "bootstrap-seed",
        "paired-effects",
        "method-config",
        "checkpoints",
        "mixed-environment",
        "mixed-execution-profile",
        "mixed-dataset",
        "wrong-phase",
        "duplicate-run-identity",
        "duplicate-record",
        "nonfinite",
    ),
)
def test_verifier_rejects_self_authenticated_or_mixed_aggregate(
    mutation: str,
):
    aggregate_cli = _load_script("aggregate_campaign")
    verify_cli = _load_script("verify_campaign")
    records = _records()
    statistics_contract = _statistics_contract()
    phase_evidence_contract = _phase_evidence_contract(records, statistics_contract)
    document = _aggregate_document(
        aggregate_cli,
        records,
        statistics_contract,
        phase_evidence_contract,
    )
    completion = verify_cli.verify_aggregate_document(
        document,
        statistics_contract=statistics_contract,
        expected_phase_evidence_contract_sha256=(
            phase_evidence_contract.canonical_sha256
        ),
    )
    assert completion["record_count"] == 4

    candidate = deepcopy(document)
    if mutation == "statistics-hash":
        candidate["statistics_contract_sha256"] = "7" * 64
    elif mutation == "phase-evidence-hash":
        candidate["phase_evidence_contract_sha256"] = "8" * 64
    elif mutation == "required-seeds":
        candidate["summary"]["required_seeds"] = [73]  # type: ignore[index]
    elif mutation == "bootstrap-count":
        candidate["summary"]["n_bootstrap"] = 15  # type: ignore[index]
    elif mutation == "bootstrap-seed":
        candidate["summary"]["bootstrap_seed"] = 41  # type: ignore[index]
    elif mutation == "paired-effects":
        candidate["summary"]["paired_effects"] = []  # type: ignore[index]
    elif mutation == "method-config":
        candidate["records"][1]["method_config_sha256"] = "9" * 64  # type: ignore[index]
    elif mutation == "checkpoints":
        candidate["records"][1]["checkpoints_sha256"] = {  # type: ignore[index]
            "diffusion": "9" * 64
        }
    elif mutation == "mixed-environment":
        candidate["records"][1]["environment_lock_sha256"] = "9" * 64  # type: ignore[index]
    elif mutation == "mixed-execution-profile":
        candidate["records"][1]["execution_profile"] = "other-profile-v1"  # type: ignore[index]
    elif mutation == "mixed-dataset":
        candidate["records"][2]["dataset_identity_sha256"] = "9" * 64  # type: ignore[index]
    elif mutation == "wrong-phase":
        candidate["records"][0]["key"]["phase_id"] = "other-phase-v1"  # type: ignore[index]
    elif mutation == "duplicate-run-identity":
        candidate["records"][1]["run_identity_sha256"] = (  # type: ignore[index]
            candidate["records"][0]["run_identity_sha256"]  # type: ignore[index]
        )
    elif mutation == "duplicate-record":
        candidate["records"].append(deepcopy(candidate["records"][0]))  # type: ignore[union-attr]
    elif mutation == "nonfinite":
        candidate["records"][0]["metrics"]["psnr_global_affine"] = float("nan")  # type: ignore[index]
    with pytest.raises((TypeError, ValueError)):
        verify_cli.verify_aggregate_document(
            candidate,
            statistics_contract=statistics_contract,
            expected_phase_evidence_contract_sha256=(
                phase_evidence_contract.canonical_sha256
            ),
        )


def test_physical_verifier_reloads_exact_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    aggregate_cli = _load_script("aggregate_campaign")
    verify_cli = _load_script("verify_campaign")
    records = _records()
    statistics_contract = _statistics_contract()
    phase_evidence_contract = _phase_evidence_contract(records, statistics_contract)
    document = _aggregate_document(
        aggregate_cli,
        records,
        statistics_contract,
        phase_evidence_contract,
    )
    monkeypatch.setattr(
        verify_cli,
        "load_complete_records",
        lambda *args, **kwargs: records,
    )
    completion = verify_cli.verify_physical_aggregate_document(
        document,
        artifact_root=tmp_path / "artifacts",
        phase_evidence_contract=phase_evidence_contract,
        statistics_contract=statistics_contract,
    )
    assert completion["record_count"] == 4

    wrong_physical_records = list(records)
    wrong_physical_records[0] = replace(
        wrong_physical_records[0],
        scientific_contract_id="other-contract-v1",
        scientific_contract_sha256="9" * 64,
    )
    monkeypatch.setattr(
        verify_cli,
        "load_complete_records",
        lambda *args, **kwargs: tuple(wrong_physical_records),
    )
    with pytest.raises(ValueError, match="scientific contracts"):
        verify_cli.verify_physical_aggregate_document(
            document,
            artifact_root=tmp_path / "artifacts",
            phase_evidence_contract=phase_evidence_contract,
            statistics_contract=statistics_contract,
        )
    monkeypatch.setattr(
        verify_cli,
        "load_complete_records",
        lambda *args, **kwargs: records,
    )

    changed_records = list(records)
    changed_metrics = dict(records[0].metrics)
    changed_metrics["psnr_global_affine"] = 12.0
    changed_records[0] = replace(
        records[0],
        metrics_sha256=_sha("changed-metrics"),
        metrics=MappingProxyType(changed_metrics),
    )
    changed_document = _aggregate_document(
        aggregate_cli,
        tuple(changed_records),
        statistics_contract,
        phase_evidence_contract,
    )
    with pytest.raises(ValueError, match="physical evidence"):
        verify_cli.verify_physical_aggregate_document(
            changed_document,
            artifact_root=tmp_path / "artifacts",
            phase_evidence_contract=phase_evidence_contract,
            statistics_contract=statistics_contract,
        )


def test_physical_verifier_has_no_count_only_expectations_interface():
    import inspect

    verify_cli = _load_script("verify_campaign")
    parameters = inspect.signature(
        verify_cli.verify_physical_aggregate_document
    ).parameters
    assert "expectations" not in parameters
    assert "phase_evidence_contract" in parameters
    assert "statistics_contract" in parameters


def test_completion_factory_supports_confirmatory_gate_without_raw_claims(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    verify_cli = _load_script("verify_campaign")
    counts = {
        "selection-replay-v1": 207,
        "selection-stress-v1": 126,
        "primary-selection-v1": 297,
    }
    plans = {
        phase_id: PhasePlan(
            phase_id=phase_id,
            phase_sha256=_sha(f"phase:{phase_id}"),
            cells=(),
            dataset_plans=(),
            expected_runs=count,
            expected_datasets=1,
        )
        for phase_id, count in counts.items()
    }
    statistics = {
        phase_id: StatisticsContract(
            phase_id=phase_id,
            phase_sha256=plan.phase_sha256,
            metric_version="metrics-v1",
            required_seeds=(7,),
            comparisons=(),
            n_bootstrap=16,
            bootstrap_seed=40,
            canonical_sha256=_sha(f"statistics:{phase_id}"),
        )
        for phase_id, plan in plans.items()
    }
    evidence = {
        phase_id: PhaseEvidenceContract(
            phase_id=phase_id,
            phase_sha256=plan.phase_sha256,
            expected_record_count=counts[phase_id],
            statistics_contract_sha256=statistics[phase_id].canonical_sha256,
            expected_identities=MappingProxyType({}),
            expected_scientific_contracts=MappingProxyType({}),
            canonical_sha256=_sha(f"evidence:{phase_id}"),
        )
        for phase_id, plan in plans.items()
    }
    monkeypatch.setattr(
        verify_cli,
        "load_statistics_contract",
        lambda path, *, expected_plan, source_protocol: statistics[
            expected_plan.phase_id
        ],
    )
    monkeypatch.setattr(
        verify_cli,
        "load_phase_evidence_contract",
        lambda path, *, expected_plan, expected_statistics_contract_sha256: (
            evidence[expected_plan.phase_id]
        ),
    )

    def verified(document, *, artifact_root, phase_evidence_contract, statistics_contract):
        del document, artifact_root
        return {
            "phase_id": phase_evidence_contract.phase_id,
            "phase_sha256": phase_evidence_contract.phase_sha256,
            "phase_evidence_contract_sha256": phase_evidence_contract.canonical_sha256,
            "statistics_contract_sha256": statistics_contract.canonical_sha256,
            "record_count": phase_evidence_contract.expected_record_count,
            "code_commit": CODE_COMMIT,
            "execution_profile": "publication-v1",
        }

    monkeypatch.setattr(verify_cli, "verify_physical_aggregate_document", verified)
    monkeypatch.setattr(
        verify_cli,
        "load_canonical_document",
        lambda path: ({"phase_id": path.stem}, path.read_bytes()),
    )
    completions = []
    for phase_id, plan in plans.items():
        aggregate_path = tmp_path / f"{phase_id}.json"
        _write_canonical(aggregate_path, {"phase_id": phase_id})
        completions.append(
            verify_cli.load_verified_phase_completion(
                aggregate_path,
                artifact_root=tmp_path / "artifacts",
                expected_plan=plan,
                source_protocol={},
                phase_evidence_contract_path=tmp_path / f"{phase_id}-evidence.json",
                statistics_contract_path=tmp_path / f"{phase_id}-statistics.json",
            )
        )
    gate = verify_confirmatory_prerequisites(completions)
    confirmatory = materialize_phase(
        "primary-confirmatory-v1",
        ablation_protocol=load_protocol(ABLATION_PROTOCOL),
        primary_protocol=load_protocol(PRIMARY_PROTOCOL),
        gate=gate,
    )
    assert confirmatory.expected_runs == 198


def test_authoritative_cli_refuses_phase_without_repository_materializer():
    aggregate_cli = _load_script("aggregate_campaign")
    for phase_id in (
        "selection-stress-v1",
        "primary-confirmatory-v1",
        "supplement-grid-v1",
        "ood-v1",
        "failure-v1",
    ):
        with pytest.raises(ValueError, match="cannot yet be materialized"):
            aggregate_cli.materialize_authoritative_phase(phase_id)


def test_authoritative_cli_materializes_tracked_phase_plans_exactly():
    aggregate_cli = _load_script("aggregate_campaign")
    expected = {
        "selection-decision-v1": 207,
        "selection-replay-v1": 207,
        "primary-selection-v1": 297,
    }
    for phase_id, expected_runs in expected.items():
        plan, source_protocol = aggregate_cli.materialize_authoritative_phase(
            phase_id
        )
        assert plan.phase_id == phase_id
        assert plan.expected_runs == expected_runs
        assert len(plan.cells) == expected_runs
        statistics = build_statistics_contract(
            plan,
            source_protocol=source_protocol,
        )
        assert statistics["phase_id"] == phase_id
        assert statistics["phase_sha256"] == plan.phase_sha256


def _install_lock_fakes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lock_cli: ModuleType,
    *,
    wrong_count_phase: str | None = None,
    conflicting_provenance_phase: str | None = None,
    conflicting_environment_phase: str | None = None,
):
    publication_path = tmp_path / "publication-contract.json"
    _write_canonical(publication_path, build_publication_lock_contract_v1())
    counts = dict(PUBLICATION_PHASES)
    aggregate_paths: dict[str, Path] = {}
    evidence_paths: dict[str, Path] = {}
    statistics_paths: dict[str, Path] = {}
    for phase_id, _count in PUBLICATION_PHASES:
        method_config_sha256 = (
            "9" * 64
            if phase_id == conflicting_provenance_phase
            else METHOD_CONFIG_SHA
        )
        checkpoint_sha256 = (
            "8" * 64
            if phase_id == conflicting_provenance_phase
            else CHECKPOINT_SHA
        )
        aggregate_path = tmp_path / f"{phase_id}-aggregate.json"
        _write_canonical(
            aggregate_path,
            {
                "records": [
                    {
                        "key": {
                            "method_id": "gsdiff_diffusion",
                            "method_config_id": "default",
                        },
                        "method_config_sha256": method_config_sha256,
                        "checkpoints_sha256": {
                            "diffusion": checkpoint_sha256,
                        },
                        "code_commit": CODE_COMMIT,
                        "metric_version": "metrics-v1",
                        "dependencies_sha256": DEPENDENCIES_SHA,
                        "environment_lock_sha256": (
                            "7" * 64
                            if phase_id == conflicting_environment_phase
                            else ENVIRONMENT_SHA
                        ),
                        "source_snapshot_sha256": SOURCE_SNAPSHOT_SHA,
                        "source_projection_sha256": SOURCE_PROJECTION_SHA,
                        "requested_runtime_device": "cuda:0",
                    }
                ]
            },
        )
        evidence_path = tmp_path / f"{phase_id}-evidence.json"
        statistics_path = tmp_path / f"{phase_id}-statistics.json"
        evidence_path.write_bytes(b"evidence-placeholder")
        statistics_path.write_bytes(b"statistics-placeholder")
        aggregate_paths[phase_id] = aggregate_path
        evidence_paths[phase_id] = evidence_path
        statistics_paths[phase_id] = statistics_path

    def materialize(phase_id: str):
        return (
            PhasePlan(
                phase_id=phase_id,
                phase_sha256=_sha(f"phase:{phase_id}"),
                cells=(),
                dataset_plans=(),
                expected_runs=counts[phase_id],
                expected_datasets=1,
            ),
            {},
        )

    def statistics_loader(path, *, expected_plan, source_protocol):
        del path, source_protocol
        return StatisticsContract(
            phase_id=expected_plan.phase_id,
            phase_sha256=expected_plan.phase_sha256,
            metric_version="metrics-v1",
            required_seeds=(7,),
            comparisons=(),
            n_bootstrap=10_000,
            bootstrap_seed=20260727,
            canonical_sha256=_sha(f"statistics:{expected_plan.phase_id}"),
        )

    def evidence_loader(
        path,
        *,
        expected_plan,
        expected_statistics_contract_sha256,
    ):
        del path
        return PhaseEvidenceContract(
            phase_id=expected_plan.phase_id,
            phase_sha256=expected_plan.phase_sha256,
            expected_record_count=expected_plan.expected_runs,
            statistics_contract_sha256=expected_statistics_contract_sha256,
            expected_identities=MappingProxyType({}),
            expected_scientific_contracts=MappingProxyType({}),
            canonical_sha256=_sha(f"evidence:{expected_plan.phase_id}"),
        )

    def physical_verifier(
        document,
        *,
        artifact_root,
        phase_evidence_contract,
        statistics_contract,
    ):
        del document
        assert artifact_root == tmp_path / "artifacts"
        count = phase_evidence_contract.expected_record_count
        if phase_evidence_contract.phase_id == wrong_count_phase:
            count -= 1
        return {
            "phase_id": phase_evidence_contract.phase_id,
            "phase_sha256": phase_evidence_contract.phase_sha256,
            "phase_evidence_contract_sha256": (
                phase_evidence_contract.canonical_sha256
            ),
            "statistics_contract_sha256": statistics_contract.canonical_sha256,
            "record_count": count,
            "code_commit": CODE_COMMIT,
            "execution_profile": f"{phase_evidence_contract.phase_id}-profile",
        }

    monkeypatch.setattr(lock_cli, "materialize_authoritative_phase", materialize)
    monkeypatch.setattr(lock_cli, "load_statistics_contract", statistics_loader)
    monkeypatch.setattr(lock_cli, "load_phase_evidence_contract", evidence_loader)
    monkeypatch.setattr(
        lock_cli,
        "verify_physical_aggregate_document",
        physical_verifier,
    )
    monkeypatch.setattr(
        lock_cli,
        "load_canonical_document",
        lambda path: (json.loads(path.read_text("utf-8")), path.read_bytes()),
    )

    def argv(order: tuple[str, ...], output: Path) -> list[str]:
        values = [
            "--publication-contract",
            str(publication_path),
            "--artifact-root",
            str(tmp_path / "artifacts"),
        ]
        for phase_id in order:
            values.extend(["--phase-id", phase_id])
            values.extend(["--aggregate", str(aggregate_paths[phase_id])])
            values.extend(
                ["--phase-evidence-contract", str(evidence_paths[phase_id])]
            )
            values.extend(
                ["--statistics-contract", str(statistics_paths[phase_id])]
            )
        values.extend(["--output", str(output)])
        return values

    return publication_path, argv


def test_results_lock_is_exact_hash_bound_and_input_order_invariant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    lock_cli = _load_script("lock_results")
    publication_path, argv = _install_lock_fakes(tmp_path, monkeypatch, lock_cli)
    output = tmp_path / "results-lock.json"
    phase_ids = tuple(phase_id for phase_id, _count in PUBLICATION_PHASES)
    assert lock_cli.main(argv(tuple(reversed(phase_ids)), output)) == 0
    first = output.read_bytes()
    assert lock_cli.main(argv(phase_ids, output)) == 0
    assert output.read_bytes() == first

    document = json.loads(first.decode("utf-8"))
    assert document["schema_version"] == "results-lock-v1"
    assert document["status"] == "complete"
    assert document["publication_lock_contract_sha256"] == hashlib.sha256(
        publication_path.read_bytes()
    ).hexdigest()
    assert document["total_records"] == 1437
    assert [phase["phase_id"] for phase in document["phases"]] == list(
        phase_ids
    )
    for phase in document["phases"]:
        phase_id = phase["phase_id"]
        assert phase["record_count"] == dict(PUBLICATION_PHASES)[phase_id]
        assert phase["execution_profile"] == f"{phase_id}-profile"
        assert phase["phase_evidence_contract_sha256"] == _sha(
            f"evidence:{phase_id}"
        )
        assert phase["statistics_contract_sha256"] == _sha(
            f"statistics:{phase_id}"
        )
    assert "--required-phase" not in document["regeneration_command"]
    assert first == canonical_json_bytes(document)


def test_results_lock_loader_rejects_publication_scope_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    lock_cli = _load_script("lock_results")
    publication_path, argv = _install_lock_fakes(tmp_path, monkeypatch, lock_cli)
    output = tmp_path / "results-lock.json"
    phase_ids = tuple(phase_id for phase_id, _count in PUBLICATION_PHASES)
    assert lock_cli.main(argv(phase_ids, output)) == 0
    loaded = load_results_lock(
        output,
        publication_contract_path=publication_path,
    )
    assert loaded["total_records"] == 1437

    document = json.loads(output.read_text("utf-8"))
    mutations: list[dict[str, object]] = []
    subset = deepcopy(document)
    subset["phases"] = subset["phases"][:-1]  # type: ignore[index]
    mutations.append(subset)
    reordered = deepcopy(document)
    reordered["phases"][0], reordered["phases"][1] = (  # type: ignore[index]
        reordered["phases"][1],  # type: ignore[index]
        reordered["phases"][0],  # type: ignore[index]
    )
    mutations.append(reordered)
    wrong_count = deepcopy(document)
    wrong_count["phases"][0]["record_count"] = 206  # type: ignore[index]
    mutations.append(wrong_count)
    wrong_total = deepcopy(document)
    wrong_total["total_records"] = 1436
    mutations.append(wrong_total)
    wrong_contract = deepcopy(document)
    wrong_contract["publication_lock_contract_sha256"] = "9" * 64
    mutations.append(wrong_contract)
    for index, mutation in enumerate(mutations):
        candidate = tmp_path / f"bad-results-lock-{index}.json"
        _write_canonical(candidate, mutation)
        with pytest.raises(ValueError):
            load_results_lock(
                candidate,
                publication_contract_path=publication_path,
            )


def test_results_lock_refuses_incomplete_or_wrong_count_without_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    lock_cli = _load_script("lock_results")
    publication_path, argv = _install_lock_fakes(
        tmp_path,
        monkeypatch,
        lock_cli,
        wrong_count_phase="failure-v1",
    )
    output = tmp_path / "results-lock.json"
    authority = b"existing-results-lock-authority"
    output.write_bytes(authority)
    phase_ids = tuple(phase_id for phase_id, _count in PUBLICATION_PHASES)
    assert lock_cli.main(argv(phase_ids, output)) != 0
    assert output.read_bytes() == authority

    assert lock_cli.main(argv(("selection-replay-v1",), output)) != 0
    assert output.read_bytes() == authority

    duplicate_phase_ids = phase_ids[:-1] + (phase_ids[0],)
    assert lock_cli.main(argv(duplicate_phase_ids, output)) != 0
    assert output.read_bytes() == authority

    mismatched_inputs = argv(phase_ids, output)
    mismatched_inputs.extend(["--aggregate", str(tmp_path / "extra.json")])
    assert lock_cli.main(mismatched_inputs) != 0
    assert output.read_bytes() == authority

    pilot_only = [
        "--publication-contract",
        str(publication_path),
        "--phase-id",
        "pilot-v1",
        "--aggregate",
        str(tmp_path / "pilot-aggregate.json"),
        "--artifact-root",
        str(tmp_path / "artifacts"),
        "--phase-evidence-contract",
        str(tmp_path / "pilot-evidence.json"),
        "--statistics-contract",
        str(tmp_path / "pilot-statistics.json"),
        "--output",
        str(output),
    ]
    assert lock_cli.main(pilot_only) != 0
    assert output.read_bytes() == authority


def test_results_lock_refuses_cross_phase_method_provenance_mixing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    lock_cli = _load_script("lock_results")
    _publication_path, argv = _install_lock_fakes(
        tmp_path,
        monkeypatch,
        lock_cli,
        conflicting_provenance_phase="failure-v1",
    )
    output = tmp_path / "results-lock.json"
    authority = b"existing-results-lock-authority"
    output.write_bytes(authority)
    phase_ids = tuple(phase_id for phase_id, _count in PUBLICATION_PHASES)

    assert lock_cli.main(argv(phase_ids, output)) != 0
    assert output.read_bytes() == authority


def test_results_lock_refuses_cross_phase_common_evidence_mixing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    lock_cli = _load_script("lock_results")
    _publication_path, argv = _install_lock_fakes(
        tmp_path,
        monkeypatch,
        lock_cli,
        conflicting_environment_phase="failure-v1",
    )
    output = tmp_path / "results-lock.json"
    authority = b"existing-results-lock-authority"
    output.write_bytes(authority)
    phase_ids = tuple(phase_id for phase_id, _count in PUBLICATION_PHASES)

    assert lock_cli.main(argv(phase_ids, output)) != 0
    assert output.read_bytes() == authority


def test_results_lock_atomic_failure_preserves_existing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    lock_cli = _load_script("lock_results")
    _publication_path, argv = _install_lock_fakes(tmp_path, monkeypatch, lock_cli)
    output = tmp_path / "results-lock.json"
    authority = b"existing-results-lock-authority"
    output.write_bytes(authority)

    def fail_publish(path: Path, document: object, **kwargs: object) -> None:
        del path, document, kwargs
        raise OSError("injected atomic failure")

    monkeypatch.setattr(lock_cli, "publish_json_atomic", fail_publish)
    phase_ids = tuple(phase_id for phase_id, _count in PUBLICATION_PHASES)
    assert lock_cli.main(argv(phase_ids, output)) != 0
    assert output.read_bytes() == authority


def test_aggregate_path_roles_refuse_alias_before_any_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    aggregate_cli = _load_script("aggregate_campaign")
    shared = tmp_path / "shared.json"
    shared.write_bytes(b"authority")
    touched = False

    def forbidden_materializer(phase_id: str):
        nonlocal touched
        touched = True
        raise AssertionError(phase_id)

    monkeypatch.setattr(
        aggregate_cli,
        "materialize_authoritative_phase",
        forbidden_materializer,
    )
    return_code = aggregate_cli.main(
        [
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--phase-id",
            "selection-replay-v1",
            "--phase-evidence-contract",
            str(shared),
            "--statistics-contract",
            str(tmp_path / "statistics.json"),
            "--output",
            str(shared),
            "--partial-report",
            str(tmp_path / "partial.json"),
        ]
    )
    assert return_code != 0
    assert touched is False
    assert shared.read_bytes() == b"authority"


@pytest.mark.parametrize(
    "protected_path",
    [
        ROOT / "schemas" / "results-lock-v1.schema.json",
        ROOT / "requirements-lock.txt",
        ROOT / "docs" / "reproducibility" / "environment-lock.json",
    ],
)
def test_aggregate_path_roles_protect_implicit_authorities_before_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    protected_path: Path,
):
    aggregate_cli = _load_script("aggregate_campaign")
    authority = protected_path.read_bytes()
    touched = False

    def forbidden_materializer(phase_id: str):
        nonlocal touched
        touched = True
        raise AssertionError(phase_id)

    monkeypatch.setattr(
        aggregate_cli,
        "materialize_authoritative_phase",
        forbidden_materializer,
    )
    return_code = aggregate_cli.main(
        [
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--phase-id",
            "selection-replay-v1",
            "--phase-evidence-contract",
            str(tmp_path / "evidence.json"),
            "--statistics-contract",
            str(tmp_path / "statistics.json"),
            "--output",
            str(protected_path),
            "--partial-report",
            str(tmp_path / "partial.json"),
        ]
    )
    assert return_code != 0
    assert touched is False
    assert protected_path.read_bytes() == authority


def test_aggregate_path_roles_protect_runtime_prefix_before_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    aggregate_cli = _load_script("aggregate_campaign")
    protected_path = Path(sys.prefix) / "codex-path-role-sentinel.json"
    assert not protected_path.exists()
    touched = False

    def forbidden_materializer(phase_id: str):
        nonlocal touched
        touched = True
        raise AssertionError(phase_id)

    monkeypatch.setattr(
        aggregate_cli,
        "materialize_authoritative_phase",
        forbidden_materializer,
    )
    return_code = aggregate_cli.main(
        [
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--phase-id",
            "selection-replay-v1",
            "--phase-evidence-contract",
            str(tmp_path / "evidence.json"),
            "--statistics-contract",
            str(tmp_path / "statistics.json"),
            "--output",
            str(protected_path),
            "--partial-report",
            str(tmp_path / "partial.json"),
        ]
    )
    assert return_code != 0
    assert touched is False
    assert not protected_path.exists()


def test_aggregate_rechecks_path_roles_immediately_before_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    aggregate_cli = _load_script("aggregate_campaign")
    records = _records("selection-replay-v1")
    statistics_contract = _statistics_contract(
        phase_id="selection-replay-v1",
        comparison=False,
    )
    phase_evidence_contract = _phase_evidence_contract(records, statistics_contract)
    plan = PhasePlan(
        phase_id="selection-replay-v1",
        phase_sha256=PHASE_SHA,
        cells=(),
        dataset_plans=(),
        expected_runs=len(records),
        expected_datasets=1,
    )
    monkeypatch.setattr(
        aggregate_cli,
        "materialize_authoritative_phase",
        lambda phase_id: (plan, {}),
    )
    monkeypatch.setattr(
        aggregate_cli,
        "load_statistics_contract",
        lambda *args, **kwargs: statistics_contract,
    )
    monkeypatch.setattr(
        aggregate_cli,
        "load_phase_evidence_contract",
        lambda *args, **kwargs: phase_evidence_contract,
    )
    monkeypatch.setattr(
        aggregate_cli,
        "load_complete_records",
        lambda *args, **kwargs: records,
    )
    checks = 0

    def role_gate(*roles: object) -> None:
        nonlocal checks
        assert roles
        checks += 1
        if checks == 2:
            raise ValueError("changed path roles")

    monkeypatch.setattr(aggregate_cli, "require_disjoint_path_roles", role_gate)
    output = tmp_path / "aggregate.json"
    output.write_bytes(b"authority")
    return_code = aggregate_cli.main(
        [
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--phase-id",
            "selection-replay-v1",
            "--phase-evidence-contract",
            str(tmp_path / "phase-evidence.json"),
            "--statistics-contract",
            str(tmp_path / "statistics.json"),
            "--output",
            str(output),
            "--partial-report",
            str(tmp_path / "partial.json"),
        ]
    )
    assert return_code != 0
    assert checks == 2
    assert output.read_bytes() == b"authority"


def test_results_lock_path_roles_refuse_alias_before_contract_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    lock_cli = _load_script("lock_results")
    shared = tmp_path / "publication.json"
    _write_canonical(shared, build_publication_lock_contract_v1())
    touched = False

    def forbidden_loader(path: Path):
        nonlocal touched
        touched = True
        raise AssertionError(path)

    monkeypatch.setattr(
        lock_cli,
        "load_publication_lock_contract",
        forbidden_loader,
    )
    return_code = lock_cli.main(
        [
            "--publication-contract",
            str(shared),
            "--phase-id",
            "selection-replay-v1",
            "--aggregate",
            str(tmp_path / "aggregate.json"),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--phase-evidence-contract",
            str(tmp_path / "evidence.json"),
            "--statistics-contract",
            str(tmp_path / "statistics.json"),
            "--output",
            str(shared),
        ]
    )
    assert return_code != 0
    assert touched is False
    assert load_publication_lock_contract(shared).required_total_records == 1437


@pytest.mark.parametrize(
    "protected_path",
    [
        ROOT / "schemas" / "results-lock-v1.schema.json",
        ROOT / "requirements-lock.txt",
        ROOT / "docs" / "reproducibility" / "environment-lock.json",
    ],
)
def test_results_lock_path_roles_protect_implicit_authorities_before_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    protected_path: Path,
):
    lock_cli = _load_script("lock_results")
    authority = protected_path.read_bytes()
    touched = False

    def forbidden_loader(path: Path):
        nonlocal touched
        touched = True
        raise AssertionError(path)

    monkeypatch.setattr(
        lock_cli,
        "load_publication_lock_contract",
        forbidden_loader,
    )
    return_code = lock_cli.main(
        [
            "--publication-contract",
            str(tmp_path / "publication.json"),
            "--phase-id",
            "selection-replay-v1",
            "--aggregate",
            str(tmp_path / "aggregate.json"),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--phase-evidence-contract",
            str(tmp_path / "evidence.json"),
            "--statistics-contract",
            str(tmp_path / "statistics.json"),
            "--output",
            str(protected_path),
        ]
    )
    assert return_code != 0
    assert touched is False
    assert protected_path.read_bytes() == authority


def test_results_lock_path_roles_protect_runtime_prefix_before_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    lock_cli = _load_script("lock_results")
    protected_path = Path(sys.prefix) / "codex-path-role-sentinel.json"
    assert not protected_path.exists()
    touched = False

    def forbidden_loader(path: Path):
        nonlocal touched
        touched = True
        raise AssertionError(path)

    monkeypatch.setattr(
        lock_cli,
        "load_publication_lock_contract",
        forbidden_loader,
    )
    return_code = lock_cli.main(
        [
            "--publication-contract",
            str(tmp_path / "publication.json"),
            "--phase-id",
            "selection-replay-v1",
            "--aggregate",
            str(tmp_path / "aggregate.json"),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--phase-evidence-contract",
            str(tmp_path / "evidence.json"),
            "--statistics-contract",
            str(tmp_path / "statistics.json"),
            "--output",
            str(protected_path),
        ]
    )
    assert return_code != 0
    assert touched is False
    assert not protected_path.exists()


def test_results_lock_rechecks_path_roles_immediately_before_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    lock_cli = _load_script("lock_results")
    _publication_path, argv = _install_lock_fakes(tmp_path, monkeypatch, lock_cli)
    checks = 0

    def role_gate(*roles: object) -> None:
        nonlocal checks
        assert roles
        checks += 1
        if checks == 2:
            raise ValueError("changed path roles")

    monkeypatch.setattr(lock_cli, "require_disjoint_path_roles", role_gate)
    output = tmp_path / "results-lock.json"
    output.write_bytes(b"authority")
    phase_ids = tuple(phase_id for phase_id, _count in PUBLICATION_PHASES)
    assert lock_cli.main(argv(phase_ids, output)) != 0
    assert checks == 2
    assert output.read_bytes() == b"authority"
