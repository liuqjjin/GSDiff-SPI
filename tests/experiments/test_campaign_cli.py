import importlib.util
import hashlib
import json
import os
import copy
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from dataclasses import replace
from types import MappingProxyType

import numpy as np
import pytest
import yaml

from gsdiff.data._artifact_dataset import (
    blind_acquisition_spec,
    save_acquisition_data,
)
from gsdiff.data._artifact_identity import array_descriptor
from gsdiff.data._artifact_io import artifact_sha256
from gsdiff.data._artifact_models import SPIAcquisitionData
from gsdiff.experiments.child_outputs import (
    validate_method_child_outputs_v2,
)
from gsdiff.experiments.execution import MaterializedMethodRequest
from gsdiff.experiments.identity import canonical_json_bytes
from gsdiff.experiments.methods import (
    AlgorithmSeed,
    derive_algorithm_seed,
    resolve_method_semantics,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_DATASETS_SCRIPT = (
    REPO_ROOT / "scripts" / "experiments" / "build_datasets.py"
)
RUN_CAMPAIGN_SCRIPT = REPO_ROOT / "scripts" / "experiments" / "run_campaign.py"
CONTROLLED_RUNTIME = {
    "dependencies_sha256": "1" * 64,
    "environment_lock_sha256": "2" * 64,
}
CONTROLLED_COMMIT = "3" * 40


def _controlled_git_state(*, dirty=False):
    return {
        "baseline": "4" * 40,
        "branch": "test-campaign-cli",
        "commit": CONTROLLED_COMMIT,
        "dirty": dirty,
    }


def _load_build_datasets_cli():
    spec = importlib.util.spec_from_file_location(
        "gsdiff_build_datasets_cli",
        BUILD_DATASETS_SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_run_campaign_cli():
    spec = importlib.util.spec_from_file_location(
        "gsdiff_run_campaign_cli",
        RUN_CAMPAIGN_SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_campaign_device_vocabulary_requires_indexed_cuda():
    parser = _load_run_campaign_cli()._parser()

    parsed = parser.parse_args(
        [
            "--protocol",
            str(REPO_ROOT / "configs/protocols/pilot-v1.yaml"),
            "--phase",
            "pilot-v1",
            "--artifact-root",
            str(REPO_ROOT / "artifacts"),
            "--device",
            "cuda:0",
        ]
    )
    assert parsed.device == "cuda:0"
    assert parsed.phase == "pilot-v1"
    for alias in ("cuda", "cuda:00", "cuda:01"):
        with pytest.raises(SystemExit):
            parser.parse_args(
                [
                    "--protocol",
                    str(REPO_ROOT / "configs/protocols/pilot-v1.yaml"),
                    "--phase",
                    "pilot-v1",
                    "--artifact-root",
                    str(REPO_ROOT / "artifacts"),
                    "--device",
                    alias,
                ]
            )


@pytest.mark.parametrize("missing", ["phase", "artifact-root"])
def test_campaign_parser_requires_explicit_phase_and_artifact_root(missing):
    parser = _load_run_campaign_cli()._parser()
    arguments = [
        "--protocol",
        str(REPO_ROOT / "configs/protocols/pilot-v1.yaml"),
        "--phase",
        "pilot-v1",
        "--artifact-root",
        str(REPO_ROOT / "artifacts"),
        "--device",
        "cpu",
    ]
    option = f"--{missing}"
    index = arguments.index(option)
    del arguments[index : index + 2]

    with pytest.raises(SystemExit):
        parser.parse_args(arguments)


def test_campaign_phase_and_acquisition_labels_enter_identity_base_config():
    cli = _load_run_campaign_cli()
    digest = "a" * 64

    decision = cli._phase_identity_base_config(
        phase_id="selection-decision-v1",
        acquisition_config_id="base",
        method_config_sha256=digest,
    )
    replay = cli._phase_identity_base_config(
        phase_id="selection-replay-v1",
        acquisition_config_id="base",
        method_config_sha256=digest,
    )
    stress = cli._phase_identity_base_config(
        phase_id="selection-decision-v1",
        acquisition_config_id="stress-snr-db-15-v1",
        method_config_sha256=digest,
    )

    assert decision == {
        "phase_id": "selection-decision-v1",
        "acquisition_config_id": "base",
        "method_config_sha256": digest,
    }
    assert decision != replay
    assert decision != stress


@pytest.mark.parametrize("phase", ["pilot-v1", "ood-v1", "failure-v1"])
def test_direct_campaign_phases_have_exact_execution_membership(phase):
    cli = _load_run_campaign_cli()

    assert cli._require_materialized_phase_execution(phase) is None


@pytest.mark.parametrize(
    "phase",
    [
        "selection-decision-v1",
        "selection-replay-v1",
        "selection-stress-v1",
        "primary-selection-v1",
        "primary-confirmatory-v1",
        "supplement-grid-v1",
    ],
)
def test_partitioned_or_delta_phases_fail_closed_until_cli_materializes_them(
    phase,
):
    cli = _load_run_campaign_cli()

    with pytest.raises(ValueError, match="materialization"):
        cli._require_materialized_phase_execution(phase)


@pytest.mark.parametrize("phase", ["pilot", "Pilot-v1", "pilot-v01", "pilot_v1"])
def test_campaign_parser_rejects_noncanonical_phase_ids(phase):
    parser = _load_run_campaign_cli()._parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--protocol",
                str(REPO_ROOT / "configs/protocols/pilot-v1.yaml"),
                "--phase",
                phase,
                "--artifact-root",
                str(REPO_ROOT / "artifacts"),
                "--device",
                "cpu",
            ]
        )


def test_campaign_runtime_metadata_uses_requested_cuda_index(monkeypatch):
    import torch

    cli = _load_run_campaign_cli()
    observed = []
    monkeypatch.setattr(
        torch.cuda,
        "get_device_name",
        lambda index: observed.append(index) or f"gpu-{index}",
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    runtime = {
        "python_version": "3.12",
        "torch_version": "2.8",
        "cuda_version": "12.8",
        "gpu_name": "gpu-0",
        "os": "Windows",
    }

    metadata = cli._runtime_manifest(runtime, "cuda:1")

    assert metadata["gpu"] == "gpu-1"
    assert observed == [1]


def test_out_of_range_cuda_is_refused_before_artifacts_or_source_access(
    tmp_path,
    capsys,
    monkeypatch,
):
    import torch

    cli = _load_run_campaign_cli()
    artifact_root = tmp_path / "must-not-be-created"
    campaign = {
        "document_kind": "campaign",
        "campaign_id": "pilot-v1",
        "execution_ready": True,
        "method_budgets": {"dgi": 1},
    }
    monkeypatch.setattr(cli, "load_protocol", lambda path: campaign)
    monkeypatch.setattr(
        cli,
        "_require_versioned_budget_contract",
        lambda value: None,
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid CUDA index reached source/artifact access")

    monkeypatch.setattr(torch.cuda, "get_device_name", forbidden)
    monkeypatch.setattr(cli, "git_state", forbidden)

    return_code = cli.main(
        [
            "--protocol",
            str(tmp_path / "ready.yaml"),
            "--phase",
            "pilot-v1",
            "--artifact-root",
            str(artifact_root),
            "--device",
            "cuda:999999",
        ]
    )

    captured = capsys.readouterr()
    assert return_code == 1
    assert captured.out == ""
    assert captured.err == "campaign execution refused: ValueError\n"
    assert not artifact_root.exists()


def test_campaign_preserves_authoritative_multi_asset_logical_mapping():
    cli = _load_run_campaign_cli()
    assets = {
        "descriptor": "1" * 64,
        "font": "1" * 64,
        "renderer": "1" * 64,
    }
    target = {
        "id": "digit5",
        "descriptor": "char:5",
        "assets_sha256": assets,
    }

    assert cli._identity_asset_mapping(target) == assets


def test_campaign_projects_file_asset_path_to_target_identity_key():
    cli = _load_run_campaign_cli()
    digest = "2" * 64
    target = {
        "id": "tank",
        "descriptor": "assets/tank.png",
        "assets_sha256": {"assets/tank.png": digest},
    }

    assert cli._identity_asset_mapping(target) == {"tank": digest}


def test_ready_campaign_planner_uses_authoritative_blind_acquisition_spec():
    cli = _load_run_campaign_cli()

    assert (
        cli._run_ready_campaign.__globals__["blind_acquisition_spec"]
        is blind_acquisition_spec
    )


def _write_hostile_pythonpath(attacker_root: Path, marker: Path) -> None:
    attacker_root.mkdir()
    payload = (
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text(__name__, encoding='utf-8')\n"
    )
    for name in ("numpy.py", "torch.py"):
        (attacker_root / name).write_text(payload, encoding="utf-8")


def test_campaign_script_refuses_nonisolated_python_before_dependency_import(
    tmp_path: Path,
):
    marker = tmp_path / "dependency-imported"
    attacker_root = tmp_path / "attacker"
    _write_hostile_pythonpath(attacker_root, marker)
    environment = {**os.environ, "PYTHONPATH": str(attacker_root)}

    result = subprocess.run(
        [sys.executable, str(RUN_CAMPAIGN_SCRIPT)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == (
        "campaign execution refused: isolated-python-required\n"
    )
    assert not marker.exists()


def test_campaign_script_isolated_mode_ignores_pythonpath_and_sitecustomize(
    tmp_path: Path,
):
    marker = tmp_path / "hostile-code-executed"
    attacker_root = tmp_path / "attacker"
    _write_hostile_pythonpath(attacker_root, marker)
    (attacker_root / "sitecustomize.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('sitecustomize', encoding='utf-8')\n",
        encoding="utf-8",
    )
    environment = {**os.environ, "PYTHONPATH": str(attacker_root)}

    result = subprocess.run(
        [
            str(Path(sys.executable).resolve()),
            "-I",
            "-B",
            "-X",
            "utf8",
            str(RUN_CAMPAIGN_SCRIPT.resolve()),
            "--protocol",
            str(REPO_ROOT / "configs/protocols/pilot-v1.yaml"),
            "--phase",
            "pilot-v1",
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--device",
            "cpu",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == "campaign execution refused: execution-not-ready\n"
    assert not marker.exists()


def test_real_corrected_glyph_dataset_preserves_all_catalog_asset_keys(
    tmp_path: Path,
):
    from gsdiff.data.artifacts import (
        publish_dataset,
        resolve_target_snapshot,
    )

    build_cli = _load_build_datasets_cli()
    run_cli = _load_run_campaign_cli()
    plan = build_cli.plan_campaign_datasets(
        repo_root=REPO_ROOT,
        protocol_path=REPO_ROOT / "configs/protocols/pilot-v1.yaml",
        runtime=CONTROLLED_RUNTIME,
        generator_commit=CONTROLLED_COMMIT,
    )
    request = plan.requests[0]
    arguments = request.generation_arguments()
    arguments["target_snapshot"] = resolve_target_snapshot(
        repo_root=REPO_ROOT,
        target_id="digit5",
        descriptor="char:5",
        H=32,
        W=32,
    )
    generated = build_cli.generate_corrected_dataset(**arguments)
    publish_dataset(tmp_path, generated)
    verified = next(iter(run_cli._dataset_catalog(tmp_path).values()))
    dataset_assets = verified.manifest["resolved_generator_config"]["target"][
        "assets_sha256"
    ]

    assert set(dataset_assets) == {"descriptor", "font", "renderer"}
    target = verified.manifest["resolved_generator_config"]["target"]
    assert run_cli._identity_asset_mapping(target) == dataset_assets


def test_failure_campaign_plans_all_acquisition_cells_without_old_key_collisions():
    build_cli = _load_build_datasets_cli()
    run_cli = _load_run_campaign_cli()
    protocol_path = REPO_ROOT / "configs/protocols/failure-v1.yaml"
    campaign = run_cli.load_protocol(protocol_path)
    plan = build_cli.plan_campaign_datasets(
        repo_root=REPO_ROOT,
        protocol_path=protocol_path,
        runtime=CONTROLLED_RUNTIME,
        generator_commit=CONTROLLED_COMMIT,
    )
    cells = run_cli.expand_cells(campaign)
    old_keys = {
        (
            cell.scientific_contract_id,
            cell.scientific_contract_sha256,
            cell.target,
            cell.motion,
            cell.seed,
        )
        for cell in cells
    }
    selected_requests = {
        run_cli._planned_dataset_request_for_cell(
            cell,
            campaign,
            plan.requests,
        ).request_sha256
        for cell in cells
    }

    assert len(old_keys) == 6
    assert len(plan.requests) == 36
    assert len(selected_requests) == 36


def test_runner_dataset_catalog_rejects_staging_directory(tmp_path: Path):
    run_cli = _load_run_campaign_cli()
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    (datasets / f".{('a' * 64)}.staging-test").mkdir()

    with pytest.raises(ValueError, match="staging|rejected"):
        run_cli._dataset_catalog(tmp_path)


def test_runner_dataset_catalog_selects_exact_frozen_plan_request(
    tmp_path: Path,
):
    from gsdiff.data.artifacts import publish_dataset

    build_cli = _load_build_datasets_cli()
    run_cli = _load_run_campaign_cli()
    plan = build_cli.plan_campaign_datasets(
        repo_root=REPO_ROOT,
        protocol_path=REPO_ROOT / "configs/protocols/pilot-v1.yaml",
        runtime=CONTROLLED_RUNTIME,
        generator_commit=CONTROLLED_COMMIT,
    )
    request = plan.requests[0]
    generated = build_cli.generate_corrected_dataset(
        **request.generation_arguments()
    )
    published = publish_dataset(tmp_path, generated)

    selected = run_cli._dataset_catalog(
        tmp_path,
        expected_requests=(request,),
    )

    assert tuple(selected) == (request.request_sha256,)
    assert (
        selected[request.request_sha256].dataset_identity_sha256
        == published.verified.dataset_identity_sha256
    )


def test_runner_dataset_catalog_rejects_staging_added_during_verification(
    tmp_path: Path,
    monkeypatch,
):
    from gsdiff.data.artifacts import publish_dataset

    build_cli = _load_build_datasets_cli()
    run_cli = _load_run_campaign_cli()
    plan = build_cli.plan_campaign_datasets(
        repo_root=REPO_ROOT,
        protocol_path=REPO_ROOT / "configs/protocols/pilot-v1.yaml",
        runtime=CONTROLLED_RUNTIME,
        generator_commit=CONTROLLED_COMMIT,
    )
    request = plan.requests[0]
    generated = build_cli.generate_corrected_dataset(
        **request.generation_arguments()
    )
    publish_dataset(tmp_path, generated)
    original_verify = run_cli.verify_dataset_directory
    injected = False

    def add_staging_then_verify(*args, **kwargs):
        nonlocal injected
        if not injected:
            injected = True
            staging = tmp_path / "datasets" / f".{('b' * 64)}.staging-race"
            staging.mkdir()
        return original_verify(*args, **kwargs)

    monkeypatch.setattr(
        run_cli,
        "verify_dataset_directory",
        add_staging_then_verify,
    )

    with pytest.raises(ValueError, match="staging|rejected"):
        run_cli._dataset_catalog(
            tmp_path,
            expected_requests=(request,),
        )


@pytest.mark.parametrize(
    ("protocol_name", "phase"),
    [
        ("pilot-v1.yaml", "pilot-v1"),
        ("supplement-grid-v1.yaml", "supplement-grid-v1"),
        ("ood-v1.yaml", "ood-v1"),
        ("failure-v1.yaml", "failure-v1"),
        ("primary-v1.yaml", "primary-selection-v1"),
        ("primary-v1.yaml", "primary-confirmatory-v1"),
        ("ablations-v1.yaml", "selection-decision-v1"),
        ("ablations-v1.yaml", "selection-replay-v1"),
        ("ablations-v1.yaml", "selection-stress-v1"),
    ],
)
def test_task5_campaign_accepts_only_declared_phase_before_readiness_gate(
    protocol_name, phase, tmp_path, capsys, monkeypatch
):
    cli = _load_run_campaign_cli()
    artifact_root = tmp_path / "must-not-be-created"

    def forbidden(*args, **kwargs):
        raise AssertionError("nonready campaign reached a method child")

    monkeypatch.setattr(cli, "run_request", forbidden)
    return_code = cli.main(
        [
            "--protocol",
            str(REPO_ROOT / "configs/protocols" / protocol_name),
            "--phase",
            phase,
            "--artifact-root",
            str(artifact_root),
            "--device",
            "cpu",
        ]
    )

    captured = capsys.readouterr()
    assert return_code == 1
    assert captured.out == ""
    assert captured.err == "campaign execution refused: execution-not-ready\n"
    assert not artifact_root.exists()


@pytest.mark.parametrize(
    ("campaign", "phase"),
    [
        (
            {
                "document_kind": "campaign",
                "campaign_id": "pilot-v1",
                "execution_ready": False,
            },
            "unknown-v1",
        ),
        (
            {
                "document_kind": "campaign",
                "campaign_id": "primary-v1",
                "execution_ready": False,
            },
            "primary-v1",
        ),
        (
            {
                "document_kind": "campaign",
                "campaign_id": "unknown-v1",
                "execution_ready": False,
            },
            "unknown-v1",
        ),
        (
            {
                "document_kind": "ablation",
                "execution_ready": False,
            },
            "pilot-v1",
        ),
    ],
)
def test_campaign_rejects_unknown_or_mismatched_phase_before_readiness_or_artifacts(
    campaign, phase, tmp_path, capsys, monkeypatch
):
    cli = _load_run_campaign_cli()
    artifact_root = tmp_path / "must-not-be-created"
    monkeypatch.setattr(cli, "load_protocol", lambda path: campaign)

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid phase reached readiness or artifact access")

    monkeypatch.setattr(cli, "_require_versioned_budget_contract", forbidden)
    monkeypatch.setattr(cli, "_run_ready_campaign", forbidden)

    return_code = cli.main(
        [
            "--protocol",
            str(tmp_path / "protocol.yaml"),
            "--phase",
            phase,
            "--artifact-root",
            str(artifact_root),
            "--device",
            "cpu",
        ]
    )

    captured = capsys.readouterr()
    assert return_code == 1
    assert captured.out == ""
    assert captured.err == "campaign execution refused: ValueError\n"
    assert not artifact_root.exists()


def test_ready_campaign_without_versioned_budget_contract_fails_before_artifacts(
    tmp_path,
    capsys,
    monkeypatch,
):
    cli = _load_run_campaign_cli()
    artifact_root = tmp_path / "must-not-be-created"
    campaign = {
        "document_kind": "campaign",
        "campaign_id": "pilot-v1",
        "execution_ready": True,
        "method_budgets": {"dgi": 1},
    }
    monkeypatch.setattr(cli, "load_protocol", lambda path: campaign)

    def forbidden(*args, **kwargs):
        raise AssertionError("unversioned budgets reached campaign execution")

    monkeypatch.setattr(cli, "_run_ready_campaign", forbidden)

    result = cli.main(
        [
            "--protocol",
            str(tmp_path / "ready.yaml"),
            "--phase",
            "pilot-v1",
            "--artifact-root",
            str(artifact_root),
            "--device",
            "cpu",
        ]
    )

    captured = capsys.readouterr()
    assert result == 1
    assert captured.out == ""
    assert captured.err == "campaign execution refused: ValueError\n"
    assert not artifact_root.exists()


def _ready_protocol_with_budgets(name: str, budgets: dict[str, int]):
    document = yaml.safe_load(
        (REPO_ROOT / "configs" / "protocols" / name).read_text(
            encoding="utf-8"
        )
    )
    ready = copy.deepcopy(document)
    ready["execution_ready"] = True
    ready["method_budgets"] = dict(budgets)
    return ready


PILOT_NATIVE_BUDGETS = {
    "dgi": 1,
    "static_cs": 1,
    "perframe_cs": 1,
    "tv3d": 1,
    "monin": 1,
    "gidc3dtv": 1,
    "recinr": 3,
    "siren": 1,
    "recinr_se2": 1,
    "gsdiff_tv": 1,
    "gsdiff_diffusion": 1,
}


PUBLICATION_NATIVE_BUDGETS = {
    "dgi": 1,
    "static_cs": 150,
    "perframe_cs": 120,
    "tv3d": 500,
    "monin": 150,
    "gidc3dtv": 2500,
    "recinr": 1900,
    "siren": 4000,
    "recinr_se2": 3000,
    "gsdiff_tv": 80,
    "gsdiff_diffusion": 80,
}


@pytest.mark.parametrize(
    ("protocol_name", "budgets"),
    [
        ("pilot-v1.yaml", PILOT_NATIVE_BUDGETS),
        ("primary-v1.yaml", PUBLICATION_NATIVE_BUDGETS),
    ],
)
def test_versioned_budget_contract_matches_resolved_native_semantics(
    protocol_name,
    budgets,
):
    cli = _load_run_campaign_cli()
    campaign = _ready_protocol_with_budgets(protocol_name, budgets)

    assert cli._require_versioned_budget_contract(campaign) is None


@pytest.mark.parametrize("mutation", ["missing", "extra", "changed"])
def test_versioned_budget_contract_rejects_nonexact_budget_maps(mutation):
    cli = _load_run_campaign_cli()
    budgets = dict(PILOT_NATIVE_BUDGETS)
    if mutation == "missing":
        budgets.pop("dgi")
    elif mutation == "extra":
        budgets["undeclared"] = 1
    else:
        budgets["recinr"] = 1
    campaign = _ready_protocol_with_budgets("pilot-v1.yaml", budgets)

    with pytest.raises((TypeError, ValueError), match="budget|method"):
        cli._require_versioned_budget_contract(campaign)


def test_cpu_smoke_profile_rejects_cuda_before_ready_campaign_execution(
    tmp_path,
    capsys,
    monkeypatch,
):
    cli = _load_run_campaign_cli()
    campaign = _ready_protocol_with_budgets(
        "pilot-v1.yaml",
        PILOT_NATIVE_BUDGETS,
    )
    artifact_root = tmp_path / "must-not-be-created"
    monkeypatch.setattr(cli, "load_protocol", lambda path: campaign)
    monkeypatch.setattr(
        cli,
        "_require_versioned_budget_contract",
        lambda protocol: None,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("CUDA smoke reached ready campaign execution")

    monkeypatch.setattr(cli, "_run_ready_campaign", forbidden)

    result = cli.main(
        [
            "--protocol",
            str(tmp_path / "pilot-v1.yaml"),
            "--phase",
            "pilot-v1",
            "--artifact-root",
            str(artifact_root),
            "--device",
            "cuda:0",
        ]
    )

    captured = capsys.readouterr()
    assert result == 1
    assert captured.out == ""
    assert captured.err == "campaign execution refused: ValueError\n"
    assert not artifact_root.exists()


def _resolved_dgi_profile(profile: str):
    return resolve_method_semantics(
        "dgi",
        method_config_id="default",
        base_config={},
        measurements_metadata={
            "H": 32,
            "W": 32,
            "T": 4,
            "K": 128,
            "holdout_K": 16,
        },
        execution_profile=profile,
    )


def test_campaign_method_policy_accepts_only_exact_cpu_pilot_smoke():
    cli = _load_run_campaign_cli()
    smoke = _resolved_dgi_profile("pilot-smoke-v1")
    campaign = {
        "campaign_id": "pilot-v1",
        "execution_profile": "pilot-smoke-v1",
    }

    assert cli._require_campaign_method_policy(
        smoke,
        phase_id="pilot-v1",
        campaign=campaign,
        requested_runtime_device="cpu",
    ) is None

    for changed in (
        {"phase_id": "ood-v1"},
        {"campaign": {**campaign, "campaign_id": "ood-v1"}},
        {"campaign": {**campaign, "execution_profile": "ood-full-v1"}},
        {"requested_runtime_device": "cuda:0"},
    ):
        arguments = {
            "phase_id": "pilot-v1",
            "campaign": campaign,
            "requested_runtime_device": "cpu",
        }
        arguments.update(changed)
        with pytest.raises(ValueError, match="promotable"):
            cli._require_campaign_method_policy(smoke, **arguments)


def test_campaign_method_policy_keeps_publication_method_promotable():
    cli = _load_run_campaign_cli()
    publication = _resolved_dgi_profile("publication-v1")

    assert cli._require_campaign_method_policy(
        publication,
        phase_id="ood-v1",
        campaign={
            "campaign_id": "ood-v1",
            "execution_profile": "ood-full-v1",
        },
        requested_runtime_device="cuda:0",
    ) is None


@pytest.mark.parametrize(
    ("script_name", "legacy_args"),
    [
        ("run_eval_matrix.py", ["--seed", "7"]),
        ("run_multiseed.py", ["--config", "configs/default.yaml"]),
        ("autoresearch.py", ["--base", "configs/default.yaml"]),
    ],
)
def test_task5_legacy_entrypoints_reject_freeform_scientific_arguments(
    script_name, legacy_args, capsys
):
    path = REPO_ROOT / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(
        f"legacy_{path.stem}", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return_code = module.main(legacy_args)

    captured = capsys.readouterr()
    assert return_code == 2
    assert "deprecated" in captured.err.lower()
    assert "--protocol" in captured.err


@pytest.mark.parametrize(
    "script_name",
    ["run_eval_matrix.py", "run_multiseed.py", "autoresearch.py"],
)
def test_legacy_script_entrypoints_refuse_nonisolated_python_before_imports(
    tmp_path: Path,
    script_name: str,
):
    marker = tmp_path / "dependency-imported"
    attacker_root = tmp_path / "attacker"
    _write_hostile_pythonpath(attacker_root, marker)

    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / script_name)],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(attacker_root)},
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == (
        "campaign execution refused: isolated-python-required\n"
    )
    assert not marker.exists()


@pytest.mark.parametrize(
    "script_name",
    ["run_eval_matrix.py", "run_multiseed.py", "autoresearch.py"],
)
def test_legacy_wrappers_launch_canonical_isolated_campaign_command(
    script_name: str,
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    path = REPO_ROOT / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(
        f"canonical_wrapper_{path.stem}", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    observed = {}

    class Completed:
        returncode = 1
        stdout = ""
        stderr = "campaign execution refused: execution-not-ready\n"

    def capture(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        return Completed()

    monkeypatch.setattr(
        module.main.__globals__["subprocess"],
        "run",
        capture,
    )
    arguments = [
        "--protocol",
        str(REPO_ROOT / "configs/protocols/pilot-v1.yaml"),
        "--phase",
        "pilot-v1",
        "--artifact-root",
        str(tmp_path / "artifacts"),
        "--device",
        "cpu",
    ]

    return_code = module.main(arguments)

    assert return_code == 1
    assert observed["command"] == [
        str(Path(sys.executable).resolve()),
        "-I",
        "-B",
        "-X",
        "utf8",
        str(RUN_CAMPAIGN_SCRIPT.resolve()),
        *arguments,
    ]
    assert observed["kwargs"] == {
        "cwd": REPO_ROOT,
        "capture_output": True,
        "text": True,
        "encoding": "utf-8",
        "check": False,
    }
    captured = capsys.readouterr()
    assert "deprecated" in captured.err.lower()
    assert "execution-not-ready" in captured.err


def test_task5_legacy_entrypoint_translates_versioned_campaign_id(
    tmp_path, capsys
):
    path = REPO_ROOT / "scripts/run_eval_matrix.py"
    spec = importlib.util.spec_from_file_location("legacy_campaign_id", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return_code = module.main(
        [
            "--campaign",
            "pilot-v1",
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--device",
            "cpu",
        ]
    )

    captured = capsys.readouterr()
    assert return_code == 1
    assert "deprecated" in captured.err.lower()
    assert "execution-not-ready" in captured.err


def test_legacy_campaign_selector_preserves_one_explicit_phase(
    tmp_path, monkeypatch
):
    path = REPO_ROOT / "scripts/run_eval_matrix.py"
    spec = importlib.util.spec_from_file_location(
        "legacy_explicit_phase", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    observed = {}

    class Completed:
        returncode = 1
        stdout = ""
        stderr = "campaign execution refused: execution-not-ready\n"

    def capture(command, **kwargs):
        observed["command"] = command
        return Completed()

    monkeypatch.setattr(module.main.__globals__["subprocess"], "run", capture)
    artifact_root = tmp_path / "artifacts"

    return_code = module.main(
        [
            "--campaign",
            "primary-v1",
            "--phase",
            "primary-confirmatory-v1",
            "--artifact-root",
            str(artifact_root),
            "--device",
            "cpu",
        ]
    )

    assert return_code == 1
    forwarded = observed["command"][6:]
    assert forwarded == [
        "--protocol",
        str(REPO_ROOT / "configs/protocols/primary-v1.yaml"),
        "--phase",
        "primary-confirmatory-v1",
        "--artifact-root",
        str(artifact_root),
        "--device",
        "cpu",
    ]


def _publish_controlled_pilot_dataset(
    cli,
    artifact_root,
    *,
    generator_commit=CONTROLLED_COMMIT,
):
    from gsdiff.data.artifacts import publish_dataset

    plan = cli.plan_campaign_datasets(
        repo_root=REPO_ROOT,
        protocol_path=(
            REPO_ROOT / "configs" / "protocols" / "pilot-v1.yaml"
        ),
        runtime=CONTROLLED_RUNTIME,
        generator_commit=generator_commit,
    )
    assert len(plan.requests) == 1
    request = plan.requests[0]
    generated = cli.generate_corrected_dataset(
        **request.generation_arguments()
    )
    publication = publish_dataset(artifact_root, generated)
    return request, publication


def _artifact_inventory(artifact_root):
    paths = (
        [artifact_root, *artifact_root.rglob("*")]
        if artifact_root.exists()
        else []
    )
    inventory = {}
    for path in paths:
        info = path.stat()
        inventory[path.relative_to(artifact_root).as_posix()] = (
            info.st_mode,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
            info.st_nlink,
            path.read_bytes() if path.is_file() else None,
        )
    return inventory


def _controlled_pilot_request_and_generated(cli):
    plan = cli.plan_campaign_datasets(
        repo_root=REPO_ROOT,
        protocol_path=(
            REPO_ROOT / "configs" / "protocols" / "pilot-v1.yaml"
        ),
        runtime=CONTROLLED_RUNTIME,
        generator_commit=CONTROLLED_COMMIT,
    )
    assert len(plan.requests) == 1
    request = plan.requests[0]
    generated = cli.generate_corrected_dataset(
        **request.generation_arguments()
    )
    return request, generated


def _coherent_semantic_twin(generated):
    from gsdiff.data.artifacts import CorrectedDataset

    record = json.loads(
        canonical_json_bytes(
            generated.noise_calibration_record
        ).decode("utf-8")
    )
    realized = record["realized_snr_db"]["train"]
    record["realized_snr_db"]["train"] = (
        123.0 if realized is None else float(realized) + 1.0
    )
    calibration_sha256 = hashlib.sha256(
        canonical_json_bytes(record)
    ).hexdigest()
    identity = json.loads(
        canonical_json_bytes(
            generated.dataset_identity_spec
        ).decode("utf-8")
    )
    identity["noise_calibration"]["sha256"] = calibration_sha256
    dataset_identity_sha256 = hashlib.sha256(
        canonical_json_bytes(identity)
    ).hexdigest()
    config = generated.resolved_generator_config
    acquisition = replace(
        generated.acquisition,
        dataset_identity_sha256=dataset_identity_sha256,
    )
    truth = replace(
        generated.truth,
        dataset_identity_sha256=dataset_identity_sha256,
        dataset_identity_spec=identity,
        evaluator_metadata={
            "resolved_generator_config": config,
            "noise_calibration_record": record,
        },
    )
    return CorrectedDataset(
        dataset_identity_sha256=dataset_identity_sha256,
        dataset_identity_spec=identity,
        resolved_generator_config=config,
        noise_calibration_record=record,
        noise_calibration_sha256=calibration_sha256,
        acquisition=acquisition,
        truth=truth,
    )


def test_task3_c3a_build_datasets_cli_exposes_main():
    assert BUILD_DATASETS_SCRIPT.is_file()
    cli = _load_build_datasets_cli()

    assert callable(cli.main)


def test_task3_c3a_root_artifacts_are_ignored_without_hiding_nested_sources():
    root_artifact = subprocess.run(
        ["git", "check-ignore", "--no-index", "--quiet", "artifacts/probe"],
        cwd=REPO_ROOT,
        check=False,
    )
    nested_source = subprocess.run(
        [
            "git",
            "check-ignore",
            "--no-index",
            "--quiet",
            "nested/artifacts/probe",
        ],
        cwd=REPO_ROOT,
        check=False,
    )

    assert root_artifact.returncode == 0
    assert nested_source.returncode == 1


def test_task3_c3a_direct_script_entrypoint_bootstraps_repo_imports(
    tmp_path,
):
    artifact_root = tmp_path / "artifacts"
    bytecode_root = tmp_path / "bytecode"
    git_index = REPO_ROOT / ".git" / "index"
    index_before = (
        git_index.read_bytes(),
        git_index.stat().st_mtime_ns,
    )
    environment = {
        **os.environ,
        "PYTHONPYCACHEPREFIX": str(bytecode_root),
    }
    environment.pop("PYTHONDONTWRITEBYTECODE", None)
    environment.pop("PYTHONPATH", None)
    environment.pop("GIT_OPTIONAL_LOCKS", None)

    result = subprocess.run(
        [
            sys.executable,
            str(BUILD_DATASETS_SCRIPT),
            "--protocol",
            "configs/protocols/pilot-v1.yaml",
            "--artifact-root",
            str(artifact_root),
            "--dry-run",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert json.loads(result.stdout)["status"] == "complete"
    assert not artifact_root.exists()
    assert not list(bytecode_root.rglob("build_datasets*.pyc"))
    assert not list(bytecode_root.rglob("_artifact_*.pyc"))
    assert (
        git_index.read_bytes(),
        git_index.stat().st_mtime_ns,
    ) == index_before
    assert str(REPO_ROOT) not in result.stdout
    assert str(tmp_path) not in result.stdout


def test_task3_c3a_bootstrap_pins_real_repo_before_hostile_pythonpath(
    tmp_path,
):
    attacker_root = tmp_path / "attacker"
    fake_package = attacker_root / "gsdiff"
    fake_package.mkdir(parents=True)
    (fake_package / "__init__.py").write_text(
        "raise RuntimeError('attacker package imported')\n",
        encoding="utf-8",
    )
    artifact_root = tmp_path / "artifacts"
    environment = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": os.pathsep.join(
            (str(attacker_root), str(REPO_ROOT))
        ),
    }

    result = subprocess.run(
        [
            sys.executable,
            str(BUILD_DATASETS_SCRIPT),
            "--protocol",
            "configs/protocols/pilot-v1.yaml",
            "--artifact-root",
            str(artifact_root),
            "--dry-run",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "attacker package imported" not in result.stderr
    assert json.loads(result.stdout)["status"] == "complete"
    assert not artifact_root.exists()


def test_task3_c3a_mode_conflict_returns_two_before_filesystem_access(
    tmp_path, capsys
):
    cli = _load_build_datasets_cli()
    missing_protocol = tmp_path / "must-not-be-read.yaml"
    artifact_root = tmp_path / "must-not-be-created"

    return_code = cli.main(
        [
            "--protocol",
            str(missing_protocol),
            "--artifact-root",
            str(artifact_root),
            "--dry-run",
            "--verify-only",
        ]
    )

    captured = capsys.readouterr()
    assert return_code == 2
    assert captured.out == ""
    assert "mutually exclusive" in captured.err
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("protocol_name", "expanded_cells", "expected_datasets"),
    [
        ("pilot-v1.yaml", 11, 1),
        ("primary-v1.yaml", 495, 45),
    ],
)
def test_task3_c3a_plan_deduplicates_by_scientific_content(
    protocol_name, expanded_cells, expected_datasets
):
    cli = _load_build_datasets_cli()

    plan = cli.plan_campaign_datasets(
        repo_root=REPO_ROOT,
        protocol_path=REPO_ROOT / "configs" / "protocols" / protocol_name,
        runtime={
            "dependencies_sha256": "1" * 64,
            "environment_lock_sha256": "2" * 64,
        },
        generator_commit="3" * 40,
    )

    assert plan.expanded_cells == expanded_cells
    assert plan.expected_datasets == expected_datasets
    assert len(plan.requests) == expected_datasets
    assert len({request.request_sha256 for request in plan.requests}) == (
        expected_datasets
    )
    assert [request.request_sha256 for request in plan.requests] == sorted(
        request.request_sha256 for request in plan.requests
    )
    for request in plan.requests:
        assert set(request.semantic_content) == {
            "schema_version",
            "scientific_contract",
            "target",
            "motion",
            "seed",
            "acquisition_config",
            "noise_calibration",
            "generator",
            "runtime",
            "resolved_generator_config",
        }
        assert "method" not in request.semantic_content
        assert "method_config_id" not in request.semantic_content
        assert "campaign_id" not in request.semantic_content
        assert "acquisition_config_id" not in request.semantic_content


def test_task3_c3a_real_pilot_dry_run_is_stable_and_writes_nothing(
    tmp_path, capsys, monkeypatch
):
    cli = _load_build_datasets_cli()
    controlled_state = {**cli._git_state(), "dirty": False}
    monkeypatch.setattr(cli, "_git_state", lambda: controlled_state)
    active_run = 0
    discovery_rechecked = [False, False]
    real_recheck = cli.verify_dataset_directory_discovery

    def observing_recheck(discovery):
        discovery_rechecked[active_run] = True
        return real_recheck(discovery)

    monkeypatch.setattr(
        cli,
        "verify_dataset_directory_discovery",
        observing_recheck,
    )
    artifact_root = tmp_path / "artifacts"
    arguments = [
        "--protocol",
        str(REPO_ROOT / "configs" / "protocols" / "pilot-v1.yaml"),
        "--artifact-root",
        str(artifact_root),
        "--dry-run",
    ]

    first_code = cli.main(arguments)
    first_capture = capsys.readouterr()
    active_run = 1
    second_code = cli.main(arguments)
    second_capture = capsys.readouterr()

    assert first_code == second_code == 0
    assert first_capture.err == second_capture.err == ""
    assert first_capture.out == second_capture.out
    assert not artifact_root.exists()
    report = json.loads(first_capture.out)
    assert first_capture.out.encode("utf-8") == (
        canonical_json_bytes(report) + b"\n"
    )
    assert report == {
        "campaign_id": "pilot-v1",
        "datasets": [
            {
                "dataset_identity_sha256": report["datasets"][0][
                    "dataset_identity_sha256"
                ],
                "request_sha256": report["datasets"][0]["request_sha256"],
                "status": "would-create",
            }
        ],
        "errors": [],
        "expanded_cells": 11,
        "expected_datasets": 1,
        "manifest_externally_anchored": False,
        "mode": "dry-run",
        "observed_datasets": 1,
        "protocol_sha256": (
            "514555268190b5cca7366eb16be61d2cc"
            "00035e042aa17bfab7f0c71d0b1a11c"
        ),
        "publishable": True,
        "rejected": [],
        "rejected_count": 0,
        "schema_version": "dataset-build-report-v1",
        "stale_staging_count": 0,
        "stale_staging": [],
        "status": "complete",
        "unmatched_datasets": [],
    }
    assert len(report["datasets"][0]["dataset_identity_sha256"]) == 64
    assert len(report["datasets"][0]["request_sha256"]) == 64
    assert discovery_rechecked == [True, True]
    assert str(REPO_ROOT) not in first_capture.out
    assert str(tmp_path) not in first_capture.out


def test_task3_c3a_dry_run_fully_verifies_existing_expected_dataset(
    tmp_path, capsys, monkeypatch
):
    cli = _load_build_datasets_cli()
    artifact_root = tmp_path / "artifacts"
    _, publication = _publish_controlled_pilot_dataset(cli, artifact_root)
    before = _artifact_inventory(artifact_root)
    monkeypatch.setattr(cli, "_environment", lambda: CONTROLLED_RUNTIME)
    monkeypatch.setattr(
        cli,
        "_git_state",
        lambda: _controlled_git_state(),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run attempted publication")

    monkeypatch.setattr(cli, "publish_dataset", forbidden)

    return_code = cli.main(
        [
            "--protocol",
            str(REPO_ROOT / "configs" / "protocols" / "pilot-v1.yaml"),
            "--artifact-root",
            str(artifact_root),
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert return_code == 0
    assert captured.err == ""
    assert report["datasets"] == [
        {
            "dataset_identity_sha256": (
                publication.verified.dataset_identity_sha256
            ),
            "request_sha256": report["datasets"][0]["request_sha256"],
            "status": "exists-valid",
        }
    ]
    assert _artifact_inventory(artifact_root) == before


def test_task3_c3a_dry_run_corrupt_existing_dataset_is_scrubbed_nonzero(
    tmp_path, capsys, monkeypatch
):
    cli = _load_build_datasets_cli()
    artifact_root = tmp_path / "artifacts"
    _, publication = _publish_controlled_pilot_dataset(cli, artifact_root)
    payload = publication.dataset_dir / "measurements.npz"
    payload.chmod(0o600)
    damaged = bytearray(payload.read_bytes())
    damaged[0] ^= 1
    payload.write_bytes(damaged)
    before = _artifact_inventory(artifact_root)
    monkeypatch.setattr(cli, "_environment", lambda: CONTROLLED_RUNTIME)
    monkeypatch.setattr(
        cli,
        "_git_state",
        lambda: _controlled_git_state(),
    )

    return_code = cli.main(
        [
            "--protocol",
            str(REPO_ROOT / "configs" / "protocols" / "pilot-v1.yaml"),
            "--artifact-root",
            str(artifact_root),
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()
    assert return_code == 1
    assert json.loads(captured.out)["errors"] == [
        {"code": "artifact-validation-error"}
    ]
    assert "Traceback" not in captured.err
    assert str(tmp_path) not in captured.out + captured.err
    assert _artifact_inventory(artifact_root) == before


def test_task3_c3d_dry_run_rejects_end_provenance_change_without_writes(
    tmp_path, capsys, monkeypatch
):
    cli = _load_build_datasets_cli()
    artifact_root = tmp_path / "secret-artifacts"
    initial_state = _controlled_git_state(dirty=True)
    changed_state = {
        **initial_state,
        "branch": "changed-during-dry-run",
    }
    states = iter((initial_state, changed_state))
    monkeypatch.setattr(cli, "_environment", lambda: CONTROLLED_RUNTIME)
    monkeypatch.setattr(cli, "_git_state", lambda: next(states))

    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run attempted publication")

    monkeypatch.setattr(cli, "publish_dataset", forbidden)

    return_code = cli.main(
        [
            "--protocol",
            str(REPO_ROOT / "configs" / "protocols" / "pilot-v1.yaml"),
            "--artifact-root",
            str(artifact_root),
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert return_code == 1
    assert report == {
        "errors": [{"code": "provenance-changed"}],
        "manifest_externally_anchored": False,
        "mode": "dry-run",
        "schema_version": "dataset-build-report-v1",
        "status": "failed",
    }
    assert captured.err == "dataset build failed: provenance-changed\n"
    assert str(REPO_ROOT) not in captured.out + captured.err
    assert str(tmp_path) not in captured.out + captured.err
    assert not artifact_root.exists()


def test_task3_c3b_verify_only_is_generation_free_stable_and_read_only(
    tmp_path, capsys, monkeypatch
):
    cli = _load_build_datasets_cli()
    artifact_root = tmp_path / "artifacts"
    request, publication = _publish_controlled_pilot_dataset(
        cli,
        artifact_root,
    )
    _, old_publication = _publish_controlled_pilot_dataset(
        cli,
        artifact_root,
        generator_commit="5" * 40,
    )
    datasets_dir = artifact_root / "datasets"
    (datasets_dir / f".{'a' * 64}.staging-crashleft").mkdir()
    (datasets_dir / f".{'a' * 64}.rejected-{'b' * 24}").mkdir()
    before = _artifact_inventory(artifact_root)
    monkeypatch.setattr(cli, "_environment", lambda: CONTROLLED_RUNTIME)
    monkeypatch.setattr(
        cli,
        "_git_state",
        lambda: _controlled_git_state(),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("verify-only attempted generation or publication")

    monkeypatch.setattr(cli, "generate_corrected_dataset", forbidden)
    monkeypatch.setattr(cli, "publish_dataset", forbidden, raising=False)
    arguments = [
        "--protocol",
        str(REPO_ROOT / "configs" / "protocols" / "pilot-v1.yaml"),
        "--artifact-root",
        str(artifact_root),
        "--verify-only",
    ]

    first_code = cli.main(arguments)
    first_capture = capsys.readouterr()
    second_code = cli.main(arguments)
    second_capture = capsys.readouterr()

    assert first_code == second_code == 0
    assert first_capture.err == second_capture.err == ""
    assert first_capture.out == second_capture.out
    report = json.loads(first_capture.out)
    assert first_capture.out.encode("utf-8") == (
        canonical_json_bytes(report) + b"\n"
    )
    assert report == {
        "campaign_id": "pilot-v1",
        "datasets": [
            {
                "dataset_identity_sha256": (
                    publication.verified.dataset_identity_sha256
                ),
                "request_sha256": request.request_sha256,
                "status": "verified",
            }
        ],
        "errors": [],
        "expanded_cells": 11,
        "expected_datasets": 1,
        "manifest_externally_anchored": False,
        "mode": "verify-only",
        "observed_datasets": 1,
        "protocol_sha256": (
            "514555268190b5cca7366eb16be61d2cc"
            "00035e042aa17bfab7f0c71d0b1a11c"
        ),
        "publishable": True,
        "rejected": [
            {
                "count": 1,
                "dataset_identity_sha256": "a" * 64,
            }
        ],
        "rejected_count": 1,
        "schema_version": "dataset-build-report-v1",
        "stale_staging_count": 1,
        "stale_staging": [
            {
                "count": 1,
                "dataset_identity_sha256": "a" * 64,
            }
        ],
        "status": "complete",
        "unmatched_datasets": [
            old_publication.verified.dataset_identity_sha256
        ],
    }
    after = _artifact_inventory(artifact_root)
    assert after == before
    assert str(REPO_ROOT) not in first_capture.out
    assert str(tmp_path) not in first_capture.out


def test_task3_c3d_verify_only_rejects_end_provenance_change_without_writes(
    tmp_path, capsys, monkeypatch
):
    cli = _load_build_datasets_cli()
    artifact_root = tmp_path / "secret-artifacts"
    _publish_controlled_pilot_dataset(cli, artifact_root)
    before = _artifact_inventory(artifact_root)
    initial_state = _controlled_git_state(dirty=True)
    changed_state = {
        **initial_state,
        "branch": "changed-during-verify",
    }
    states = iter((initial_state, changed_state))
    monkeypatch.setattr(cli, "_environment", lambda: CONTROLLED_RUNTIME)
    monkeypatch.setattr(cli, "_git_state", lambda: next(states))

    def forbidden(*args, **kwargs):
        raise AssertionError("verify-only attempted generation or publication")

    monkeypatch.setattr(cli, "generate_corrected_dataset", forbidden)
    monkeypatch.setattr(cli, "publish_dataset", forbidden)

    return_code = cli.main(
        [
            "--protocol",
            str(REPO_ROOT / "configs" / "protocols" / "pilot-v1.yaml"),
            "--artifact-root",
            str(artifact_root),
            "--verify-only",
        ]
    )

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert return_code == 1
    assert report == {
        "errors": [{"code": "provenance-changed"}],
        "manifest_externally_anchored": False,
        "mode": "verify-only",
        "schema_version": "dataset-build-report-v1",
        "status": "failed",
    }
    assert captured.err == "dataset build failed: provenance-changed\n"
    assert str(REPO_ROOT) not in captured.out + captured.err
    assert str(tmp_path) not in captured.out + captured.err
    assert _artifact_inventory(artifact_root) == before


def test_task3_c3b_semantic_selector_uses_exact_canonical_projection(
    tmp_path,
):
    cli = _load_build_datasets_cli()
    request, publication = _publish_controlled_pilot_dataset(
        cli,
        tmp_path / "artifacts",
    )
    manifest = publication.verified.manifest
    request_key = cli._semantic_projection_from_request(request)

    assert cli._semantic_projection_from_manifest(manifest) == request_key

    numeric_twin = json.loads(json.dumps(manifest))
    numeric_twin["resolved_generator_config"]["acquisition"]["snr_db"] = (
        25.0
    )
    assert numeric_twin == manifest
    assert (
        cli._semantic_projection_from_manifest(numeric_twin)
        != request_key
    )

    semantic_with_untrusted_entry_text = json.loads(
        canonical_json_bytes(request.semantic_content).decode("utf-8")
    )
    semantic_with_untrusted_entry_text["noise_calibration"]["entry"][
        "mode"
    ] = "not-manifest-evidence"
    altered_request = replace(
        request,
        semantic_content=semantic_with_untrusted_entry_text,
    )
    assert (
        cli._semantic_projection_from_request(altered_request)
        == request_key
    )

    wrong_registry_hash = json.loads(json.dumps(manifest))
    wrong_registry_hash["noise_calibration_record"]["calibration"][
        "registry_entry_sha256"
    ] = "f" * 64
    assert (
        cli._semantic_projection_from_manifest(wrong_registry_hash)
        != request_key
    )

    mutations = {}
    mutations["scientific_contract"] = json.loads(json.dumps(manifest))
    mutations["scientific_contract"]["dataset_identity_spec"][
        "scientific_contract"
    ]["id"] = "other-contract"
    mutations["target"] = json.loads(json.dumps(manifest))
    mutations["target"]["resolved_generator_config"]["target"][
        "descriptor"
    ] = "char:R"
    mutations["motion"] = json.loads(json.dumps(manifest))
    mutations["motion"]["resolved_generator_config"]["motion"][
        "velocity"
    ] = [9, 8]
    mutations["seed"] = json.loads(json.dumps(manifest))
    mutations["seed"]["dataset_identity_spec"]["seed"] = 11
    mutations["dimensions"] = json.loads(json.dumps(manifest))
    mutations["dimensions"]["resolved_generator_config"]["dimensions"][
        "T"
    ] = 5
    mutations["acquisition"] = json.loads(json.dumps(manifest))
    mutations["acquisition"]["resolved_generator_config"]["acquisition"][
        "pattern_order"
    ] = "other-order"
    mutations["calibration"] = json.loads(json.dumps(manifest))
    mutations["calibration"]["noise_calibration_record"]["calibration"][
        "id"
    ] = "other-calibration"
    mutations["generator"] = json.loads(json.dumps(manifest))
    mutations["generator"]["dataset_identity_spec"]["generator"][
        "version"
    ] = "generator-v2"
    mutations["runtime"] = json.loads(json.dumps(manifest))
    mutations["runtime"]["dataset_identity_spec"]["runtime"][
        "dependencies_sha256"
    ] = "e" * 64
    mutations["resolved_config"] = json.loads(json.dumps(manifest))
    mutations["resolved_config"]["resolved_generator_config"]["rng"][
        "bit_generator"
    ] = "other-rng"
    for field, changed_manifest in mutations.items():
        assert (
            cli._semantic_projection_from_manifest(changed_manifest)
            != request_key
        ), field


def test_task3_c3b_verify_only_missing_is_nonzero_and_writes_nothing(
    tmp_path, capsys, monkeypatch
):
    cli = _load_build_datasets_cli()
    artifact_root = tmp_path / "artifacts"
    monkeypatch.setattr(cli, "_environment", lambda: CONTROLLED_RUNTIME)
    monkeypatch.setattr(
        cli,
        "_git_state",
        lambda: _controlled_git_state(),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("verify-only attempted generation")

    monkeypatch.setattr(cli, "generate_corrected_dataset", forbidden)
    monkeypatch.setattr(cli, "publish_dataset", forbidden, raising=False)

    return_code = cli.main(
        [
            "--protocol",
            str(REPO_ROOT / "configs" / "protocols" / "pilot-v1.yaml"),
            "--artifact-root",
            str(artifact_root),
            "--verify-only",
        ]
    )

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert return_code == 1
    assert captured.err == "dataset verification failed\n"
    assert report["status"] == "failed"
    assert report["datasets"] == []
    assert report["unmatched_datasets"] == []
    assert report["stale_staging"] == []
    assert report["rejected"] == []
    assert report["errors"] == [
        {
            "code": "missing-current-dataset",
            "request_sha256": report["errors"][0]["request_sha256"],
        }
    ]
    assert len(report["errors"][0]["request_sha256"]) == 64
    assert report["manifest_externally_anchored"] is False
    assert not artifact_root.exists()
    assert str(REPO_ROOT) not in captured.out
    assert str(tmp_path) not in captured.out


def test_task3_c3b_verify_only_reports_corrupt_current_dataset(
    tmp_path, capsys, monkeypatch
):
    cli = _load_build_datasets_cli()
    artifact_root = tmp_path / "artifacts"
    request, publication = _publish_controlled_pilot_dataset(
        cli,
        artifact_root,
    )
    payload = publication.dataset_dir / "measurements.npz"
    payload.chmod(0o600)
    damaged = bytearray(payload.read_bytes())
    damaged[-1] ^= 1
    payload.write_bytes(damaged)
    monkeypatch.setattr(cli, "_environment", lambda: CONTROLLED_RUNTIME)
    monkeypatch.setattr(
        cli,
        "_git_state",
        lambda: _controlled_git_state(),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("verify-only attempted generation or publication")

    monkeypatch.setattr(cli, "generate_corrected_dataset", forbidden)
    monkeypatch.setattr(cli, "publish_dataset", forbidden)

    return_code = cli.main(
        [
            "--protocol",
            str(REPO_ROOT / "configs" / "protocols" / "pilot-v1.yaml"),
            "--artifact-root",
            str(artifact_root),
            "--verify-only",
        ]
    )

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert return_code == 1
    assert captured.err == "dataset verification failed\n"
    assert report["datasets"] == []
    assert report["errors"] == [
        {
            "code": "corrupt-dataset",
            "dataset_identity_sha256": (
                publication.verified.dataset_identity_sha256
            ),
        },
        {
            "code": "missing-current-dataset",
            "request_sha256": request.request_sha256,
        },
    ]


def test_task3_c3b_verify_only_reports_ambiguous_semantic_match(
    tmp_path, capsys, monkeypatch
):
    from types import SimpleNamespace

    cli = _load_build_datasets_cli()
    artifact_root = tmp_path / "artifacts"
    request, publication = _publish_controlled_pilot_dataset(
        cli,
        artifact_root,
    )
    second_identity = "e" * 64
    (artifact_root / "datasets" / second_identity).mkdir()
    compact_verified = SimpleNamespace(
        manifest=publication.verified.manifest,
        manifest_externally_anchored=False,
        expected_generated_verified=False,
    )
    monkeypatch.setattr(cli, "_environment", lambda: CONTROLLED_RUNTIME)
    monkeypatch.setattr(
        cli,
        "_git_state",
        lambda: _controlled_git_state(),
    )
    monkeypatch.setattr(
        cli,
        "verify_dataset_directory",
        lambda *args, **kwargs: compact_verified,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("verify-only attempted generation or publication")

    monkeypatch.setattr(cli, "generate_corrected_dataset", forbidden)
    monkeypatch.setattr(cli, "publish_dataset", forbidden)

    return_code = cli.main(
        [
            "--protocol",
            str(REPO_ROOT / "configs" / "protocols" / "pilot-v1.yaml"),
            "--artifact-root",
            str(artifact_root),
            "--verify-only",
        ]
    )

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    expected_identities = sorted(
        (
            publication.verified.dataset_identity_sha256,
            second_identity,
        )
    )
    assert return_code == 1
    assert captured.err == "dataset verification failed\n"
    assert report["datasets"] == []
    assert report["errors"] == [
        {
            "code": "ambiguous-current-dataset",
            "dataset_identity_sha256": expected_identities,
            "request_sha256": request.request_sha256,
        }
    ]


def test_task3_c3b_verify_only_discovery_race_is_scrubbed_and_nonzero(
    tmp_path, capsys, monkeypatch
):
    from gsdiff.data.artifacts import ArtifactValidationError

    cli = _load_build_datasets_cli()
    artifact_root = tmp_path / "artifacts"
    _publish_controlled_pilot_dataset(cli, artifact_root)
    monkeypatch.setattr(cli, "_environment", lambda: CONTROLLED_RUNTIME)
    monkeypatch.setattr(
        cli,
        "_git_state",
        lambda: _controlled_git_state(),
    )

    def race(*args, **kwargs):
        raise ArtifactValidationError("secret discovery race path")

    monkeypatch.setattr(
        cli,
        "verify_dataset_directory_discovery",
        race,
    )

    return_code = cli.main(
        [
            "--protocol",
            str(REPO_ROOT / "configs" / "protocols" / "pilot-v1.yaml"),
            "--artifact-root",
            str(artifact_root),
            "--verify-only",
        ]
    )

    captured = capsys.readouterr()
    assert return_code == 1
    assert json.loads(captured.out)["errors"] == [
        {"code": "artifact-validation-error"}
    ]
    assert "secret discovery race path" not in captured.out + captured.err


@pytest.mark.parametrize("mode", ["--dry-run", "--verify-only"])
def test_task3_c3b_operational_discovery_failure_is_scrubbed(
    tmp_path, capsys, monkeypatch, mode
):
    cli = _load_build_datasets_cli()
    artifact_root = tmp_path / "secret-artifacts"
    datasets_dir = artifact_root / "datasets"
    datasets_dir.mkdir(parents=True)
    (datasets_dir / "unexpected-secret.txt").write_bytes(b"unexpected")
    before = _artifact_inventory(artifact_root)
    monkeypatch.setattr(cli, "_environment", lambda: CONTROLLED_RUNTIME)
    monkeypatch.setattr(
        cli,
        "_git_state",
        lambda: _controlled_git_state(),
    )

    return_code = cli.main(
        [
            "--protocol",
            str(REPO_ROOT / "configs" / "protocols" / "pilot-v1.yaml"),
            "--artifact-root",
            str(artifact_root),
            mode,
        ]
    )

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert return_code == 1
    assert captured.out.encode("utf-8") == (
        canonical_json_bytes(report) + b"\n"
    )
    assert report == {
        "errors": [{"code": "artifact-validation-error"}],
        "manifest_externally_anchored": False,
        "mode": mode.removeprefix("--"),
        "schema_version": "dataset-build-report-v1",
        "status": "failed",
    }
    assert captured.err == "dataset build failed: artifact-validation-error\n"
    assert "Traceback" not in captured.err
    assert str(REPO_ROOT) not in captured.out + captured.err
    assert str(tmp_path) not in captured.out + captured.err
    assert _artifact_inventory(artifact_root) == before


def test_task3_c3b_direct_verify_failure_has_no_traceback_or_path_leak(
    tmp_path,
):
    artifact_root = tmp_path / "secret-artifacts"
    datasets_dir = artifact_root / "datasets"
    datasets_dir.mkdir(parents=True)
    (datasets_dir / "unexpected-secret.txt").write_bytes(b"unexpected")
    before = _artifact_inventory(artifact_root)
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONDONTWRITEBYTECODE", None)

    result = subprocess.run(
        [
            sys.executable,
            str(BUILD_DATASETS_SCRIPT),
            "--protocol",
            "configs/protocols/pilot-v1.yaml",
            "--artifact-root",
            str(artifact_root),
            "--verify-only",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    report = json.loads(result.stdout)
    assert result.returncode == 1
    assert report["errors"] == [{"code": "artifact-validation-error"}]
    assert "Traceback" not in result.stderr
    assert str(REPO_ROOT) not in result.stdout + result.stderr
    assert str(tmp_path) not in result.stdout + result.stderr
    assert _artifact_inventory(artifact_root) == before


def test_task3_c3c_dirty_build_fails_before_generation_or_artifact_creation(
    tmp_path, capsys, monkeypatch
):
    cli = _load_build_datasets_cli()
    artifact_root = tmp_path / "artifacts"
    monkeypatch.setattr(cli, "_environment", lambda: CONTROLLED_RUNTIME)
    monkeypatch.setattr(
        cli,
        "_git_state",
        lambda: _controlled_git_state(dirty=True),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("dirty build attempted generation or publication")

    monkeypatch.setattr(cli, "generate_corrected_dataset", forbidden)
    monkeypatch.setattr(cli, "publish_dataset", forbidden, raising=False)
    monkeypatch.setattr(cli, "plan_campaign_datasets", forbidden)
    monkeypatch.setattr(cli, "discover_dataset_directories", forbidden)

    return_code = cli.main(
        [
            "--protocol",
            str(REPO_ROOT / "configs" / "protocols" / "pilot-v1.yaml"),
            "--artifact-root",
            str(artifact_root),
        ]
    )

    captured = capsys.readouterr()
    assert return_code == 1
    assert json.loads(captured.out)["errors"] == [
        {"code": "dirty-worktree"}
    ]
    assert captured.err == "dataset build failed: dirty-worktree\n"
    assert not artifact_root.exists()


def test_task3_c3c_provenance_change_after_generation_prevents_publication(
    tmp_path, capsys, monkeypatch
):
    cli = _load_build_datasets_cli()
    artifact_root = tmp_path / "artifacts"
    states = iter(
        (
            _controlled_git_state(),
            _controlled_git_state(),
            _controlled_git_state(dirty=True),
        )
    )
    monkeypatch.setattr(cli, "_environment", lambda: CONTROLLED_RUNTIME)
    monkeypatch.setattr(cli, "_git_state", lambda: next(states))

    def forbidden(*args, **kwargs):
        raise AssertionError("changed provenance attempted publication")

    monkeypatch.setattr(cli, "publish_dataset", forbidden, raising=False)

    return_code = cli.main(
        [
            "--protocol",
            str(REPO_ROOT / "configs" / "protocols" / "pilot-v1.yaml"),
            "--artifact-root",
            str(artifact_root),
        ]
    )

    captured = capsys.readouterr()
    assert return_code == 1
    assert json.loads(captured.out)["errors"] == [
        {"code": "provenance-changed"}
    ]
    assert captured.err == "dataset build failed: provenance-changed\n"
    assert not artifact_root.exists()


def test_task3_c3c_final_discovery_is_followed_by_provenance_recheck(
    tmp_path, capsys, monkeypatch
):
    cli = _load_build_datasets_cli()
    artifact_root = tmp_path / "artifacts"
    final_discovery_seen = False
    real_discovery = cli.discover_dataset_directories

    def observing_discovery(path):
        nonlocal final_discovery_seen
        result = real_discovery(path)
        if result.canonical_directories:
            final_discovery_seen = True
        return result

    def changing_git_state():
        return _controlled_git_state(dirty=final_discovery_seen)

    monkeypatch.setattr(cli, "_environment", lambda: CONTROLLED_RUNTIME)
    monkeypatch.setattr(cli, "_git_state", changing_git_state)
    monkeypatch.setattr(
        cli,
        "discover_dataset_directories",
        observing_discovery,
    )

    return_code = cli.main(
        [
            "--protocol",
            str(REPO_ROOT / "configs" / "protocols" / "pilot-v1.yaml"),
            "--artifact-root",
            str(artifact_root),
        ]
    )

    captured = capsys.readouterr()
    assert final_discovery_seen is True
    assert return_code == 1
    assert json.loads(captured.out)["errors"] == [
        {"code": "provenance-changed"}
    ]
    assert captured.err == "dataset build failed: provenance-changed\n"


def test_task3_c3c_concurrent_winner_during_generation_is_safely_reused(
    tmp_path, capsys, monkeypatch
):
    from gsdiff.data.artifacts import publish_dataset as real_publish

    cli = _load_build_datasets_cli()
    artifact_root = tmp_path / "artifacts"
    real_generate = cli.generate_corrected_dataset
    winner_identity = None
    loser_publications = 0

    def generate_after_concurrent_winner(**kwargs):
        nonlocal winner_identity
        generated = real_generate(**kwargs)
        winner = real_publish(artifact_root, generated)
        winner_identity = winner.verified.dataset_identity_sha256
        return generated

    def observing_publish(root, generated):
        nonlocal loser_publications
        loser_publications += 1
        return real_publish(root, generated)

    monkeypatch.setattr(cli, "_environment", lambda: CONTROLLED_RUNTIME)
    monkeypatch.setattr(
        cli,
        "_git_state",
        lambda: _controlled_git_state(),
    )
    monkeypatch.setattr(
        cli,
        "generate_corrected_dataset",
        generate_after_concurrent_winner,
    )
    monkeypatch.setattr(cli, "publish_dataset", observing_publish)

    return_code = cli.main(
        [
            "--protocol",
            str(REPO_ROOT / "configs" / "protocols" / "pilot-v1.yaml"),
            "--artifact-root",
            str(artifact_root),
        ]
    )

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert return_code == 0
    assert captured.err == ""
    assert loser_publications == 1
    assert report["datasets"] == [
        {
            "dataset_identity_sha256": winner_identity,
            "request_sha256": report["datasets"][0]["request_sha256"],
            "status": "reused",
        }
    ]
    assert {
        path.name
        for path in (artifact_root / "datasets").iterdir()
    } == {winner_identity}


def test_task3_c3c_winner_at_publisher_boundary_is_safely_reused(
    tmp_path, capsys, monkeypatch
):
    from gsdiff.data.artifacts import publish_dataset as real_publish

    cli = _load_build_datasets_cli()
    artifact_root = tmp_path / "artifacts"

    def publish_after_winner(root, generated):
        winner = real_publish(root, generated)
        assert winner.status == "created"
        return real_publish(root, generated)

    monkeypatch.setattr(cli, "_environment", lambda: CONTROLLED_RUNTIME)
    monkeypatch.setattr(
        cli,
        "_git_state",
        lambda: _controlled_git_state(),
    )
    monkeypatch.setattr(cli, "publish_dataset", publish_after_winner)

    return_code = cli.main(
        [
            "--protocol",
            str(REPO_ROOT / "configs" / "protocols" / "pilot-v1.yaml"),
            "--artifact-root",
            str(artifact_root),
        ]
    )

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert return_code == 0
    assert captured.err == ""
    assert report["datasets"][0]["status"] == "reused"


@pytest.mark.parametrize("mutation", ["add", "remove"])
def test_task3_c3c_unrelated_staging_race_does_not_block_publication(
    tmp_path, capsys, monkeypatch, mutation
):
    cli = _load_build_datasets_cli()
    artifact_root = tmp_path / "artifacts"
    datasets_dir = artifact_root / "datasets"
    staging = datasets_dir / f".{'a' * 64}.staging-race"
    if mutation == "remove":
        staging.mkdir(parents=True)

    real_publish = cli.publish_dataset

    def publish_after_staging_race(root, generated):
        if mutation == "add":
            staging.mkdir(parents=True)
        else:
            staging.rmdir()
        return real_publish(root, generated)

    monkeypatch.setattr(cli, "_environment", lambda: CONTROLLED_RUNTIME)
    monkeypatch.setattr(
        cli,
        "_git_state",
        lambda: _controlled_git_state(),
    )
    monkeypatch.setattr(cli, "publish_dataset", publish_after_staging_race)

    return_code = cli.main(
        [
            "--protocol",
            str(REPO_ROOT / "configs" / "protocols" / "pilot-v1.yaml"),
            "--artifact-root",
            str(artifact_root),
        ]
    )

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert return_code == 0
    assert captured.err == ""
    assert report["datasets"][0]["status"] == "created"
    assert report["stale_staging_count"] == (
        1 if mutation == "add" else 0
    )


def test_task3_c3c_default_build_creates_then_reuses_without_byte_changes(
    tmp_path, capsys, monkeypatch
):
    cli = _load_build_datasets_cli()
    artifact_root = tmp_path / "artifacts"
    git_checks = 0

    def stable_git_state():
        nonlocal git_checks
        git_checks += 1
        return _controlled_git_state()

    monkeypatch.setattr(cli, "_environment", lambda: CONTROLLED_RUNTIME)
    monkeypatch.setattr(cli, "_git_state", stable_git_state)
    arguments = [
        "--protocol",
        str(REPO_ROOT / "configs" / "protocols" / "pilot-v1.yaml"),
        "--artifact-root",
        str(artifact_root),
    ]

    first_code = cli.main(arguments)
    first_capture = capsys.readouterr()
    first_inventory = _artifact_inventory(artifact_root)
    second_code = cli.main(arguments)
    second_capture = capsys.readouterr()
    second_inventory = _artifact_inventory(artifact_root)

    assert first_code == second_code == 0
    assert first_capture.err == second_capture.err == ""
    first_report = json.loads(first_capture.out)
    second_report = json.loads(second_capture.out)
    assert first_report["status"] == second_report["status"] == "complete"
    assert first_report["mode"] == second_report["mode"] == "build"
    assert first_report["datasets"][0]["status"] == "created"
    assert second_report["datasets"][0]["status"] == "reused"
    assert first_report["datasets"][0]["dataset_identity_sha256"] == (
        second_report["datasets"][0]["dataset_identity_sha256"]
    )
    assert first_report["datasets"][0]["request_sha256"] == (
        second_report["datasets"][0]["request_sha256"]
    )
    assert first_report["manifest_externally_anchored"] is False
    assert second_report["manifest_externally_anchored"] is False
    assert first_report["stale_staging"] == second_report[
        "stale_staging"
    ] == []
    assert first_report["rejected"] == second_report["rejected"] == []
    assert first_inventory == second_inventory
    assert git_checks >= 8
    assert first_capture.out.encode("utf-8") == (
        canonical_json_bytes(first_report) + b"\n"
    )
    assert second_capture.out.encode("utf-8") == (
        canonical_json_bytes(second_report) + b"\n"
    )
    assert str(REPO_ROOT) not in first_capture.out + second_capture.out
    assert str(tmp_path) not in first_capture.out + second_capture.out


def test_task3_round1_build_reuses_unique_current_without_generation_or_publish(
    tmp_path, capsys, monkeypatch
):
    cli = _load_build_datasets_cli()
    artifact_root = tmp_path / "artifacts"
    request, publication = _publish_controlled_pilot_dataset(
        cli, artifact_root
    )
    before = _artifact_inventory(artifact_root)
    monkeypatch.setattr(cli, "_environment", lambda: CONTROLLED_RUNTIME)
    monkeypatch.setattr(
        cli, "_git_state", lambda: _controlled_git_state()
    )

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "unique current reuse attempted generation or publication"
        )

    monkeypatch.setattr(cli, "generate_corrected_dataset", forbidden)
    monkeypatch.setattr(cli, "publish_dataset", forbidden)

    return_code = cli.main(
        [
            "--protocol",
            str(REPO_ROOT / "configs" / "protocols" / "pilot-v1.yaml"),
            "--artifact-root",
            str(artifact_root),
        ]
    )

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert return_code == 0
    assert captured.err == ""
    assert report["datasets"] == [
        {
            "dataset_identity_sha256": (
                publication.verified.dataset_identity_sha256
            ),
            "request_sha256": request.request_sha256,
            "status": "reused",
        }
    ]
    assert report["corrupt_count"] == 0
    assert report["corrupt_datasets"] == []
    assert _artifact_inventory(artifact_root) == before


def test_task3_round1_default_build_reuses_existing_and_creates_only_missing(
    tmp_path, capsys, monkeypatch
):
    cli = _load_build_datasets_cli()
    artifact_root = tmp_path / "artifacts"
    base_request, base_publication = _publish_controlled_pilot_dataset(
        cli, artifact_root
    )
    base_identity = base_publication.verified.dataset_identity_sha256
    base_before = _artifact_inventory(base_publication.dataset_dir)

    missing_seed = base_request.seed + 1
    missing_arguments = base_request.generation_arguments()
    missing_arguments["seed"] = missing_seed
    missing_semantic_content = cli.resolve_corrected_dataset_request(
        **missing_arguments
    )
    missing_encoded = canonical_json_bytes(missing_semantic_content)
    missing_request = replace(
        base_request,
        request_sha256=hashlib.sha256(missing_encoded).hexdigest(),
        semantic_content=missing_semantic_content,
        seed=missing_seed,
    )
    base_plan = cli.plan_campaign_datasets(
        repo_root=REPO_ROOT,
        protocol_path=(
            REPO_ROOT / "configs" / "protocols" / "pilot-v1.yaml"
        ),
        runtime=CONTROLLED_RUNTIME,
        generator_commit=CONTROLLED_COMMIT,
    )
    mixed_plan = replace(
        base_plan,
        expanded_cells=2,
        expected_datasets=2,
        requests=tuple(
            sorted(
                (base_request, missing_request),
                key=lambda request: request.request_sha256,
            )
        ),
    )

    real_generate = cli.generate_corrected_dataset
    real_publish = cli.publish_dataset
    generated_calls = []
    published_calls = []

    def observing_generate(**kwargs):
        generated = real_generate(**kwargs)
        generated_calls.append(
            (kwargs["seed"], generated.dataset_identity_sha256)
        )
        return generated

    def observing_publish(root, generated):
        publication = real_publish(root, generated)
        published_calls.append(
            (
                generated.dataset_identity_sha256,
                publication.status,
            )
        )
        return publication

    monkeypatch.setattr(cli, "_environment", lambda: CONTROLLED_RUNTIME)
    monkeypatch.setattr(
        cli, "_git_state", lambda: _controlled_git_state()
    )
    monkeypatch.setattr(
        cli, "plan_campaign_datasets", lambda **kwargs: mixed_plan
    )
    monkeypatch.setattr(
        cli, "generate_corrected_dataset", observing_generate
    )
    monkeypatch.setattr(cli, "publish_dataset", observing_publish)

    return_code = cli.main(
        [
            "--protocol",
            str(REPO_ROOT / "configs" / "protocols" / "pilot-v1.yaml"),
            "--artifact-root",
            str(artifact_root),
        ]
    )

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert return_code == 0
    assert captured.err == ""
    assert len(generated_calls) == len(published_calls) == 1
    missing_identity = generated_calls[0][1]
    assert generated_calls == [(missing_seed, missing_identity)]
    assert published_calls == [(missing_identity, "created")]
    assert {
        record["request_sha256"]: (
            record["dataset_identity_sha256"],
            record["status"],
        )
        for record in report["datasets"]
    } == {
        base_request.request_sha256: (base_identity, "reused"),
        missing_request.request_sha256: (missing_identity, "created"),
    }
    assert report["observed_datasets"] == report["expected_datasets"] == 2
    assert {
        path.name
        for path in (artifact_root / "datasets").iterdir()
    } == {base_identity, missing_identity}
    assert _artifact_inventory(base_publication.dataset_dir) == base_before


def test_task3_round1_build_rejects_real_coherent_twins_before_any_write(
    tmp_path, capsys, monkeypatch
):
    from gsdiff.data.artifacts import publish_dataset

    cli = _load_build_datasets_cli()
    artifact_root = tmp_path / "artifacts"
    request, generated = _controlled_pilot_request_and_generated(cli)
    first = publish_dataset(artifact_root, generated)
    twin = _coherent_semantic_twin(generated)
    second = publish_dataset(artifact_root, twin)
    assert first.verified.dataset_identity_sha256 != (
        second.verified.dataset_identity_sha256
    )
    assert cli._semantic_projection_from_manifest(
        first.verified.manifest
    ) == cli._semantic_projection_from_manifest(second.verified.manifest)
    before = _artifact_inventory(artifact_root)
    monkeypatch.setattr(cli, "_environment", lambda: CONTROLLED_RUNTIME)
    monkeypatch.setattr(
        cli, "_git_state", lambda: _controlled_git_state()
    )

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "ambiguous current datasets attempted generation or publication"
        )

    monkeypatch.setattr(cli, "generate_corrected_dataset", forbidden)
    monkeypatch.setattr(cli, "publish_dataset", forbidden)

    return_code = cli.main(
        [
            "--protocol",
            str(REPO_ROOT / "configs" / "protocols" / "pilot-v1.yaml"),
            "--artifact-root",
            str(artifact_root),
        ]
    )

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert return_code == 1
    assert report["errors"] == [{"code": "ambiguous-current-dataset"}]
    assert captured.err == (
        "dataset build failed: ambiguous-current-dataset\n"
    )
    assert _artifact_inventory(artifact_root) == before
    assert len(request.request_sha256) == 64


def test_task3_round1_build_reuses_valid_current_and_reports_corrupt_unrelated(
    tmp_path, capsys, monkeypatch
):
    cli = _load_build_datasets_cli()
    artifact_root = tmp_path / "artifacts"
    request, publication = _publish_controlled_pilot_dataset(
        cli, artifact_root
    )
    corrupt_identity = "e" * 64
    corrupt_dir = (
        artifact_root / "datasets" / corrupt_identity
    )
    shutil.copytree(publication.dataset_dir, corrupt_dir)
    before = _artifact_inventory(artifact_root)
    monkeypatch.setattr(cli, "_environment", lambda: CONTROLLED_RUNTIME)
    monkeypatch.setattr(
        cli, "_git_state", lambda: _controlled_git_state()
    )

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "valid current plus corrupt unrelated attempted generation"
        )

    monkeypatch.setattr(cli, "generate_corrected_dataset", forbidden)
    monkeypatch.setattr(cli, "publish_dataset", forbidden)

    return_code = cli.main(
        [
            "--protocol",
            str(REPO_ROOT / "configs" / "protocols" / "pilot-v1.yaml"),
            "--artifact-root",
            str(artifact_root),
        ]
    )

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert return_code == 0
    assert captured.err == ""
    assert report["datasets"] == [
        {
            "dataset_identity_sha256": (
                publication.verified.dataset_identity_sha256
            ),
            "request_sha256": request.request_sha256,
            "status": "reused",
        }
    ]
    assert report["corrupt_count"] == 1
    assert report["corrupt_datasets"] == [
        {"dataset_identity_sha256": corrupt_identity}
    ]
    assert _artifact_inventory(artifact_root) == before


def test_task3_round1_build_does_not_repair_corrupt_exact_final(
    tmp_path, capsys, monkeypatch
):
    cli = _load_build_datasets_cli()
    artifact_root = tmp_path / "artifacts"
    _, publication = _publish_controlled_pilot_dataset(
        cli, artifact_root
    )
    payload = publication.dataset_dir / "measurements.npz"
    payload.chmod(0o600)
    damaged = bytearray(payload.read_bytes())
    damaged[-1] ^= 1
    payload.write_bytes(damaged)
    before = _artifact_inventory(artifact_root)
    generated_calls = 0
    publish_calls = 0
    real_generate = cli.generate_corrected_dataset
    real_publish = cli.publish_dataset

    def counting_generate(**kwargs):
        nonlocal generated_calls
        generated_calls += 1
        return real_generate(**kwargs)

    def counting_publish(root, generated):
        nonlocal publish_calls
        publish_calls += 1
        return real_publish(root, generated)

    monkeypatch.setattr(cli, "_environment", lambda: CONTROLLED_RUNTIME)
    monkeypatch.setattr(
        cli, "_git_state", lambda: _controlled_git_state()
    )
    monkeypatch.setattr(
        cli, "generate_corrected_dataset", counting_generate
    )
    monkeypatch.setattr(cli, "publish_dataset", counting_publish)

    return_code = cli.main(
        [
            "--protocol",
            str(REPO_ROOT / "configs" / "protocols" / "pilot-v1.yaml"),
            "--artifact-root",
            str(artifact_root),
        ]
    )

    captured = capsys.readouterr()
    assert return_code == 1
    assert json.loads(captured.out)["errors"] == [
        {"code": "artifact-validation-error"}
    ]
    assert generated_calls == publish_calls == 1
    assert _artifact_inventory(artifact_root) == before


def test_task3_round1_build_handles_winner_after_initial_scan_in_publisher(
    tmp_path, capsys, monkeypatch
):
    from gsdiff.data.artifacts import publish_dataset as real_publish

    cli = _load_build_datasets_cli()
    artifact_root = tmp_path / "artifacts"
    initial_scan_seen = False
    publisher_calls = 0
    real_discover = cli.discover_dataset_directories
    real_generate = cli.generate_corrected_dataset

    def observing_discover(path):
        nonlocal initial_scan_seen
        discovery = real_discover(path)
        if not initial_scan_seen:
            assert discovery.canonical_directories == ()
            initial_scan_seen = True
        return discovery

    def generate_after_winner(**kwargs):
        assert initial_scan_seen is True
        generated = real_generate(**kwargs)
        real_publish(artifact_root, generated)
        return generated

    def observing_publish(root, generated):
        nonlocal publisher_calls
        publisher_calls += 1
        return real_publish(root, generated)

    monkeypatch.setattr(cli, "_environment", lambda: CONTROLLED_RUNTIME)
    monkeypatch.setattr(
        cli, "_git_state", lambda: _controlled_git_state()
    )
    monkeypatch.setattr(
        cli, "discover_dataset_directories", observing_discover
    )
    monkeypatch.setattr(
        cli, "generate_corrected_dataset", generate_after_winner
    )
    monkeypatch.setattr(cli, "publish_dataset", observing_publish)

    return_code = cli.main(
        [
            "--protocol",
            str(REPO_ROOT / "configs" / "protocols" / "pilot-v1.yaml"),
            "--artifact-root",
            str(artifact_root),
        ]
    )

    captured = capsys.readouterr()
    assert return_code == 0
    assert captured.err == ""
    assert initial_scan_seen is True
    assert publisher_calls == 1
    assert json.loads(captured.out)["datasets"][0]["status"] == "reused"


@pytest.mark.parametrize(
    "mutation",
    ["disappear", "replace", "exact-replace"],
)
def test_task3_round1_build_fails_closed_when_unique_candidate_changes(
    tmp_path, capsys, monkeypatch, mutation
):
    cli = _load_build_datasets_cli()
    artifact_root = tmp_path / "artifacts"
    _, publication = _publish_controlled_pilot_dataset(
        cli, artifact_root
    )
    dataset_dir = publication.dataset_dir
    real_verify = cli.verify_dataset_directory
    verify_calls = 0

    def verify_then_mutate(*args, **kwargs):
        nonlocal verify_calls
        verify_calls += 1
        verified = real_verify(*args, **kwargs)
        if verify_calls == 1:
            if mutation == "disappear":
                dataset_dir.rename(tmp_path / "removed-dataset")
            elif mutation == "replace":
                dataset_dir.rename(tmp_path / "replaced-dataset")
                dataset_dir.mkdir()
            else:
                clone_source = tmp_path / "clone-source"
                shutil.copytree(dataset_dir, clone_source)
                dataset_dir.rename(tmp_path / "replaced-dataset")
                shutil.copytree(clone_source, dataset_dir)
        return verified

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "changed unique candidate attempted generation or publication"
        )

    monkeypatch.setattr(cli, "_environment", lambda: CONTROLLED_RUNTIME)
    monkeypatch.setattr(
        cli, "_git_state", lambda: _controlled_git_state()
    )
    monkeypatch.setattr(
        cli, "verify_dataset_directory", verify_then_mutate
    )
    monkeypatch.setattr(cli, "generate_corrected_dataset", forbidden)
    monkeypatch.setattr(cli, "publish_dataset", forbidden)

    return_code = cli.main(
        [
            "--protocol",
            str(REPO_ROOT / "configs" / "protocols" / "pilot-v1.yaml"),
            "--artifact-root",
            str(artifact_root),
        ]
    )

    captured = capsys.readouterr()
    assert return_code == 1
    assert json.loads(captured.out)["errors"] == [
        {"code": "artifact-validation-error"}
    ]
    assert captured.err == (
        "dataset build failed: artifact-validation-error\n"
    )
    assert verify_calls == 2


def test_task3_round1_final_catalog_rejects_twin_injected_after_publication(
    tmp_path, capsys, monkeypatch
):
    from gsdiff.data.artifacts import publish_dataset as real_publish

    cli = _load_build_datasets_cli()
    artifact_root = tmp_path / "artifacts"
    injected_identities = []
    generated_for_twin = None
    real_verify = cli.verify_dataset_directory
    verify_calls = 0

    def observing_publish(root, generated):
        nonlocal generated_for_twin
        publication = real_publish(root, generated)
        generated_for_twin = generated
        injected_identities.append(
            publication.verified.dataset_identity_sha256
        )
        return publication

    def verify_then_inject_twin(*args, **kwargs):
        nonlocal verify_calls
        verify_calls += 1
        verified = real_verify(*args, **kwargs)
        if verify_calls == 1:
            assert generated_for_twin is not None
            twin = _coherent_semantic_twin(generated_for_twin)
            twin_publication = real_publish(artifact_root, twin)
            injected_identities.append(
                twin_publication.verified.dataset_identity_sha256
            )
        return verified

    monkeypatch.setattr(cli, "_environment", lambda: CONTROLLED_RUNTIME)
    monkeypatch.setattr(
        cli, "_git_state", lambda: _controlled_git_state()
    )
    monkeypatch.setattr(cli, "publish_dataset", observing_publish)
    monkeypatch.setattr(
        cli, "verify_dataset_directory", verify_then_inject_twin
    )

    return_code = cli.main(
        [
            "--protocol",
            str(REPO_ROOT / "configs" / "protocols" / "pilot-v1.yaml"),
            "--artifact-root",
            str(artifact_root),
        ]
    )

    captured = capsys.readouterr()
    assert return_code == 1
    assert json.loads(captured.out)["errors"] == [
        {"code": "ambiguous-current-dataset"}
    ]
    assert len(set(injected_identities)) == 2
    assert {
        path.name for path in (artifact_root / "datasets").iterdir()
    } == set(injected_identities)


def test_task3_round1_final_catalog_rechecks_replaced_corrupt_addition(
    tmp_path, capsys, monkeypatch
):
    from gsdiff.data.artifacts import (
        ArtifactValidationError,
        publish_dataset as real_publish,
    )

    cli = _load_build_datasets_cli()
    artifact_root = tmp_path / "artifacts"
    generated_for_twin = None
    original_dir = None
    twin = None
    twin_dir = None
    injected = False
    replacement_installed = False
    real_verify = cli.verify_dataset_directory

    def observing_publish(root, generated):
        nonlocal generated_for_twin, original_dir
        publication = real_publish(root, generated)
        generated_for_twin = generated
        original_dir = publication.dataset_dir
        return publication

    def verify_with_replaced_addition(*args, **kwargs):
        nonlocal twin, twin_dir, injected, replacement_installed
        path = Path(args[0])
        if path == original_dir:
            verified = real_verify(*args, **kwargs)
            if not injected:
                assert generated_for_twin is not None
                twin = _coherent_semantic_twin(generated_for_twin)
                twin_dir = (
                    artifact_root
                    / "datasets"
                    / twin.dataset_identity_sha256
                )
                twin_dir.mkdir()
                injected = True
            return verified
        if path == twin_dir and not replacement_installed:
            with pytest.raises(ArtifactValidationError):
                real_verify(*args, **kwargs)
            twin_dir.rmdir()
            real_publish(artifact_root, twin)
            replacement_installed = True
            raise ArtifactValidationError(
                "injected corrupt addition verification failure"
            )
        return real_verify(*args, **kwargs)

    monkeypatch.setattr(cli, "_environment", lambda: CONTROLLED_RUNTIME)
    monkeypatch.setattr(
        cli, "_git_state", lambda: _controlled_git_state()
    )
    monkeypatch.setattr(cli, "publish_dataset", observing_publish)
    monkeypatch.setattr(
        cli, "verify_dataset_directory", verify_with_replaced_addition
    )

    return_code = cli.main(
        [
            "--protocol",
            str(REPO_ROOT / "configs" / "protocols" / "pilot-v1.yaml"),
            "--artifact-root",
            str(artifact_root),
        ]
    )

    captured = capsys.readouterr()
    assert injected is replacement_installed is True
    assert return_code == 1
    assert json.loads(captured.out)["errors"] == [
        {"code": "ambiguous-current-dataset"}
    ]


def test_task3_round1_final_catalog_rechecks_corrupt_candidate_leaf_recovery(
    tmp_path, capsys, monkeypatch
):
    from gsdiff.data.artifacts import (
        ArtifactValidationError,
        publish_dataset,
    )

    cli = _load_build_datasets_cli()
    artifact_root = tmp_path / "artifacts"
    _, generated = _controlled_pilot_request_and_generated(cli)
    first = publish_dataset(artifact_root, generated)
    twin = _coherent_semantic_twin(generated)
    second = publish_dataset(artifact_root, twin)
    payload = second.dataset_dir / "measurements.npz"
    payload.chmod(0o600)
    original_payload = payload.read_bytes()
    damaged = bytearray(original_payload)
    damaged[-1] ^= 1
    payload.write_bytes(damaged)
    real_verify = cli.verify_dataset_directory
    corrupt_attempts = 0
    restored = False

    def verify_then_restore(*args, **kwargs):
        nonlocal corrupt_attempts, restored
        if Path(args[0]) != second.dataset_dir:
            return real_verify(*args, **kwargs)
        corrupt_attempts += 1
        try:
            return real_verify(*args, **kwargs)
        except ArtifactValidationError:
            if corrupt_attempts == 2:
                payload.write_bytes(original_payload)
                restored = True
            raise

    monkeypatch.setattr(cli, "_environment", lambda: CONTROLLED_RUNTIME)
    monkeypatch.setattr(
        cli, "_git_state", lambda: _controlled_git_state()
    )
    monkeypatch.setattr(cli, "verify_dataset_directory", verify_then_restore)

    return_code = cli.main(
        [
            "--protocol",
            str(REPO_ROOT / "configs" / "protocols" / "pilot-v1.yaml"),
            "--artifact-root",
            str(artifact_root),
        ]
    )

    captured = capsys.readouterr()
    assert restored is True
    assert return_code == 1
    assert json.loads(captured.out)["errors"] == [
        {"code": "ambiguous-current-dataset"}
    ]
    assert cli._semantic_projection_from_manifest(
        first.verified.manifest
    ) == cli._semantic_projection_from_manifest(second.verified.manifest)


def test_task3_round1_final_catalog_consumes_last_anchor_addition(
    tmp_path, capsys, monkeypatch
):
    from gsdiff.data.artifacts import publish_dataset as real_publish

    cli = _load_build_datasets_cli()
    artifact_root = tmp_path / "artifacts"
    generated_for_twin = None
    injected = False
    real_recheck = cli.verify_canonical_dataset_directory_discovery

    def observing_publish(root, generated):
        nonlocal generated_for_twin
        publication = real_publish(root, generated)
        generated_for_twin = generated
        return publication

    def recheck_then_inject(discovery):
        nonlocal injected
        if (
            not injected
            and generated_for_twin is not None
            and discovery.datasets_dir_exists
            and tuple(
                path.name for path in discovery.canonical_directories
            )
            == (generated_for_twin.dataset_identity_sha256,)
        ):
            real_publish(
                artifact_root,
                _coherent_semantic_twin(generated_for_twin),
            )
            injected = True
        return real_recheck(discovery)

    monkeypatch.setattr(cli, "_environment", lambda: CONTROLLED_RUNTIME)
    monkeypatch.setattr(
        cli, "_git_state", lambda: _controlled_git_state()
    )
    monkeypatch.setattr(cli, "publish_dataset", observing_publish)
    monkeypatch.setattr(
        cli,
        "verify_canonical_dataset_directory_discovery",
        recheck_then_inject,
    )

    return_code = cli.main(
        [
            "--protocol",
            str(REPO_ROOT / "configs" / "protocols" / "pilot-v1.yaml"),
            "--artifact-root",
            str(artifact_root),
        ]
    )

    captured = capsys.readouterr()
    assert injected is True
    assert return_code == 1
    assert json.loads(captured.out)["errors"] == [
        {"code": "ambiguous-current-dataset"}
    ]


def test_task3_round1_initial_stabilization_absorbs_missing_root_winner(
    tmp_path, capsys, monkeypatch
):
    from gsdiff.data.artifacts import publish_dataset as real_publish

    cli = _load_build_datasets_cli()
    artifact_root = tmp_path / "artifacts"
    request, generated = _controlled_pilot_request_and_generated(cli)
    injected = False
    real_recheck = cli.verify_canonical_dataset_directory_discovery

    def inject_winner_before_missing_recheck(discovery):
        nonlocal injected
        if not discovery.datasets_dir_exists and not injected:
            real_publish(artifact_root, generated)
            injected = True
        return real_recheck(discovery)

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "missing-root winner absorption attempted generation/publication"
        )

    monkeypatch.setattr(cli, "_environment", lambda: CONTROLLED_RUNTIME)
    monkeypatch.setattr(
        cli, "_git_state", lambda: _controlled_git_state()
    )
    monkeypatch.setattr(
        cli,
        "verify_canonical_dataset_directory_discovery",
        inject_winner_before_missing_recheck,
    )
    monkeypatch.setattr(cli, "generate_corrected_dataset", forbidden)
    monkeypatch.setattr(cli, "publish_dataset", forbidden)

    return_code = cli.main(
        [
            "--protocol",
            str(REPO_ROOT / "configs" / "protocols" / "pilot-v1.yaml"),
            "--artifact-root",
            str(artifact_root),
        ]
    )

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert injected is True
    assert return_code == 0
    assert captured.err == ""
    assert report["datasets"] == [
        {
            "dataset_identity_sha256": (
                generated.dataset_identity_sha256
            ),
            "request_sha256": request.request_sha256,
            "status": "reused",
        }
    ]


@pytest.mark.parametrize("mutation", ["add", "remove"])
def test_task3_round1_unique_reuse_ignores_unrelated_staging_churn(
    tmp_path, capsys, monkeypatch, mutation
):
    cli = _load_build_datasets_cli()
    artifact_root = tmp_path / "artifacts"
    _, publication = _publish_controlled_pilot_dataset(
        cli, artifact_root
    )
    staging = (
        artifact_root
        / "datasets"
        / f".{'f' * 64}.staging-round1"
    )
    if mutation == "remove":
        staging.mkdir()
    real_verify = cli.verify_dataset_directory
    verify_calls = 0

    def verify_then_churn(*args, **kwargs):
        nonlocal verify_calls
        verify_calls += 1
        verified = real_verify(*args, **kwargs)
        if verify_calls == 1:
            if mutation == "add":
                staging.mkdir()
            else:
                staging.rmdir()
        return verified

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "staging churn blocked no-generation unique reuse"
        )

    monkeypatch.setattr(cli, "_environment", lambda: CONTROLLED_RUNTIME)
    monkeypatch.setattr(
        cli, "_git_state", lambda: _controlled_git_state()
    )
    monkeypatch.setattr(
        cli, "verify_dataset_directory", verify_then_churn
    )
    monkeypatch.setattr(cli, "generate_corrected_dataset", forbidden)
    monkeypatch.setattr(cli, "publish_dataset", forbidden)

    return_code = cli.main(
        [
            "--protocol",
            str(REPO_ROOT / "configs" / "protocols" / "pilot-v1.yaml"),
            "--artifact-root",
            str(artifact_root),
        ]
    )

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert return_code == 0
    assert captured.err == ""
    assert report["datasets"][0]["status"] == "reused"
    assert report["datasets"][0]["dataset_identity_sha256"] == (
        publication.verified.dataset_identity_sha256
    )
    assert report["stale_staging_count"] == (
        1 if mutation == "add" else 0
    )
    assert verify_calls >= 3


# Task 6 binds the production entry points to one frozen materialized request.
# These helpers deliberately build a real acquisition and real canonical method
# while replacing only the already-tested materialized-config loader.
def _strict_method_acquisition() -> SPIAcquisitionData:
    rng = np.random.default_rng(20260728)
    T, H, W, rows = 4, 8, 8, 2
    patterns = rng.random((T * rows, H, W), dtype=np.float32)
    frame_indices = np.repeat(np.arange(T, dtype=np.int64), rows)
    source = rng.random((T, H, W), dtype=np.float32)
    measurements = np.einsum(
        "khw,khw->k",
        patterns,
        source[frame_indices],
    ).astype(np.float32)
    holdout_patterns = rng.random((T, H, W), dtype=np.float32)
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
        dataset_identity_sha256="a" * 64,
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


def _strict_method_case(
    tmp_path: Path,
    method_id: str,
    *,
    requested_device: str = "cpu",
    child_device: str = "cpu",
    save_measurements: bool = False,
) -> tuple[
    MaterializedMethodRequest,
    SPIAcquisitionData,
    Path,
    list[str],
]:
    acquisition = _strict_method_acquisition()
    stage = tmp_path / f"stage-{method_id}"
    measurements_path = stage / "input" / "measurements.npz"
    method_config_path = stage / "config" / "method-config.json"
    child_output_dir = stage / "child-output"
    measurements_path.parent.mkdir(parents=True)
    method_config_path.parent.mkdir(parents=True)
    child_output_dir.mkdir(parents=True)
    method_config_path.write_text("{}", encoding="utf-8")
    if save_measurements:
        save_acquisition_data(acquisition, measurements_path)
    else:
        measurements_path.write_bytes(b"unit-test-measurements")
    base_config = (
        {"gaussian_count": 1000}
        if method_id in {"gsdiff_tv", "gsdiff_diffusion"}
        else {}
    )
    method = resolve_method_semantics(
        method_id,
        method_config_id="smoke-default-v1",
        base_config=base_config,
        measurements_metadata={
            "H": acquisition.H,
            "W": acquisition.W,
            "T": acquisition.T,
            "K": acquisition.K,
            "holdout_K": acquisition.holdout_K,
        },
        execution_profile="controller-cpu-smoke-v1",
    )
    algorithm_seed = derive_algorithm_seed(
        cell_seed=17,
        dataset_identity_sha256=acquisition.dataset_identity_sha256,
        method_id=method.method_id,
        method_config_sha256=method.method_config_sha256,
    )
    checkpoint_paths: dict[str, Path] = {}
    for requirement in method.checkpoint_requirements:
        checkpoint_path = (
            stage / "checkpoints" / f"{requirement.logical_id}.checkpoint"
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_bytes(b"test-checkpoint")
        checkpoint_paths[requirement.logical_id] = checkpoint_path
    request = MaterializedMethodRequest(
        method=method,
        algorithm_seed=algorithm_seed,
        dataset_identity_sha256=acquisition.dataset_identity_sha256,
        measurements_file_sha256=artifact_sha256(measurements_path),
        expected_acquisition_spec=blind_acquisition_spec(acquisition),
        measurements_path=measurements_path.resolve(),
        child_output_dir=child_output_dir.resolve(),
        checkpoint_paths=MappingProxyType(checkpoint_paths),
        requested_runtime_device=requested_device,
        child_runtime_device=child_device,
    )
    argv = [
        "--method",
        method.method_id,
        "--dataset",
        str(request.measurements_path),
        "--dataset-identity-sha256",
        request.dataset_identity_sha256,
        "--method-config",
        str(method_config_path.resolve()),
        "--algorithm-seed",
        str(request.algorithm_seed.seed_u32),
        "--device",
        request.child_runtime_device,
        "--output-dir",
        str(request.child_output_dir),
    ]
    for logical_id, checkpoint_path in request.checkpoint_paths.items():
        argv.extend(
            ["--checkpoint", f"{logical_id}={checkpoint_path}"]
        )
    return request, acquisition, method_config_path.resolve(), argv


def _strict_entry_module(family: str):
    if family == "baseline":
        import scripts.run_baselines as module
    elif family == "gsdiff":
        import train as module
    else:
        raise AssertionError(f"unknown test family: {family}")
    return module


@pytest.mark.parametrize("family", ["baseline", "gsdiff"])
def test_method_entry_usage_documents_strict_default_and_marked_legacy(
    tmp_path,
    monkeypatch,
    family,
):
    module = _strict_entry_module(family)
    usage = module.__doc__ or ""
    for option in (
        "--method",
        "--dataset",
        "--dataset-identity-sha256",
        "--method-config",
        "--algorithm-seed",
        "--device",
        "--output-dir",
    ):
        assert option in usage
    assert "--legacy-compatibility" in usage
    for line in usage.splitlines():
        stripped = line.strip()
        if not stripped.startswith("python "):
            continue
        if any(
            option in stripped
            for option in ("--config", "--solver", "--baselines")
        ):
            assert "--legacy-compatibility" in stripped

    commands = [
        shlex.split(line.strip())
        for line in usage.splitlines()
        if line.strip().startswith("python ")
    ]
    assert len(commands) == 2
    strict_documented = commands[0][2:]
    legacy_documented = commands[1][2:]

    method_id = "dgi" if family == "baseline" else "gsdiff_tv"
    request, _acquisition, config_path, canonical_argv = (
        _strict_method_case(tmp_path, method_id)
    )
    canonical_values = {
        canonical_argv[index]: canonical_argv[index + 1]
        for index in range(0, len(canonical_argv), 2)
    }
    placeholder_values = {
        "<absolute-measurements>": canonical_values["--dataset"],
        "<sha256>": canonical_values["--dataset-identity-sha256"],
        "<absolute-config>": canonical_values["--method-config"],
        "<u32>": canonical_values["--algorithm-seed"],
        "<absolute-output>": canonical_values["--output-dir"],
    }
    strict_argv = [
        placeholder_values.get(value, value)
        for value in strict_documented
    ]
    parsed = module._strict_parser().parse_args(strict_argv)
    code_dir = config_path.parent.parent / "code"
    code_dir.mkdir()
    monkeypatch.chdir(code_dir)
    locked_config = module._crosslock_method_config_before_load(
        parsed.method_config
    )
    if family == "baseline":
        module._crosslock_request(parsed, request, locked_config)
    else:
        module._crosslock_request(
            parsed,
            request,
            locked_config,
            module._checkpoint_mapping(parsed.checkpoint),
        )

    assert "--legacy-compatibility" in legacy_documented
    assert "--truth-path" in legacy_documented
    assert "--measurements-path" in legacy_documented
    assert "--dataset-identity-sha256" in legacy_documented
    if family == "baseline":
        assert "--name" in legacy_documented
    legacy_values = {
        "<truth>": str(tmp_path / "truth.npz"),
        "<measurements>": str(tmp_path / "measurements.npz"),
        "<sha256>": "a" * 64,
    }
    legacy_argv = [
        legacy_values.get(value, value)
        for value in legacy_documented
    ]
    observed = []
    monkeypatch.setattr(
        module,
        "legacy_main",
        lambda argv: observed.append(list(argv)) or 0,
    )
    assert module.main(legacy_argv) == 0
    assert observed


def _patch_strict_request_loader(
    monkeypatch,
    *,
    expected_path: Path,
    request: MaterializedMethodRequest,
) -> list[Path]:
    import gsdiff.experiments.execution as execution

    code_dir = request.measurements_path.parent.parent / "code"
    code_dir.mkdir(exist_ok=True)
    monkeypatch.chdir(code_dir)
    loaded: list[Path] = []

    def load(path):
        loaded.append(path)
        assert path == expected_path
        return request

    monkeypatch.setattr(
        execution,
        "load_materialized_method_request",
        load,
    )
    return loaded


@pytest.mark.parametrize("family", ["baseline", "gsdiff"])
def test_strict_method_rejects_external_config_before_loader_or_read(
    tmp_path,
    monkeypatch,
    family,
):
    import gsdiff.experiments.execution as execution

    method_id = "dgi" if family == "baseline" else "gsdiff_tv"
    request, _acquisition, config_path, argv = _strict_method_case(
        tmp_path, method_id
    )
    code_dir = config_path.parent.parent / "code"
    code_dir.mkdir()
    monkeypatch.chdir(code_dir)
    external_config = (
        tmp_path
        / "external-stage"
        / "config"
        / "method-config.json"
    )
    external_config.parent.mkdir(parents=True)
    external_config.write_bytes(b'{"external_sentinel":true}')
    changed = list(argv)
    changed[changed.index("--method-config") + 1] = str(
        external_config.resolve()
    )
    loader_calls = []
    read_calls = []
    real_loader = execution.load_materialized_method_request
    real_read = execution._read_stable_regular_bytes

    def tracked_read(path, *, noun):
        read_calls.append(path)
        return real_read(path, noun=noun)

    def tracked_loader(path):
        loader_calls.append(path)
        return real_loader(path)

    monkeypatch.setattr(execution, "_read_stable_regular_bytes", tracked_read)
    monkeypatch.setattr(
        execution,
        "load_materialized_method_request",
        tracked_loader,
    )

    with pytest.raises(ValueError, match="method config.*crosslock"):
        _strict_entry_module(family).strict_main(changed)
    assert loader_calls == []
    assert read_calls == []
    assert external_config.read_bytes() == b'{"external_sentinel":true}'


@pytest.mark.parametrize("family", ["baseline", "gsdiff"])
def test_strict_method_dispatch_is_default_and_legacy_requires_marker(
    monkeypatch,
    family,
):
    module = _strict_entry_module(family)
    calls = []

    def strict(argv):
        calls.append(("strict", list(argv)))
        return 71

    def legacy(argv):
        calls.append(("legacy", list(argv)))
        return 72

    monkeypatch.setattr(module, "strict_main", strict, raising=False)
    monkeypatch.setattr(module, "legacy_main", legacy, raising=False)

    assert module.main(["--method", "dgi"]) == 71
    assert module.main(
        [
            "--legacy-compatibility",
            "--config",
            "legacy.yaml",
            "--truth-path",
            "truth.npz",
        ]
    ) == 72
    assert calls == [
        ("strict", ["--method", "dgi"]),
        (
            "legacy",
            ["--config", "legacy.yaml", "--truth-path", "truth.npz"],
        ),
    ]


@pytest.mark.parametrize("family", ["baseline", "gsdiff"])
@pytest.mark.parametrize(
    "legacy_argv",
    [
        ["--config", "legacy.yaml", "--task6-invalid-sentinel"],
        ["--truth-path", "truth.npz", "--task6-invalid-sentinel"],
        ["--override", "solver.steps=1", "--task6-invalid-sentinel"],
        ["--name", "legacy-name", "--task6-invalid-sentinel"],
        ["--baselines", "dgi", "--task6-invalid-sentinel"],
        ["--solver", "sgd", "--task6-invalid-sentinel"],
    ],
)
def test_strict_method_default_rejects_legacy_flags_with_migration_message(
    capsys,
    family,
    legacy_argv,
):
    module = _strict_entry_module(family)
    with pytest.raises(SystemExit):
        module.main(legacy_argv)
    captured = capsys.readouterr()
    assert "--legacy-compatibility" in captured.err


@pytest.mark.parametrize("family", ["baseline", "gsdiff"])
@pytest.mark.parametrize(
    "duplicate_flag",
    [
        "--method",
        "--dataset",
        "--dataset-identity-sha256",
        "--method-config",
        "--algorithm-seed",
        "--device",
        "--output-dir",
    ],
)
def test_strict_method_rejects_duplicate_singleton_options(
    tmp_path,
    family,
    duplicate_flag,
):
    method_id = "dgi" if family == "baseline" else "gsdiff_tv"
    _request, _acquisition, _config, argv = _strict_method_case(
        tmp_path, method_id
    )
    index = argv.index(duplicate_flag)
    duplicate_value = argv[index + 1]
    module = _strict_entry_module(family)

    with pytest.raises(SystemExit, match="duplicate"):
        module.strict_main(
            [*argv, duplicate_flag, duplicate_value]
        )


@pytest.mark.parametrize(
    ("family", "method_id"),
    [
        ("baseline", "gsdiff_tv"),
        ("gsdiff", "dgi"),
    ],
)
def test_strict_method_family_gate_precedes_adapter_and_writer(
    tmp_path,
    monkeypatch,
    family,
    method_id,
):
    import gsdiff.data as data
    import gsdiff.experiments.adapters as adapters
    import gsdiff.experiments.child_outputs as child_outputs

    request, _acquisition, config_path, argv = _strict_method_case(
        tmp_path, method_id
    )
    _patch_strict_request_loader(
        monkeypatch,
        expected_path=config_path,
        request=request,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("family mismatch reached execution")

    monkeypatch.setattr(data, "load_acquisition_data", forbidden)
    monkeypatch.setattr(adapters, "run_canonical_method", forbidden)
    monkeypatch.setattr(
        child_outputs,
        "write_method_child_outputs_v2",
        forbidden,
    )

    with pytest.raises(ValueError, match="family"):
        _strict_entry_module(family).strict_main(argv)


@pytest.mark.parametrize(
    ("family", "method_id", "replacement"),
    [
        ("baseline", "dgi", ("--method", "unknown-method")),
        ("baseline", "dgi", ("--method", "gsdiff_diff")),
        ("gsdiff", "gsdiff_diffusion", ("--method", "gsdiff_diff")),
    ],
)
def test_strict_method_rejects_unknown_ids_and_aliases(
    tmp_path,
    monkeypatch,
    family,
    method_id,
    replacement,
):
    request, _acquisition, config_path, argv = _strict_method_case(
        tmp_path, method_id
    )
    _patch_strict_request_loader(
        monkeypatch,
        expected_path=config_path,
        request=request,
    )
    option, value = replacement
    changed = list(argv)
    changed[changed.index(option) + 1] = value

    with pytest.raises(ValueError, match="method"):
        _strict_entry_module(family).strict_main(changed)


@pytest.mark.parametrize("family", ["baseline", "gsdiff"])
@pytest.mark.parametrize(
    "field",
    [
        "method",
        "dataset",
        "dataset_identity",
        "method_config",
        "output",
        "algorithm_seed",
        "device",
    ],
)
def test_strict_method_crosslocks_every_cli_value_before_execution(
    tmp_path,
    monkeypatch,
    family,
    field,
):
    import gsdiff.experiments.adapters as adapters
    import gsdiff.experiments.child_outputs as child_outputs

    method_id = "dgi" if family == "baseline" else "gsdiff_tv"
    request, _acquisition, config_path, argv = _strict_method_case(
        tmp_path, method_id
    )
    changed = list(argv)
    replacements = {
        "method": ("--method", "static_cs" if family == "baseline" else "siren"),
        "dataset": ("--dataset", str(tmp_path / "other-measurements.npz")),
        "dataset_identity": (
            "--dataset-identity-sha256",
            "f" * 64,
        ),
        "method_config": (
            "--method-config",
            str(tmp_path / "other-method-config.json"),
        ),
        "output": ("--output-dir", str(tmp_path / "other-output")),
        "algorithm_seed": (
            "--algorithm-seed",
            str((request.algorithm_seed.seed_u32 + 1) % (2**32)),
        ),
        "device": ("--device", "cuda:0"),
    }
    option, value = replacements[field]
    changed[changed.index(option) + 1] = value

    # A config-path mutation is rejected from trusted cwd/stage structure
    # before the request loader can read or hash the supplied file. Other
    # parent-known mutations are cross-locked against the loaded request.
    loaded = _patch_strict_request_loader(
        monkeypatch,
        expected_path=config_path,
        request=request,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("tampered request reached execution")

    monkeypatch.setattr(adapters, "run_canonical_method", forbidden)
    monkeypatch.setattr(
        child_outputs,
        "write_method_child_outputs_v2",
        forbidden,
    )

    with pytest.raises(ValueError, match="mismatch|crosslock"):
        _strict_entry_module(family).strict_main(changed)
    assert loaded == ([] if field == "method_config" else [config_path])


def test_strict_method_baseline_rejects_any_checkpoint(tmp_path):
    _request, _acquisition, _config, argv = _strict_method_case(
        tmp_path, "dgi"
    )
    with pytest.raises(SystemExit):
        _strict_entry_module("baseline").strict_main(
            [*argv, "--checkpoint", f"extra={tmp_path / 'extra.pt'}"]
        )


@pytest.mark.parametrize(
    "checkpoint_args",
    [
        ["--checkpoint", "extra=extra.pt"],
        [
            "--checkpoint",
            "extra=extra.pt",
            "--checkpoint",
            "extra=extra.pt",
        ],
    ],
)
def test_strict_method_non_diffusion_gsdiff_rejects_checkpoints(
    tmp_path,
    monkeypatch,
    checkpoint_args,
):
    request, _acquisition, config_path, argv = _strict_method_case(
        tmp_path, "gsdiff_tv"
    )
    _patch_strict_request_loader(
        monkeypatch,
        expected_path=config_path,
        request=request,
    )
    with pytest.raises(ValueError, match="checkpoint"):
        _strict_entry_module("gsdiff").strict_main(
            [*argv, *checkpoint_args]
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "duplicate",
        "extra",
        "wrong_logical_id",
        "wrong_path",
    ],
)
def test_strict_method_diffusion_requires_exact_checkpoint_mapping(
    tmp_path,
    monkeypatch,
    mutation,
):
    request, _acquisition, config_path, argv = _strict_method_case(
        tmp_path, "gsdiff_diffusion"
    )
    _patch_strict_request_loader(
        monkeypatch,
        expected_path=config_path,
        request=request,
    )
    checkpoint_index = argv.index("--checkpoint")
    assignment = argv[checkpoint_index + 1]
    changed = list(argv)
    if mutation == "missing":
        del changed[checkpoint_index : checkpoint_index + 2]
    elif mutation == "duplicate":
        changed.extend(["--checkpoint", assignment])
    elif mutation == "extra":
        changed.extend(["--checkpoint", f"extra={tmp_path / 'extra.pt'}"])
    elif mutation == "wrong_logical_id":
        changed[checkpoint_index + 1] = (
            f"wrong={next(iter(request.checkpoint_paths.values()))}"
        )
    elif mutation == "wrong_path":
        changed[checkpoint_index + 1] = (
            f"gsdiff-diffusion-prior-v1={tmp_path / 'wrong.pt'}"
        )
    else:
        raise AssertionError(mutation)

    with pytest.raises(ValueError, match="checkpoint"):
        _strict_entry_module("gsdiff").strict_main(changed)


def test_strict_method_exact_diffusion_mapping_and_child_device_are_forwarded(
    tmp_path,
    monkeypatch,
):
    import gsdiff.data as data
    import gsdiff.experiments.adapters as adapters
    import gsdiff.experiments.child_outputs as child_outputs

    request, acquisition, config_path, argv = _strict_method_case(
        tmp_path,
        "gsdiff_diffusion",
        requested_device="cuda:3",
        child_device="cuda:0",
    )
    _patch_strict_request_loader(
        monkeypatch,
        expected_path=config_path,
        request=request,
    )
    calls = []
    sentinel_result = object()

    def load_acquisition(path, **kwargs):
        calls.append(("load", path, kwargs))
        return acquisition

    def run(method, loaded, **kwargs):
        calls.append(("run", method, loaded, kwargs))
        return sentinel_result

    def write(output_dir, **kwargs):
        calls.append(("write", output_dir, kwargs))
        return {
            "reconstruction.npz": "1" * 64,
            "method-info.json": "2" * 64,
        }

    monkeypatch.setattr(data, "load_acquisition_data", load_acquisition)
    monkeypatch.setattr(adapters, "run_canonical_method", run)
    monkeypatch.setattr(
        child_outputs,
        "write_method_child_outputs_v2",
        write,
    )

    assert _strict_entry_module("gsdiff").strict_main(argv) == 0
    run_call = next(call for call in calls if call[0] == "run")
    assert run_call[3]["device"] == "cuda:0"
    assert run_call[3]["device"] != request.requested_runtime_device
    assert run_call[3]["checkpoint_paths"] == request.checkpoint_paths
    write_call = next(call for call in calls if call[0] == "write")
    assert write_call[1] == request.child_output_dir
    assert write_call[2]["result"] is sentinel_result


def test_strict_method_real_dgi_success_writes_exact_two_v2_files(
    tmp_path,
    monkeypatch,
    capsys,
):
    request, acquisition, config_path, argv = _strict_method_case(
        tmp_path, "dgi", save_measurements=True
    )
    _patch_strict_request_loader(
        monkeypatch,
        expected_path=config_path,
        request=request,
    )

    assert _strict_entry_module("baseline").strict_main(argv) == 0
    assert {path.name for path in request.child_output_dir.iterdir()} == {
        "reconstruction.npz",
        "method-info.json",
    }
    validate_method_child_outputs_v2(
        request.child_output_dir,
        expected_method=request.method,
        expected_acquisition=acquisition,
        expected_dataset_identity_sha256=request.dataset_identity_sha256,
        expected_measurements_file_sha256=(
            request.measurements_file_sha256
        ),
        expected_algorithm_seed=request.algorithm_seed,
    )
    output = capsys.readouterr().out.lower()
    assert "reconstruction.npz" in output
    assert "method-info.json" in output
    for forbidden in ("psnr", "ssim", "nrmse", "truth", "figure"):
        assert forbidden not in output


@pytest.mark.parametrize("family", ["baseline", "gsdiff"])
def test_strict_method_explicit_legacy_never_calls_v2_writer(
    monkeypatch,
    capsys,
    family,
):
    import gsdiff.experiments.child_outputs as child_outputs

    module = _strict_entry_module(family)
    observed = []

    def legacy(argv):
        observed.append(list(argv))
        return 19

    def forbidden(*args, **kwargs):
        raise AssertionError("legacy compatibility reached v2 writer")

    monkeypatch.setattr(module, "legacy_main", legacy, raising=False)
    monkeypatch.setattr(
        child_outputs,
        "write_method_child_outputs_v2",
        forbidden,
    )

    assert module.main(
        [
            "--legacy-compatibility",
            "--config",
            "legacy.yaml",
            "--truth-path",
            "truth.npz",
        ]
    ) == 19
    captured = capsys.readouterr()
    assert "nonpromotable" in (captured.out + captured.err).lower()
    assert observed == [
        ["--config", "legacy.yaml", "--truth-path", "truth.npz"]
    ]


@pytest.mark.parametrize("family", ["baseline", "gsdiff"])
def test_explicit_legacy_without_truth_fails_closed_before_artifacts(
    tmp_path,
    monkeypatch,
    capsys,
    family,
):
    import gsdiff.experiments.child_outputs as child_outputs

    module = _strict_entry_module(family)
    output_dir = tmp_path / "legacy-output"

    def forbidden(*args, **kwargs):
        raise AssertionError("truthless legacy reached execution or writer")

    monkeypatch.setattr(module, "legacy_main", forbidden, raising=False)
    monkeypatch.setattr(
        child_outputs,
        "write_method_child_outputs_v2",
        forbidden,
    )
    monkeypatch.setattr(
        module,
        "write_method_child_outputs",
        forbidden,
    )

    with pytest.raises(SystemExit):
        module.main(
            [
                "--legacy-compatibility",
                "--config",
                "legacy.yaml",
                "--measurements-path",
                str(tmp_path / "measurements.npz"),
                "--output-dir",
                str(output_dir),
            ]
        )
    captured = capsys.readouterr()
    message = (captured.out + captured.err).lower()
    assert "truth" in message
    assert "strict" in message
    assert not output_dir.exists()


@pytest.mark.parametrize("family", ["baseline", "gsdiff"])
@pytest.mark.parametrize(
    ("truth_args", "expected_message"),
    [
        (["--truth-path="], "truthless"),
        (["--truth-path"], "truthless"),
        (["--truth-path", ""], "truthless"),
        (["--truth-path=   "], "truthless"),
        (["--truth-path", "   "], "truthless"),
        (
            ["--truth-path=truth.npz", "--truth-path="],
            "duplicate",
        ),
        (
            ["--truth-path", "truth.npz", "--truth-path="],
            "duplicate",
        ),
        (
            [
                "--truth-path=truth.npz",
                "--truth-path",
                "other-truth.npz",
            ],
            "duplicate",
        ),
    ],
)
def test_explicit_legacy_rejects_empty_or_duplicate_truth_option(
    monkeypatch,
    capsys,
    family,
    truth_args,
    expected_message,
):
    module = _strict_entry_module(family)
    calls = []

    def legacy(argv):
        calls.append(list(argv))
        return 91

    monkeypatch.setattr(module, "legacy_main", legacy, raising=False)
    with pytest.raises(SystemExit) as caught:
        module.main(
            [
                "--legacy-compatibility",
                "--config",
                "legacy.yaml",
                *truth_args,
            ]
        )
    message = (
        capsys.readouterr().err + str(caught.value)
    ).lower()
    assert expected_message in message
    assert calls == []


@pytest.mark.parametrize("family", ["baseline", "gsdiff"])
@pytest.mark.parametrize(
    "abbreviated_override",
    [
        ["--truth="],
        ["--truth-pa="],
        ["--truth-p", ""],
    ],
)
def test_legacy_parser_rejects_abbreviated_truth_override_before_io(
    tmp_path,
    monkeypatch,
    family,
    abbreviated_override,
):
    module = _strict_entry_module(family)
    output_dir = tmp_path / "output"
    argv = [
        "--legacy-compatibility",
        "--config",
        str(tmp_path / "must-not-read.yaml"),
        "--truth-path=truth.npz",
        *abbreviated_override,
        "--output-dir",
        str(output_dir),
    ]
    if family == "baseline":
        argv.extend(["--name", "legacy-test"])

    def forbidden_open(*args, **kwargs):
        raise AssertionError("abbreviated truth override reached config I/O")

    monkeypatch.setattr("builtins.open", forbidden_open)
    with pytest.raises(SystemExit):
        module.main(argv)
    assert not output_dir.exists()


def test_strict_method_pilot_readiness_remains_false_with_exact_blockers():
    pilot = yaml.safe_load(
        (REPO_ROOT / "configs" / "protocols" / "pilot-v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    diffusion = resolve_method_semantics(
        "gsdiff_diffusion",
        method_config_id="default",
        base_config={"gaussian_count": 1000},
        measurements_metadata={
            "H": 32,
            "W": 32,
            "T": 4,
            "K": 128,
            "holdout_K": 16,
        },
        execution_profile="publication-v1",
    )

    assert pilot["execution_ready"] is False
    assert pilot["method_budgets"] is None
    assert diffusion.execution_blockers == (
        "missing-reproducible-checkpoint-locator",
        "missing-checkpoint-training-provenance",
    )
