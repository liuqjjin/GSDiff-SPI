import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from dataclasses import replace

import pytest

from gsdiff.experiments.identity import canonical_json_bytes


REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_DATASETS_SCRIPT = (
    REPO_ROOT / "scripts" / "experiments" / "build_datasets.py"
)
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
    discovery_rechecks = 0
    real_recheck = cli.verify_dataset_directory_discovery

    def observing_recheck(discovery):
        nonlocal discovery_rechecks
        discovery_rechecks += 1
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
    assert discovery_rechecks == 2
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
    discovery_calls = 0
    real_discovery = cli.discover_dataset_directories

    def observing_discovery(path):
        nonlocal discovery_calls, final_discovery_seen
        discovery_calls += 1
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
    assert discovery_calls == 1
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
