from __future__ import annotations

import ast
from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

import pytest

from gsdiff.experiments.identity import canonical_json_bytes
from gsdiff.experiments.audit import validate_audit_log
from gsdiff.experiments.execution import (
    load_materialized_method_request,
    materialize_method_execution,
)
from gsdiff.experiments.methods import (
    CheckpointRequirement,
    ResolvedMethod,
    derive_algorithm_seed,
    resolve_method_semantics,
    thaw_json,
)


SOURCE_ROOT = Path(__file__).resolve().parents[2]
REGISTRY = SOURCE_ROOT / "configs" / "protocols" / "methods-v1.yaml"
PYTHON = Path(r"D:\conda\envs\spi\python.exe")
DATASET_IDENTITY_SHA256 = "1" * 64
TRUTH_SEEKING_CHILD = (
    SOURCE_ROOT
    / "tests"
    / "experiments"
    / "fixtures"
    / "truth_seeking_child.py"
)


def acquisition_spec() -> dict[str, object]:
    return {
        "schema_version": "blind-acquisition-spec-v1",
        "dimensions": {"H": 8, "W": 8, "T": 4, "K": 16, "holdout_K": 4},
        "acquisition": {
            "pattern_family": "bernoulli",
            "pattern_values": [0, 1],
            "pattern_order": "sequential",
            "time_assignment": "uniform",
            "holdout_pattern_family": "uniform-random",
            "noise_convention": "detector-absolute",
            "noise_sigma_absolute": 0.0,
        },
    }


def resolved_dgi() -> ResolvedMethod:
    return resolve_method_semantics(
        "dgi",
        method_config_id="smoke-default-v1",
        base_config={},
        measurements_metadata={"H": 8, "W": 8, "T": 4, "K": 16, "holdout_K": 4},
        execution_profile="controller-cpu-smoke-v1",
        registry_path=REGISTRY,
    )


def resolved_diffusion() -> ResolvedMethod:
    return resolve_method_semantics(
        "gsdiff_diffusion",
        method_config_id="smoke-default-v1",
        base_config={"gaussian_count": 1000},
        measurements_metadata={"H": 8, "W": 8, "T": 4, "K": 16, "holdout_K": 4},
        execution_profile="controller-cpu-smoke-v1",
        registry_path=REGISTRY,
    )


def algorithm_seed(method: ResolvedMethod | None = None):
    selected = resolved_dgi() if method is None else method
    return derive_algorithm_seed(
        cell_seed=7,
        dataset_identity_sha256=DATASET_IDENTITY_SHA256,
        method_id=selected.method_id,
        method_config_sha256=selected.method_config_sha256,
    )


def rehash_method(method: ResolvedMethod) -> ResolvedMethod:
    payload = {
        "method_id": method.method_id,
        "method_config_id": method.method_config_id,
        "execution_family": method.execution_family,
        "execution_profile": method.execution_profile,
        "command_template": list(method.command_template),
        "semantic_config": {
            key: value for key, value in method.semantic_config.items()
        },
        "checkpoint_requirements": [
            {
                "logical_id": item.logical_id,
                "sha256": item.sha256,
                "provenance_status": item.provenance_status,
            }
            for item in method.checkpoint_requirements
        ],
        "required_child_outputs": [
            "reconstruction.npz",
            "method-info.json",
        ],
        "profile_policy": {
            "publication_eligible": method.publication_eligible,
            "selection_eligible": method.selection_eligible,
            "promotion_eligible": method.promotion_eligible,
            "convergence_status": method.convergence_status,
            "execution_ready": method.execution_ready,
            "execution_blockers": list(method.execution_blockers),
        },
    }
    return replace(
        method,
        method_config_sha256=hashlib.sha256(
            canonical_json_bytes(payload)
        ).hexdigest(),
    )


def measurement_source(tmp_path: Path) -> tuple[Path, str]:
    path = tmp_path / "source-measurements.npz"
    path.write_bytes(b"blind-measurements-v1\x00payload")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def materialize(
    stage_root: Path,
    measurements_source: Path,
    measurements_sha256: str,
    *,
    method: ResolvedMethod | None = None,
    checkpoint_store: dict[str, Path] | None = None,
    requested_runtime_device: str = "cpu",
    source_root: Path = SOURCE_ROOT,
    python_executable: Path = PYTHON,
):
    selected = resolved_dgi() if method is None else method
    return materialize_method_execution(
        selected,
        stage_root=stage_root,
        measurements_source=measurements_source,
        measurements_file_sha256=measurements_sha256,
        dataset_identity_sha256=DATASET_IDENTITY_SHA256,
        expected_acquisition_spec=acquisition_spec(),
        algorithm_seed=algorithm_seed(selected),
        checkpoint_store={} if checkpoint_store is None else checkpoint_store,
        python_executable=python_executable,
        source_root=source_root,
        requested_runtime_device=requested_runtime_device,
    )


def test_materializer_rejects_non_windows_before_inspecting_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gsdiff.experiments import execution

    monkeypatch.setattr(execution, "os", SimpleNamespace(name="posix"))
    with pytest.raises(NotImplementedError, match="Windows-only"):
        execution.materialize_method_execution(
            object(),
            stage_root=object(),
            measurements_source=object(),
            measurements_file_sha256=object(),
            dataset_identity_sha256=object(),
            expected_acquisition_spec=object(),
            algorithm_seed=object(),
            checkpoint_store=object(),
            python_executable=object(),
            source_root=object(),
            requested_runtime_device=object(),
        )


def test_bootstrap_rejects_non_windows_before_parsing_or_reading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = importlib.util.spec_from_file_location(
        "tested_non_windows_method_child_bootstrap",
        SOURCE_ROOT
        / "scripts"
        / "experiments"
        / "method_child_bootstrap.py",
    )
    assert spec is not None and spec.loader is not None
    bootstrap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bootstrap)
    monkeypatch.setattr(bootstrap, "os", SimpleNamespace(name="posix"))

    with pytest.raises(NotImplementedError, match="Windows-only"):
        bootstrap.main()


def minimal_source_root(tmp_path: Path) -> Path:
    root = tmp_path / "minimal-source"
    for relative in (
        Path("gsdiff/__init__.py"),
        Path("train.py"),
        Path("scripts/run_baselines.py"),
        Path("scripts/experiments/method_child_bootstrap.py"),
        Path("schemas/method-info-v2.schema.json"),
    ):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((SOURCE_ROOT / relative).read_bytes())
    return root


def checkpoint_method_and_store(
    tmp_path: Path,
) -> tuple[ResolvedMethod, dict[str, Path]]:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint-loader-v1")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    base = resolved_dgi()
    method = rehash_method(
        replace(
            base,
            command_template=(
                *base.command_template,
                "--checkpoint",
                "${CHECKPOINT:model-v1}",
            ),
            checkpoint_requirements=(
                CheckpointRequirement(
                    logical_id="model-v1",
                    sha256=digest,
                    provenance_status="verified",
                ),
            ),
        )
    )
    return method, {"model-v1": checkpoint}


def rewrite_method_config(
    path: Path,
    mutation,
) -> dict[str, object]:
    document = json.loads(path.read_text(encoding="utf-8"))
    mutation(document)
    path.write_bytes(canonical_json_bytes(document))
    return document


def create_directory_reparse(
    link: Path,
    target: Path,
) -> None:
    if os.name != "nt":
        try:
            link.symlink_to(target, target_is_directory=True)
        except (OSError, NotImplementedError) as error:
            pytest.skip(f"directory symlink unavailable: {error}")
        return
    result = subprocess.run(
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
    if result.returncode != 0 or not os.path.lexists(link):
        pytest.skip(
            "directory junction unavailable: "
            + result.stdout.decode("utf-8", errors="replace")
        )
    attributes = getattr(os.lstat(link), "st_file_attributes", 0)
    assert attributes & 0x400


def test_materialization_identity_is_stable_across_absolute_stage_roots(
    tmp_path: Path,
) -> None:
    source, digest = measurement_source(tmp_path)
    first = materialize(tmp_path / "one" / "stage", source, digest)
    second = materialize(tmp_path / "two" / "stage", source, digest)

    first_config = json.loads(first.method_config_path.read_text(encoding="utf-8"))
    second_config = json.loads(second.method_config_path.read_text(encoding="utf-8"))
    assert first_config == second_config
    assert first_config["semantic"]["method_config_sha256"] == (
        second_config["semantic"]["method_config_sha256"]
    )
    assert first_config["semantic_sha256"] == second_config["semantic_sha256"]
    assert hashlib.sha256(first.method_config_path.read_bytes()).hexdigest() == (
        hashlib.sha256(second.method_config_path.read_bytes()).hexdigest()
    )
    assert first.argv != second.argv
    assert first.cwd != second.cwd
    assert first.materialization_record["logical"] == (
        second.materialization_record["logical"]
    )
    assert first.materialization_record["runtime"] != (
        second.materialization_record["runtime"]
    )


@pytest.mark.parametrize(
    ("requested", "visible", "cuda_visible"),
    [
        ("cpu", "cpu", None),
        ("cuda:0", "cuda:0", "0"),
        ("cuda:1", "cuda:0", "1"),
    ],
)
def test_materialized_device_mapping_is_exact(
    tmp_path: Path,
    requested: str,
    visible: str,
    cuda_visible: str | None,
) -> None:
    source, digest = measurement_source(tmp_path)
    execution = materialize(
        tmp_path / "stage",
        source,
        digest,
        requested_runtime_device=requested,
    )
    config = json.loads(execution.method_config_path.read_text(encoding="utf-8"))
    assert execution.requested_runtime_device == requested
    assert execution.child_runtime_device == visible
    assert config["runtime"]["requested_runtime_device"] == requested
    assert config["runtime"]["child_runtime_device"] == visible
    device_index = execution.argv.index("--device")
    assert execution.argv[device_index + 1] == visible
    if cuda_visible is None:
        assert "CUDA_VISIBLE_DEVICES" not in execution.env
    else:
        assert execution.env["CUDA_VISIBLE_DEVICES"] == cuda_visible


@pytest.mark.parametrize(
    "device",
    [
        "",
        "cuda",
        "cuda:",
        "cuda:-1",
        "cuda: 1",
        " cuda:1",
        "cuda:1 ",
        "CUDA:1",
        "cpu ",
        "cpu\n",
    ],
)
def test_materialized_device_rejects_malformed_value_before_staging(
    tmp_path: Path,
    device: str,
) -> None:
    source, digest = measurement_source(tmp_path)
    stage = tmp_path / "stage"
    with pytest.raises(ValueError, match="device"):
        materialize(
            stage,
            source,
            digest,
            requested_runtime_device=device,
        )
    assert not stage.exists()


def test_materialized_private_tree_and_parent_owned_logs_are_exact(
    tmp_path: Path,
) -> None:
    source, digest = measurement_source(tmp_path)
    execution = materialize(tmp_path / "stage", source, digest)
    stage = tmp_path / "stage"

    assert {entry.name for entry in stage.iterdir()} == {
        "input",
        "config",
        "checkpoints",
        "code",
        "work",
        "child-output",
        "parent",
    }
    assert {entry.name for entry in (stage / "parent").iterdir()} == {
        "audit",
        "logs",
    }
    assert {entry.name for entry in (stage / "parent" / "audit").iterdir()} == {
        "policy.json",
        "file-opens.jsonl",
    }
    assert {entry.name for entry in (stage / "parent" / "logs").iterdir()} == {
        "stdout.log",
        "stderr.log",
    }
    assert execution.audit_log_path.read_bytes() == b""
    assert execution.stdout_path.read_bytes() == b""
    assert execution.stderr_path.read_bytes() == b""
    assert execution.audit_log_path.parent != execution.child_output_dir
    assert execution.stdout_path.parent != execution.child_output_dir
    assert execution.stderr_path.parent != execution.child_output_dir


def test_materialized_source_snapshot_is_the_strict_python_closure(
    tmp_path: Path,
) -> None:
    source, digest = measurement_source(tmp_path)
    execution = materialize(tmp_path / "stage", source, digest)
    code = execution.cwd
    inventory = {
        path.relative_to(code).as_posix()
        for path in code.rglob("*")
        if path.is_file()
    }

    assert "gsdiff/experiments/execution.py" in inventory
    assert "gsdiff/experiments/audit.py" in inventory
    assert "scripts/experiments/method_child_bootstrap.py" in inventory
    assert "scripts/run_baselines.py" in inventory
    assert "train.py" in inventory
    assert "schemas/method-info-v2.schema.json" in inventory
    assert not any(path.startswith("gsdiff/evaluation/") for path in inventory)
    assert "gsdiff/baselines/_evaluation.py" not in inventory
    assert "gsdiff/data/_artifact_truth.py" not in inventory
    assert not any(
        component in {
            "data",
            "results",
            "checkpoints",
            ".git",
            ".claude",
            ".superpowers",
            "__pycache__",
        }
        for path in inventory
        for component in Path(path).parts[:-1]
        if not path.startswith("gsdiff/data/")
    )
    assert not any(path.endswith((".pyc", ".pyo")) for path in inventory)


def test_materialized_measurements_and_checkpoints_are_copied_and_hashed(
    tmp_path: Path,
) -> None:
    source, digest = measurement_source(tmp_path)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint-v1")
    checkpoint_digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    base = resolved_dgi()
    method = rehash_method(
        replace(
            base,
            command_template=(
                *base.command_template,
                "--checkpoint",
                "${CHECKPOINT:model-v1}",
            ),
            checkpoint_requirements=(
                CheckpointRequirement(
                    logical_id="model-v1",
                    sha256=checkpoint_digest,
                    provenance_status="verified",
                ),
            ),
        )
    )
    execution = materialize(
        tmp_path / "stage",
        source,
        digest,
        method=method,
        checkpoint_store={"model-v1": checkpoint},
    )
    config = json.loads(execution.method_config_path.read_text(encoding="utf-8"))
    staged_checkpoint = (
        execution.method_config_path.parents[1]
        / config["runtime"]["checkpoints"]["model-v1"]["path"]
    )

    assert execution.measurements_path.read_bytes() == source.read_bytes()
    assert execution.measurements_path != source
    assert staged_checkpoint.read_bytes() == checkpoint.read_bytes()
    execution.measurements_path.write_bytes(b"changed-copy")
    staged_checkpoint.write_bytes(b"changed-copy")
    assert source.read_bytes() == b"blind-measurements-v1\x00payload"
    assert checkpoint.read_bytes() == b"checkpoint-v1"


def test_materialized_real_diffusion_checkpoint_assignment_is_exact(
    tmp_path: Path,
) -> None:
    source, digest = measurement_source(tmp_path)
    method = resolved_diffusion()
    execution = materialize(
        tmp_path / "stage",
        source,
        digest,
        method=method,
        checkpoint_store={
            "gsdiff-diffusion-prior-v1": (
                SOURCE_ROOT / "checkpoints" / "diffusion_prior.pt"
            )
        },
    )
    checkpoint_argument = next(
        argument
        for argument in execution.argv
        if argument.startswith("gsdiff-diffusion-prior-v1=")
    )
    logical_id, staged_path = checkpoint_argument.split("=", 1)
    assert logical_id == "gsdiff-diffusion-prior-v1"
    assert Path(staged_path).parent == (
        tmp_path / "stage" / "checkpoints"
    ).resolve()
    assert "${" not in checkpoint_argument


@pytest.mark.parametrize(
    "bad_assignment",
    [
        "wrong=${CHECKPOINT:gsdiff-diffusion-prior-v1}",
        "gsdiff-diffusion-prior-v1=prefix-${CHECKPOINT:gsdiff-diffusion-prior-v1}",
        "prefix-gsdiff-diffusion-prior-v1=${CHECKPOINT:gsdiff-diffusion-prior-v1}",
        "gsdiff-diffusion-prior-v1=${CHECKPOINT:gsdiff-diffusion-prior-v1}-suffix",
    ],
)
def test_materialized_rejects_nonexact_checkpoint_assignment(
    tmp_path: Path,
    bad_assignment: str,
) -> None:
    source, digest = measurement_source(tmp_path)
    base = resolved_diffusion()
    method = rehash_method(
        replace(
            base,
            command_template=(
                *base.command_template[:-1],
                bad_assignment,
            ),
        )
    )
    stage = tmp_path / "stage"
    with pytest.raises(ValueError, match="token|checkpoint"):
        materialize(
            stage,
            source,
            digest,
            method=method,
            checkpoint_store={
                "gsdiff-diffusion-prior-v1": (
                    SOURCE_ROOT / "checkpoints" / "diffusion_prior.pt"
                )
            },
        )
    assert not stage.exists()


def test_materialized_rejects_measurement_source_toctou(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gsdiff.experiments import execution

    source, digest = measurement_source(tmp_path)
    original = execution._sha256_regular_file
    calls = 0

    def mutate_after_first_hash(path: Path, *, noun: str) -> str:
        nonlocal calls
        observed = original(path, noun=noun)
        if path == source and calls == 0:
            source.write_bytes(b"replacement")
        calls += 1
        return observed

    monkeypatch.setattr(execution, "_sha256_regular_file", mutate_after_first_hash)
    with pytest.raises(ValueError, match="changed|hash"):
        materialize(tmp_path / "stage", source, digest)


def test_materialized_rejects_nonempty_stage_and_blocked_method(
    tmp_path: Path,
) -> None:
    source, digest = measurement_source(tmp_path)
    nonempty = tmp_path / "nonempty"
    nonempty.mkdir()
    (nonempty / "user-file").write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        materialize(nonempty, source, digest)
    assert (nonempty / "user-file").read_text(encoding="utf-8") == "keep"

    blocker = "missing-versioned-native-budget"
    blocked = replace(
        resolved_dgi(),
        execution_ready=False,
        execution_blockers=(blocker,),
    )
    stage = tmp_path / "blocked"
    with pytest.raises(ValueError, match=blocker):
        materialize(stage, source, digest, method=blocked)
    assert not stage.exists()


def test_materialized_rejects_linked_stage_ancestry_when_supported(
    tmp_path: Path,
) -> None:
    source, digest = measurement_source(tmp_path)
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    try:
        linked.symlink_to(real, target_is_directory=True)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"directory symlinks unavailable: {error}")
    with pytest.raises(ValueError, match="symlink|reparse|linked"):
        materialize(linked / "stage", source, digest)
    assert not (real / "stage").exists()


def test_materialized_rejects_reparse_inside_source_closure_before_staging(
    tmp_path: Path,
) -> None:
    source, digest = measurement_source(tmp_path)
    source_root = minimal_source_root(tmp_path / "source")
    outside = tmp_path / "outside-source"
    outside.mkdir()
    (outside / "leak.py").write_text("SECRET = True\n", encoding="utf-8")
    reparse = source_root / "gsdiff" / "linked-package"
    create_directory_reparse(reparse, outside)
    stage = tmp_path / "stage"
    try:
        with pytest.raises(ValueError, match="source|symlink|reparse|linked"):
            materialize(
                stage,
                source,
                digest,
                source_root=source_root,
            )
        assert not stage.exists()
    finally:
        if os.path.lexists(reparse):
            reparse.rmdir()


@pytest.mark.parametrize(
    "bad_token",
    ["${HOME}", "${CHECKPOINT:undeclared}", "prefix-${DEVICE}", "${DEVICE}-suffix"],
)
def test_materialized_replaces_only_exact_approved_tokens(
    tmp_path: Path,
    bad_token: str,
) -> None:
    source, digest = measurement_source(tmp_path)
    method = rehash_method(
        replace(
            resolved_dgi(),
            command_template=(*resolved_dgi().command_template, bad_token),
        )
    )
    stage = tmp_path / "stage"
    with pytest.raises(ValueError, match="token|checkpoint"):
        materialize(stage, source, digest, method=method)
    assert not stage.exists()


def test_materialized_environment_is_fresh_and_stage_scoped(
    tmp_path: Path,
) -> None:
    source, digest = measurement_source(tmp_path)
    execution = materialize(tmp_path / "stage", source, digest)
    allowed = {
        "SYSTEMROOT",
        "WINDIR",
        "PATH",
        "TEMP",
        "TMP",
        "HOME",
        "USERPROFILE",
        "XDG_CACHE_HOME",
        "TORCH_HOME",
        "MPLCONFIGDIR",
    }
    assert set(execution.env) == allowed
    assert "PYTHONPATH" not in execution.env
    assert str(SOURCE_ROOT).lower() not in execution.env["PATH"].lower()
    for name in (
        "TEMP",
        "TMP",
        "HOME",
        "USERPROFILE",
        "XDG_CACHE_HOME",
        "TORCH_HOME",
        "MPLCONFIGDIR",
    ):
        Path(execution.env[name]).relative_to(tmp_path / "stage" / "work")


def test_fresh_environment_rejects_reparse_runtime_path_entry(
    tmp_path: Path,
) -> None:
    from gsdiff.experiments import execution

    runtime = tmp_path / "runtime"
    runtime.mkdir()
    outside = tmp_path / "outside-runtime"
    outside.mkdir()
    scripts = runtime / "Scripts"
    create_directory_reparse(scripts, outside)
    work_root = tmp_path / "work"
    work_directories = {
        name: work_root / name
        for name in ("tmp", "home", "xdg-cache", "torch", "matplotlib")
    }
    for directory in work_directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    system_root = execution._windows_system_root()
    system32 = (
        execution._resolved_real_directory(
            system_root / "System32",
            noun="test System32 directory",
        )
        if os.name == "nt"
        else system_root
    )
    try:
        with pytest.raises(ValueError, match="runtime|PATH|reparse|linked"):
            execution._fresh_child_environment(
                runtime_root=runtime,
                system_root=system_root,
                system32=system32,
                work_directories=work_directories,
                physical_cuda=None,
            )
    finally:
        if os.path.lexists(scripts):
            scripts.rmdir()


def test_fresh_environment_omits_missing_optional_runtime_path_entries(
    tmp_path: Path,
) -> None:
    from gsdiff.experiments import execution

    runtime = tmp_path / "runtime"
    runtime.mkdir()
    work_directories = {
        name: tmp_path / "work" / name
        for name in ("tmp", "home", "xdg-cache", "torch", "matplotlib")
    }
    for directory in work_directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    system_root = execution._windows_system_root()
    system32 = (
        execution._resolved_real_directory(
            system_root / "System32",
            noun="test System32 directory",
        )
        if os.name == "nt"
        else system_root
    )
    environment = execution._fresh_child_environment(
        runtime_root=runtime,
        system_root=system_root,
        system32=system32,
        work_directories=work_directories,
        physical_cuda=None,
    )
    assert environment["PATH"].split(os.pathsep) == [
        str(runtime.resolve()),
        str(system32.resolve()),
    ]


def test_materialized_bootstrap_argv_uses_isolated_no_site_mode(
    tmp_path: Path,
) -> None:
    source, digest = measurement_source(tmp_path)
    execution = materialize(tmp_path / "stage", source, digest)
    assert execution.argv[:6] == (
        str(PYTHON.resolve()),
        "-I",
        "-S",
        "-B",
        "-X",
        "utf8",
    )
    assert execution.argv[6].endswith(
        r"scripts\experiments\method_child_bootstrap.py"
    )
    assert execution.argv[7:13:2] == (
        "--policy",
        "--code-root",
        "--entrypoint",
    )
    assert execution.argv[13] == "--"
    assert execution.cwd == (tmp_path / "stage" / "code").resolve()


def test_materialized_copy_destination_injection_cannot_modify_external_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gsdiff.experiments import execution

    source, digest = measurement_source(tmp_path)
    outside = tmp_path / "outside-sentinel"
    sentinel = b"external-bytes-must-not-change"
    outside.write_bytes(sentinel)
    original = execution._copy_exact_file
    injected = False

    def inject_hardlink(
        copy_source: Path,
        destination: Path,
        *,
        expected_sha256: str | None,
        noun: str,
        parent_identity,
    ) -> str:
        nonlocal injected
        if destination.name == "measurements.npz" and not injected:
            os.link(outside, destination)
            injected = True
        return original(
            copy_source,
            destination,
            expected_sha256=expected_sha256,
            noun=noun,
            parent_identity=parent_identity,
        )

    monkeypatch.setattr(execution, "_copy_exact_file", inject_hardlink)
    with pytest.raises(ValueError, match="exist|exclusive|destination|linked"):
        materialize(tmp_path / "stage", source, digest)
    assert injected
    assert outside.read_bytes() == sentinel


@pytest.mark.parametrize("document_kind", ["config", "policy"])
def test_materialized_metadata_destination_injection_cannot_modify_external_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    document_kind: str,
) -> None:
    from gsdiff.experiments import execution

    source, digest = measurement_source(tmp_path)
    stage = tmp_path / "stage"
    outside = tmp_path / f"{document_kind}-sentinel"
    sentinel = f"{document_kind}-external-bytes".encode("ascii")
    outside.write_bytes(sentinel)
    original = execution.canonical_json_bytes
    injected = False

    def inject_before_write(value: object) -> bytes:
        nonlocal injected
        raw = original(value)
        if (
            type(value) is dict
            and value.get("schema")
            == (
                "materialized-method-config-v1"
                if document_kind == "config"
                else "method-audit-policy-v1"
            )
            and not injected
        ):
            destination = (
                stage / "config" / "method-config.json"
                if document_kind == "config"
                else stage / "parent" / "audit" / "policy.json"
            )
            os.link(outside, destination)
            injected = True
        return raw

    monkeypatch.setattr(execution, "canonical_json_bytes", inject_before_write)
    with pytest.raises(ValueError, match="exist|exclusive|destination|linked"):
        materialize(stage, source, digest)
    assert injected
    assert outside.read_bytes() == sentinel


def test_materialized_parent_logs_must_be_exclusively_created(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gsdiff.experiments import execution

    source, digest = measurement_source(tmp_path)
    stage = tmp_path / "stage"
    sentinels = {
        name: tmp_path / f"outside-{name}"
        for name in ("audit", "stdout", "stderr")
    }
    for name, path in sentinels.items():
        path.write_bytes(f"{name}-sentinel".encode("ascii"))
    original = execution.canonical_json_bytes
    injected = False

    def inject_before_log_creation(value: object) -> bytes:
        nonlocal injected
        raw = original(value)
        if (
            type(value) is dict
            and value.get("schema") == "method-audit-policy-v1"
            and not injected
        ):
            os.link(
                sentinels["audit"],
                stage / "parent" / "audit" / "file-opens.jsonl",
            )
            os.link(
                sentinels["stdout"],
                stage / "parent" / "logs" / "stdout.log",
            )
            os.link(
                sentinels["stderr"],
                stage / "parent" / "logs" / "stderr.log",
            )
            injected = True
        return raw

    monkeypatch.setattr(
        execution, "canonical_json_bytes", inject_before_log_creation
    )
    with pytest.raises(ValueError, match="exist|exclusive|destination|linked"):
        materialize(stage, source, digest)
    assert injected
    for name, path in sentinels.items():
        assert path.read_bytes() == f"{name}-sentinel".encode("ascii")


def test_materialized_detects_stage_directory_replacement_before_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gsdiff.experiments import execution

    source, digest = measurement_source(tmp_path)
    stage = tmp_path / "stage"
    original = execution._copy_exact_file
    replaced = False

    def replace_parent(
        copy_source: Path,
        destination: Path,
        *,
        expected_sha256: str | None,
        noun: str,
        parent_identity,
    ) -> str:
        nonlocal replaced
        if destination.name == "measurements.npz" and not replaced:
            displaced = stage / "displaced-input"
            destination.parent.rename(displaced)
            destination.parent.mkdir()
            replaced = True
        return original(
            copy_source,
            destination,
            expected_sha256=expected_sha256,
            noun=noun,
            parent_identity=parent_identity,
        )

    monkeypatch.setattr(execution, "_copy_exact_file", replace_parent)
    with pytest.raises(ValueError, match="stage|directory|identity|changed"):
        materialize(stage, source, digest)
    assert replaced


def test_materialized_detects_existing_stage_root_replacement_before_pin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gsdiff.experiments import execution

    source, digest = measurement_source(tmp_path)
    stage = tmp_path / "stage"
    stage.mkdir()
    original = execution._validate_stage_root_candidate
    replaced = False

    def replace_after_initial_observation(stage_root: Path) -> Path:
        nonlocal replaced
        result = original(stage_root)
        if not replaced:
            stage.rename(tmp_path / "displaced-stage")
            stage.mkdir()
            replaced = True
        return result

    monkeypatch.setattr(
        execution,
        "_validate_stage_root_candidate",
        replace_after_initial_observation,
    )
    with pytest.raises(ValueError, match="stage|identity|changed|replaced"):
        materialize(stage, source, digest)
    assert replaced


def test_materialized_checkpoint_physical_names_are_win32_collision_free(
    tmp_path: Path,
) -> None:
    source, digest = measurement_source(tmp_path)
    logical_ids = ("model", "MODEL", "x.", "CON")
    requirements: list[CheckpointRequirement] = []
    checkpoint_store: dict[str, Path] = {}
    for index, logical_id in enumerate(logical_ids):
        checkpoint = tmp_path / f"source-checkpoint-{index}"
        checkpoint.write_bytes(f"checkpoint-{logical_id}".encode("ascii"))
        checkpoint_digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        requirements.append(
            CheckpointRequirement(
                logical_id=logical_id,
                sha256=checkpoint_digest,
                provenance_status="verified",
            )
        )
        checkpoint_store[logical_id] = checkpoint
    base = resolved_dgi()
    method = rehash_method(
        replace(
            base,
            command_template=(
                *base.command_template,
                *(
                    token
                    for logical_id in logical_ids
                    for token in (
                        "--checkpoint",
                        f"${{CHECKPOINT:{logical_id}}}",
                    )
                ),
            ),
            checkpoint_requirements=tuple(requirements),
        )
    )
    execution = materialize(
        tmp_path / "stage",
        source,
        digest,
        method=method,
        checkpoint_store=checkpoint_store,
    )
    config = json.loads(execution.method_config_path.read_text(encoding="utf-8"))
    runtime = config["runtime"]["checkpoints"]
    physical_paths = {
        logical_id: Path(runtime[logical_id]["path"])
        for logical_id in logical_ids
    }
    assert len(
        {
            os.path.normcase(str(path)).rstrip(" .")
            for path in physical_paths.values()
        }
    ) == len(logical_ids)
    for logical_id, path in physical_paths.items():
        assert re.fullmatch(r"[0-9a-f]{64}\.checkpoint", path.name)
        staged = execution.method_config_path.parents[1] / path
        assert staged.read_bytes() == checkpoint_store[logical_id].read_bytes()


def test_materialized_rehashes_all_staged_inputs_before_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gsdiff.experiments import execution

    source, digest = measurement_source(tmp_path)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint-v1")
    checkpoint_digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    base = resolved_dgi()
    method = rehash_method(
        replace(
            base,
            command_template=(
                *base.command_template,
                "--checkpoint",
                "${CHECKPOINT:model-v1}",
            ),
            checkpoint_requirements=(
                CheckpointRequirement(
                    logical_id="model-v1",
                    sha256=checkpoint_digest,
                    provenance_status="verified",
                ),
            ),
        )
    )
    stage = tmp_path / "stage"
    original = execution._fresh_child_environment
    tampered = False

    def tamper_after_materialization(**kwargs):
        nonlocal tampered
        environment = original(**kwargs)
        staged_checkpoint = next((stage / "checkpoints").iterdir())
        staged_checkpoint.write_bytes(b"tampered-after-copy")
        tampered = True
        return environment

    monkeypatch.setattr(
        execution, "_fresh_child_environment", tamper_after_materialization
    )
    with pytest.raises(ValueError, match="checkpoint|hash|changed"):
        materialize(
            stage,
            source,
            digest,
            method=method,
            checkpoint_store={"model-v1": checkpoint},
        )
    assert tampered


@pytest.mark.parametrize("action", ["add", "replace"])
def test_materialized_revalidates_final_source_closure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    action: str,
) -> None:
    from gsdiff.experiments import execution

    source, digest = measurement_source(tmp_path)
    source_root = minimal_source_root(tmp_path)
    original = execution._fresh_child_environment
    mutated = False

    def mutate_source_after_copy(**kwargs):
        nonlocal mutated
        environment = original(**kwargs)
        if action == "add":
            (source_root / "gsdiff" / "late.py").write_text(
                "LATE = True\n", encoding="utf-8"
            )
        else:
            (source_root / "gsdiff" / "__init__.py").write_text(
                "MUTATED = True\n", encoding="utf-8"
            )
        mutated = True
        return environment

    monkeypatch.setattr(
        execution, "_fresh_child_environment", mutate_source_after_copy
    )
    with pytest.raises(ValueError, match="source|closure|inventory|changed"):
        materialize(
            tmp_path / "stage",
            source,
            digest,
            source_root=source_root,
        )
    assert mutated


@pytest.mark.parametrize("action", ["extra", "tamper"])
def test_materialized_revalidates_exact_staged_code_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    action: str,
) -> None:
    from gsdiff.experiments import execution

    source, digest = measurement_source(tmp_path)
    stage = tmp_path / "stage"
    original = execution._fresh_child_environment
    mutated = False

    def mutate_staged_code(**kwargs):
        nonlocal mutated
        environment = original(**kwargs)
        if action == "extra":
            (stage / "code" / "unexpected.py").write_text(
                "UNEXPECTED = True\n", encoding="utf-8"
            )
        else:
            staged = stage / "code" / "gsdiff" / "__init__.py"
            staged.write_text("TAMPERED = True\n", encoding="utf-8")
        mutated = True
        return environment

    monkeypatch.setattr(
        execution, "_fresh_child_environment", mutate_staged_code
    )
    with pytest.raises(ValueError, match="source|code|inventory|hash|extra"):
        materialize(stage, source, digest)
    assert mutated


def test_materialized_detects_same_name_replacement_after_final_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gsdiff.experiments import execution

    source, digest = measurement_source(tmp_path)
    stage = tmp_path / "stage"
    target = stage / "code" / "gsdiff" / "__init__.py"
    original = execution._sha256_regular_file
    replaced = False

    def replace_after_hash(path: Path, *, noun: str) -> str:
        nonlocal replaced
        result = original(path, noun=noun)
        if (
            not replaced
            and noun.startswith("staged file ")
            and execution._same_lexical_path(path, target)
        ):
            path.unlink()
            path.write_bytes(b"same-name-replacement")
            replaced = True
        return result

    monkeypatch.setattr(
        execution, "_sha256_regular_file", replace_after_hash
    )
    with pytest.raises(ValueError, match="stage|identity|changed|hash"):
        materialize(stage, source, digest)
    assert replaced


def test_materialized_rejects_config_tampering_before_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gsdiff.experiments import execution

    source, digest = measurement_source(tmp_path)
    stage = tmp_path / "stage"
    original = execution._fresh_child_environment

    def tamper_config(**kwargs):
        environment = original(**kwargs)
        (stage / "config" / "method-config.json").write_bytes(b"{}")
        return environment

    monkeypatch.setattr(execution, "_fresh_child_environment", tamper_config)
    with pytest.raises(ValueError, match="config|hash|changed|bytes"):
        materialize(stage, source, digest)


def test_materialized_rejects_fake_existing_windows_system_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if os.name != "nt":
        pytest.skip("authoritative Windows-directory check is Windows-only")
    from gsdiff.experiments import execution

    fake = tmp_path / "fake-windows"
    (fake / "System32").mkdir(parents=True)
    monkeypatch.setenv("SYSTEMROOT", str(fake))
    monkeypatch.setenv("WINDIR", str(fake))
    with pytest.raises(ValueError, match="Windows|SystemRoot|authoritative"):
        execution._windows_system_root()


@pytest.mark.parametrize(
    "field",
    ["semantic", "command", "config_id", "profile"],
)
def test_materialized_method_identity_rejects_absolute_path_literals(
    tmp_path: Path,
    field: str,
) -> None:
    source, digest = measurement_source(tmp_path)
    base = resolved_dgi()
    absolute = str((tmp_path / "secret").resolve())
    if field == "semantic":
        changed = replace(base, semantic_config={"secret_path": absolute})
    elif field == "command":
        changed = replace(
            base,
            command_template=(*base.command_template, f"--secret={absolute}"),
        )
    elif field == "config_id":
        changed = replace(
            base,
            requested_method_config_id=absolute,
            method_config_id=absolute,
        )
    else:
        changed = replace(base, execution_profile=absolute)
    method = rehash_method(changed)
    with pytest.raises(ValueError, match="absolute|path-free|path"):
        materialize(tmp_path / "stage", source, digest, method=method)


@pytest.mark.parametrize(
    "literal",
    [
        r"checkpoint stored at D:\secret\weights.pt",
        r"checkpoint stored at \\server\share\weights.pt",
        r"checkpoint stored at \\?\D:\secret\weights.pt",
        "checkpoint stored at /var/secret/weights.pt",
        "checkpoint,/var/secret/weights.pt",
        "checkpoint;/var/secret/weights.pt",
    ],
)
def test_materialized_method_identity_rejects_embedded_absolute_paths(
    tmp_path: Path,
    literal: str,
) -> None:
    source, digest = measurement_source(tmp_path)
    method = rehash_method(
        replace(
            resolved_dgi(),
            semantic_config={"description": literal},
        )
    )
    with pytest.raises(ValueError, match="absolute|path-free|path"):
        materialize(tmp_path / "stage", source, digest, method=method)


def test_source_exclusions_are_win32_case_insensitive() -> None:
    from gsdiff.experiments import execution

    assert execution._is_excluded_source(
        Path("GSDIFF/EVALUATION"), is_directory=True
    )
    assert execution._is_excluded_source(
        Path("GSDIFF/DATA/_ARTIFACT_TRUTH.PY"), is_directory=False
    )


def test_materialized_snapshots_mutable_semantic_config_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gsdiff.experiments import execution

    source, digest = measurement_source(tmp_path)
    mutable_config: dict[str, object] = {"iterations": 1}
    method = rehash_method(
        replace(resolved_dgi(), semantic_config=mutable_config)
    )
    original = execution._validate_stage_root_candidate
    mutated = False

    def mutate_after_validation(stage_root: Path) -> Path:
        nonlocal mutated
        result = original(stage_root)
        mutable_config["iterations"] = 2
        mutated = True
        return result

    monkeypatch.setattr(
        execution, "_validate_stage_root_candidate", mutate_after_validation
    )
    execution_result = materialize(
        tmp_path / "stage", source, digest, method=method
    )
    config = json.loads(
        execution_result.method_config_path.read_text(encoding="utf-8")
    )
    assert mutated
    assert config["semantic"]["semantic_config"] == {"iterations": 1}


@pytest.mark.parametrize("overlap", ["stage", "source"])
def test_materialized_rejects_inputs_inside_broad_runtime_read_root(
    tmp_path: Path,
    overlap: str,
) -> None:
    source, digest = measurement_source(tmp_path)
    runtime = tmp_path / "fake-runtime"
    runtime.mkdir()
    python = runtime / "python.exe"
    python.write_bytes(b"not-executed")
    if overlap == "stage":
        source_root = minimal_source_root(tmp_path / "outside")
        stage = runtime / "stage"
    else:
        source_root = minimal_source_root(runtime)
        stage = tmp_path / "stage"
    with pytest.raises(ValueError, match="overlap|runtime|read root|contain"):
        materialize(
            stage,
            source,
            digest,
            source_root=source_root,
            python_executable=python,
        )


def test_materialized_request_loader_returns_frozen_bound_request(
    tmp_path: Path,
) -> None:
    source, digest = measurement_source(tmp_path)
    method, checkpoint_store = checkpoint_method_and_store(tmp_path)
    execution = materialize(
        tmp_path / "stage",
        source,
        digest,
        method=method,
        checkpoint_store=checkpoint_store,
        requested_runtime_device="cuda:1",
    )

    request = load_materialized_method_request(
        execution.method_config_path
    )

    assert request.method == method
    assert request.algorithm_seed == algorithm_seed(method)
    assert request.dataset_identity_sha256 == DATASET_IDENTITY_SHA256
    assert request.measurements_file_sha256 == digest
    assert thaw_json(request.expected_acquisition_spec) == acquisition_spec()
    assert request.measurements_path == execution.measurements_path
    assert request.child_output_dir == execution.child_output_dir
    assert set(request.checkpoint_paths) == {"model-v1"}
    assert (
        request.checkpoint_paths["model-v1"].read_bytes()
        == checkpoint_store["model-v1"].read_bytes()
    )
    assert request.requested_runtime_device == "cuda:1"
    assert request.child_runtime_device == "cuda:0"
    with pytest.raises(TypeError):
        request.checkpoint_paths["other"] = tmp_path  # type: ignore[index]
    with pytest.raises(TypeError):
        request.expected_acquisition_spec["other"] = 1  # type: ignore[index]


def test_method_execution_interfaces_are_public() -> None:
    from gsdiff import experiments

    assert experiments.MaterializedMethodExecution is not None
    assert experiments.MaterializedMethodRequest is not None
    assert experiments.materialize_method_execution is (
        materialize_method_execution
    )
    assert experiments.load_materialized_method_request is (
        load_materialized_method_request
    )
    assert experiments.validate_audit_log is validate_audit_log


def test_train_matplotlib_import_is_compatibility_lazy() -> None:
    tree = ast.parse(
        (SOURCE_ROOT / "train.py").read_text(encoding="utf-8")
    )
    top_level_imports = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert not any(
        (
            isinstance(node, ast.Import)
            and any(
                alias.name == "matplotlib"
                or alias.name.startswith("matplotlib.")
                for alias in node.names
            )
        )
        or (
            isinstance(node, ast.ImportFrom)
            and node.module is not None
            and (
                node.module == "matplotlib"
                or node.module.startswith("matplotlib.")
            )
        )
        for node in top_level_imports
    )


def test_materialized_request_loader_rejects_duplicate_json_keys(
    tmp_path: Path,
) -> None:
    source, digest = measurement_source(tmp_path)
    execution = materialize(tmp_path / "stage", source, digest)
    original = execution.method_config_path.read_text(encoding="utf-8")
    execution.method_config_path.write_text(
        '{"schema":"materialized-method-config-v1",' + original[1:],
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_materialized_method_request(execution.method_config_path)


@pytest.mark.parametrize(
    "section",
    ["top", "semantic", "profile", "request", "seed", "runtime"],
)
def test_materialized_request_loader_rejects_extra_keys(
    tmp_path: Path,
    section: str,
) -> None:
    source, digest = measurement_source(tmp_path)
    execution = materialize(tmp_path / "stage", source, digest)

    def add_extra(document: dict[str, object]) -> None:
        target: dict[str, object]
        if section == "top":
            target = document
        elif section == "semantic":
            target = document["semantic"]  # type: ignore[assignment]
        elif section == "profile":
            target = document["semantic"]["profile_policy"]  # type: ignore[index,assignment]
        elif section == "request":
            target = document["request"]  # type: ignore[assignment]
        elif section == "seed":
            target = document["request"]["algorithm_seed"]  # type: ignore[index,assignment]
        else:
            target = document["runtime"]  # type: ignore[assignment]
        target["unexpected"] = True

    rewrite_method_config(execution.method_config_path, add_extra)
    with pytest.raises(ValueError, match="extra|keys|schema"):
        load_materialized_method_request(execution.method_config_path)


def test_materialized_request_loader_rejects_semantic_hash_mismatch(
    tmp_path: Path,
) -> None:
    source, digest = measurement_source(tmp_path)
    execution = materialize(tmp_path / "stage", source, digest)
    rewrite_method_config(
        execution.method_config_path,
        lambda document: document.__setitem__(
            "semantic_sha256", "0" * 64
        ),
    )
    with pytest.raises(ValueError, match="semantic|hash"):
        load_materialized_method_request(execution.method_config_path)


@pytest.mark.parametrize(
    "field",
    [
        "method_id",
        "method_config_id",
        "execution_profile",
        "method_config_sha256",
        "semantic_sha256",
    ],
)
def test_materialized_request_loader_rejects_request_crosslock_mismatch(
    tmp_path: Path,
    field: str,
) -> None:
    source, digest = measurement_source(tmp_path)
    execution = materialize(tmp_path / "stage", source, digest)
    replacements = {
        "method_id": "tv",
        "method_config_id": "wrong-config",
        "execution_profile": "wrong-profile",
        "method_config_sha256": "0" * 64,
        "semantic_sha256": "0" * 64,
    }

    def break_crosslock(document: dict[str, object]) -> None:
        document["request"][field] = replacements[field]  # type: ignore[index]

    rewrite_method_config(execution.method_config_path, break_crosslock)
    with pytest.raises(ValueError, match="request|semantic|method|profile|hash"):
        load_materialized_method_request(execution.method_config_path)


@pytest.mark.parametrize(
    "field",
    [
        "dataset_hash",
        "measurement_hash",
        "acquisition",
        "seed_digest",
        "seed_type",
        "seed_mismatch",
    ],
)
def test_materialized_request_loader_rejects_malformed_request_fields(
    tmp_path: Path,
    field: str,
) -> None:
    source, digest = measurement_source(tmp_path)
    execution = materialize(tmp_path / "stage", source, digest)

    def corrupt(document: dict[str, object]) -> None:
        request = document["request"]
        if field == "dataset_hash":
            request["dataset_identity_sha256"] = "A" * 64
        elif field == "measurement_hash":
            request["measurements_file_sha256"] = "not-a-hash"
        elif field == "acquisition":
            request["expected_acquisition_spec"] = {}
        elif field == "seed_digest":
            request["algorithm_seed"]["derivation_sha256"] = "1"
        elif field == "seed_type":
            request["algorithm_seed"]["seed_u32"] = True
        else:
            request["algorithm_seed"]["seed_u32"] ^= 1

    rewrite_method_config(execution.method_config_path, corrupt)
    with pytest.raises(
        ValueError,
        match="dataset|measurement|acquisition|seed|sha256|uint32|digest",
    ):
        load_materialized_method_request(execution.method_config_path)


@pytest.mark.parametrize(
    "case",
    ["ids", "record_hash", "file_hash", "path_mismatch", "extra_key"],
)
def test_materialized_request_loader_rejects_checkpoint_disagreement(
    tmp_path: Path,
    case: str,
) -> None:
    source, digest = measurement_source(tmp_path)
    method, checkpoint_store = checkpoint_method_and_store(tmp_path)
    execution = materialize(
        tmp_path / "stage",
        source,
        digest,
        method=method,
        checkpoint_store=checkpoint_store,
    )
    if case == "file_hash":
        staged = next((tmp_path / "stage" / "checkpoints").iterdir())
        staged.write_bytes(b"tampered-checkpoint")
    else:
        def corrupt(document: dict[str, object]) -> None:
            checkpoints = document["runtime"]["checkpoints"]
            if case == "ids":
                checkpoints["undeclared"] = checkpoints.pop("model-v1")
            elif case == "record_hash":
                checkpoints["model-v1"]["sha256"] = "0" * 64
            elif case == "path_mismatch":
                checkpoints["model-v1"]["path"] = "input/measurements.npz"
            else:
                checkpoints["model-v1"]["unexpected"] = True

        rewrite_method_config(execution.method_config_path, corrupt)
    with pytest.raises(
        ValueError,
        match="checkpoint|logical|hash|path|keys",
    ):
        load_materialized_method_request(execution.method_config_path)


@pytest.mark.parametrize("field", ["measurement", "output", "checkpoint"])
def test_materialized_request_loader_rejects_paths_outside_stage(
    tmp_path: Path,
    field: str,
) -> None:
    source, digest = measurement_source(tmp_path)
    method, checkpoint_store = checkpoint_method_and_store(tmp_path)
    execution = materialize(
        tmp_path / "stage",
        source,
        digest,
        method=method,
        checkpoint_store=checkpoint_store,
    )
    (tmp_path / "outside").mkdir()

    def escape(document: dict[str, object]) -> None:
        runtime = document["runtime"]
        if field == "measurement":
            runtime["measurements_path"] = "../source-measurements.npz"
        elif field == "output":
            runtime["child_output_dir"] = "../outside"
        else:
            runtime["checkpoints"]["model-v1"]["path"] = str(
                checkpoint_store["model-v1"].resolve()
            )

    rewrite_method_config(execution.method_config_path, escape)
    with pytest.raises(ValueError, match="path|stage|relative|escape"):
        load_materialized_method_request(execution.method_config_path)


def test_materialized_request_loader_rejects_absolute_semantic_path(
    tmp_path: Path,
) -> None:
    source, digest = measurement_source(tmp_path)
    execution = materialize(tmp_path / "stage", source, digest)
    absolute = str((tmp_path / "secret").resolve())

    def inject_absolute(document: dict[str, object]) -> None:
        document["semantic"]["semantic_config"]["absolute_path"] = absolute
        semantic_hash = hashlib.sha256(
            canonical_json_bytes(document["semantic"])
        ).hexdigest()
        document["semantic_sha256"] = semantic_hash
        document["request"]["semantic_sha256"] = semantic_hash

    rewrite_method_config(execution.method_config_path, inject_absolute)
    with pytest.raises(ValueError, match="absolute|path-free|path"):
        load_materialized_method_request(execution.method_config_path)


def run_audited_child(
    execution,
    *,
    action: str,
    target: Path | str | None = None,
    target2: Path | str | None = None,
    expect_denied: bool = False,
) -> subprocess.CompletedProcess[bytes]:
    entrypoint = execution.cwd / "scripts" / "run_baselines.py"
    entrypoint.write_bytes(TRUTH_SEEKING_CHILD.read_bytes())
    delimiter = execution.argv.index("--")
    child_arguments = ["--audit-action", action]
    if target is not None:
        child_arguments.extend(("--target", str(target)))
    if target2 is not None:
        child_arguments.extend(("--target2", str(target2)))
    if expect_denied:
        child_arguments.append("--expect-denied")
    argv = (*execution.argv[: delimiter + 1], *child_arguments)
    with (
        execution.stdout_path.open("wb") as stdout,
        execution.stderr_path.open("wb") as stderr,
    ):
        return subprocess.run(
            argv,
            cwd=execution.cwd,
            env=dict(execution.env),
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            shell=False,
            check=False,
            timeout=30,
        )


def load_audit_events(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8", errors="strict"
        ).splitlines()
    ]


@pytest.mark.parametrize("kind", ["measurements", "config", "code", "policy"])
def test_audit_allows_and_logs_declared_reads(
    tmp_path: Path,
    kind: str,
) -> None:
    source, digest = measurement_source(tmp_path)
    execution = materialize(tmp_path / "stage", source, digest)
    targets = {
        "measurements": execution.measurements_path,
        "config": execution.method_config_path,
        "code": execution.cwd / "gsdiff" / "__init__.py",
        "policy": execution.audit_policy_path,
    }
    result = run_audited_child(
        execution, action="read", target=targets[kind]
    )
    assert result.returncode == 0, execution.stderr_path.read_text(
        encoding="utf-8", errors="strict"
    )
    events = load_audit_events(execution.audit_log_path)
    assert any(
        event.get("operation") == "open"
        and event.get("decision") == "allow"
        and event.get("resolved_path")
        == str(targets[kind].resolve())
        for event in events
    )


@pytest.mark.parametrize("root_name", ["child-output", "work"])
def test_audit_allows_write_then_read_only_in_declared_roots(
    tmp_path: Path,
    root_name: str,
) -> None:
    source, digest = measurement_source(tmp_path)
    execution = materialize(tmp_path / "stage", source, digest)
    root = (
        execution.child_output_dir
        if root_name == "child-output"
        else execution.cwd.parent / "work"
    )
    target = root / "written.txt"
    result = run_audited_child(
        execution, action="write-read", target=target
    )
    assert result.returncode == 0, execution.stderr_path.read_text(
        encoding="utf-8", errors="strict"
    )
    assert target.read_text(encoding="utf-8", errors="strict") == "盲态验证"


@pytest.mark.parametrize(
    ("action", "target_kind", "expected_access"),
    [
        ("os-open-read", "measurements", "read"),
        ("os-open-write", "child-output", "write"),
    ],
)
def test_audit_authorizes_windows_low_level_os_open(
    tmp_path: Path,
    action: str,
    target_kind: str,
    expected_access: str,
) -> None:
    source, digest = measurement_source(tmp_path)
    execution = materialize(tmp_path / "stage", source, digest)
    target = (
        execution.measurements_path
        if target_kind == "measurements"
        else execution.child_output_dir / "low-level-write.txt"
    )

    result = run_audited_child(execution, action=action, target=target)

    assert result.returncode == 0, execution.stderr_path.read_text(
        encoding="utf-8", errors="strict"
    )
    if target_kind == "child-output":
        assert target.read_bytes() == "盲态验证".encode("utf-8")
    assert any(
        event.get("operation") == "open"
        and event.get("decision") == "allow"
        and event.get("access") == expected_access
        and event.get("resolved_path") == str(target.resolve())
        for event in load_audit_events(execution.audit_log_path)
    )
    validate_audit_log(
        execution.audit_log_path,
        expected_policy_sha256=execution.audit_policy_sha256,
    )


def test_audit_denies_windows_low_level_os_open_outside_write_roots(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside-low-level-write.txt"
    sentinel = b"outside-low-level-write-must-not-change"
    outside.write_bytes(sentinel)
    source, digest = measurement_source(tmp_path)
    execution = materialize(tmp_path / "stage", source, digest)

    result = run_audited_child(
        execution,
        action="os-open-write",
        target=outside,
        expect_denied=True,
    )

    assert result.returncode == 0, execution.stderr_path.read_text(
        encoding="utf-8", errors="strict"
    )
    assert outside.read_bytes() == sentinel
    with pytest.raises(ValueError, match="denied"):
        validate_audit_log(
            execution.audit_log_path,
            expected_policy_sha256=execution.audit_policy_sha256,
        )


def test_audit_denies_delete_on_close_for_exact_read_target(
    tmp_path: Path,
) -> None:
    if os.name != "nt" or not hasattr(os, "O_TEMPORARY"):
        pytest.skip("Windows O_TEMPORARY is unavailable")
    source, digest = measurement_source(tmp_path)
    execution = materialize(tmp_path / "stage", source, digest)
    target = execution.measurements_path
    before = target.read_bytes()

    result = run_audited_child(
        execution,
        action="os-open-temporary",
        target=target,
        expect_denied=True,
    )

    assert result.returncode == 0, execution.stderr_path.read_text(
        encoding="utf-8", errors="strict"
    )
    assert target.read_bytes() == before
    with pytest.raises(ValueError, match="denied"):
        validate_audit_log(
            execution.audit_log_path,
            expected_policy_sha256=execution.audit_policy_sha256,
        )


def test_v2_child_output_transaction_succeeds_under_real_audit_hook(
    tmp_path: Path,
) -> None:
    source, digest = measurement_source(tmp_path)
    execution = materialize(tmp_path / "stage", source, digest)

    result = run_audited_child(
        execution,
        action="write-v2-outputs",
        target=execution.child_output_dir,
        target2=execution.method_config_path,
    )

    assert result.returncode == 0, execution.stderr_path.read_text(
        encoding="utf-8", errors="strict"
    )
    assert {
        path.name for path in execution.child_output_dir.iterdir()
    } == {"method-info.json", "reconstruction.npz"}
    assert all(
        path.stat().st_size > 0
        for path in execution.child_output_dir.iterdir()
    )
    assert "V2-HASHES=" in execution.stdout_path.read_text(
        encoding="utf-8", errors="strict"
    )
    validate_audit_log(
        execution.audit_log_path,
        expected_policy_sha256=execution.audit_policy_sha256,
    )


def test_audit_rejects_preexisting_hardlink_leaf_before_external_write(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside-sentinel"
    sentinel = b"outside-hardlink-must-not-change"
    outside.write_bytes(sentinel)
    source, digest = measurement_source(tmp_path)
    execution = materialize(tmp_path / "stage", source, digest)
    injected = execution.child_output_dir / "injected-hardlink"
    os.link(outside, injected)

    result = run_audited_child(
        execution,
        action="write-read",
        target=injected,
        expect_denied=True,
    )

    assert result.returncode == 0, execution.stderr_path.read_text(
        encoding="utf-8", errors="strict"
    )
    assert outside.read_bytes() == sentinel
    with pytest.raises(ValueError, match="denied"):
        validate_audit_log(
            execution.audit_log_path,
            expected_policy_sha256=execution.audit_policy_sha256,
        )


@pytest.mark.parametrize(
    ("action", "relative_target"),
    [
        ("read", "../evaluation-truth.npz"),
        ("listdir", ".."),
        ("scandir", ".."),
        ("chdir", ".."),
    ],
)
def test_audit_denies_lexical_upstream_access_even_when_child_catches_it(
    tmp_path: Path,
    action: str,
    relative_target: str,
) -> None:
    source, digest = measurement_source(tmp_path)
    execution = materialize(tmp_path / "stage", source, digest)
    target = execution.cwd / relative_target
    result = run_audited_child(
        execution,
        action=action,
        target=target,
        expect_denied=True,
    )
    assert result.returncode == 0
    assert b"DENIED-CAUGHT" in execution.stdout_path.read_bytes()
    with pytest.raises(ValueError, match="denied"):
        validate_audit_log(
            execution.audit_log_path,
            expected_policy_sha256=execution.audit_policy_sha256,
        )
    denied = [
        event
        for event in load_audit_events(execution.audit_log_path)
        if event.get("decision") == "deny"
    ]
    assert denied
    assert all(
        isinstance(event.get("sequence"), int)
        and isinstance(event.get("timestamp_utc"), str)
        and isinstance(event.get("operation"), str)
        and (
            "resolved_path" in event or "command_class" in event
        )
        for event in denied
    )


def test_audit_denies_known_absolute_truth_path(
    tmp_path: Path,
) -> None:
    truth = tmp_path / "known-evaluation-truth.npz"
    truth.write_bytes(b"secret")
    source, digest = measurement_source(tmp_path)
    execution = materialize(tmp_path / "stage", source, digest)
    result = run_audited_child(
        execution,
        action="read",
        target=truth,
        expect_denied=True,
    )
    assert result.returncode == 0
    denied = [
        event
        for event in load_audit_events(execution.audit_log_path)
        if event.get("decision") == "deny"
    ]
    assert any(
        event.get("resolved_path") == str(truth.resolve())
        for event in denied
    )


@pytest.mark.parametrize("action", ["subprocess", "system", "spawn"])
def test_audit_denies_all_nested_subprocess_families(
    tmp_path: Path,
    action: str,
) -> None:
    source, digest = measurement_source(tmp_path)
    execution = materialize(tmp_path / "stage", source, digest)
    result = run_audited_child(
        execution, action=action, expect_denied=True
    )
    assert result.returncode == 0
    denied = [
        event
        for event in load_audit_events(execution.audit_log_path)
        if event.get("decision") == "deny"
    ]
    assert any(
        isinstance(event.get("command_class"), str)
        for event in denied
    )


@pytest.mark.parametrize(
    "action",
    [
        "mkdir",
        "remove",
        "rmdir",
        "truncate",
        "chmod",
        "utime",
    ],
)
def test_audit_denies_mutations_outside_write_roots(
    tmp_path: Path,
    action: str,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "target"
    if action in {"remove", "truncate", "chmod", "utime"}:
        target.write_text("keep", encoding="utf-8")
    elif action == "rmdir":
        target.mkdir()
    source, digest = measurement_source(tmp_path)
    execution = materialize(tmp_path / "stage", source, digest)
    result = run_audited_child(
        execution,
        action=action,
        target=target,
        expect_denied=True,
    )
    assert result.returncode == 0
    assert target.exists() == (action != "mkdir")


@pytest.mark.parametrize(
    ("action", "operation"),
    [
        ("audit-chown", "os.chown"),
        ("audit-chflags", "os.chflags"),
        ("audit-setxattr", "os.setxattr"),
        ("audit-removexattr", "os.removexattr"),
        ("audit-mknod", "os.mknod"),
        (
            "audit-unknown-mutation",
            "os.future_filesystem_mutation",
        ),
    ],
)
def test_audit_denies_extended_and_unknown_mutation_events_outside_roots(
    tmp_path: Path,
    action: str,
    operation: str,
) -> None:
    outside = tmp_path / "outside-extended-mutation"
    outside.write_bytes(b"must-remain")
    source, digest = measurement_source(tmp_path)
    execution = materialize(tmp_path / "stage", source, digest)

    result = run_audited_child(
        execution,
        action=action,
        target=outside,
        expect_denied=True,
    )

    assert result.returncode == 0, execution.stderr_path.read_text(
        encoding="utf-8", errors="strict"
    )
    assert outside.read_bytes() == b"must-remain"
    assert any(
        event.get("operation") == operation
        and event.get("decision") == "deny"
        for event in load_audit_events(execution.audit_log_path)
    )


@pytest.mark.parametrize(
    "action",
    ["audit-chown-dir-fd", "audit-mknod-dir-fd"],
)
def test_audit_denies_extended_mutation_directory_descriptors(
    tmp_path: Path,
    action: str,
) -> None:
    outside = tmp_path / "outside-dir-fd"
    outside.write_bytes(b"must-remain")
    source, digest = measurement_source(tmp_path)
    execution = materialize(tmp_path / "stage", source, digest)

    result = run_audited_child(
        execution,
        action=action,
        target=outside,
        expect_denied=True,
    )

    assert result.returncode == 0, execution.stderr_path.read_text(
        encoding="utf-8", errors="strict"
    )
    assert outside.read_bytes() == b"must-remain"


@pytest.mark.parametrize(
    "action",
    [
        "audit-chown-fd",
        "audit-setxattr-fd",
        "audit-removexattr-fd",
    ],
)
def test_audit_denies_extended_mutation_file_descriptors(
    tmp_path: Path,
    action: str,
) -> None:
    source, digest = measurement_source(tmp_path)
    execution = materialize(tmp_path / "stage", source, digest)
    before = execution.measurements_path.read_bytes()

    result = run_audited_child(
        execution,
        action=action,
        target=execution.measurements_path,
        expect_denied=True,
    )

    assert result.returncode == 0, execution.stderr_path.read_text(
        encoding="utf-8", errors="strict"
    )
    assert execution.measurements_path.read_bytes() == before


@pytest.mark.parametrize(
    "action",
    [
        "audit-chown-malformed",
        "audit-chflags-malformed",
        "audit-setxattr-malformed",
        "audit-removexattr-malformed",
    ],
)
def test_audit_denies_malformed_extended_mutation_events(
    tmp_path: Path,
    action: str,
) -> None:
    source, digest = measurement_source(tmp_path)
    execution = materialize(tmp_path / "stage", source, digest)
    target = execution.child_output_dir / "malformed-target"

    result = run_audited_child(
        execution,
        action=action,
        target=target,
        expect_denied=True,
    )

    assert result.returncode == 0, execution.stderr_path.read_text(
        encoding="utf-8", errors="strict"
    )
    assert not target.exists()


@pytest.mark.parametrize(
    "action",
    ["winapi-create-file", "winapi-create-junction"],
)
def test_audit_denies_direct_windows_filesystem_primitives(
    tmp_path: Path,
    action: str,
) -> None:
    if os.name != "nt":
        pytest.skip("direct Windows filesystem primitives are unavailable")
    import _winapi

    required = (
        "CreateFile"
        if action == "winapi-create-file"
        else "CreateJunction"
    )
    if not hasattr(_winapi, required):
        pytest.skip(f"_winapi.{required} is unavailable")
    outside = tmp_path / "outside-winapi"
    outside.mkdir()
    if action == "winapi-create-file":
        source_target = outside / "created.txt"
        destination = None
    else:
        source_target = outside / "junction-source"
        source_target.mkdir()
        destination = outside / "created-junction"
    source, digest = measurement_source(tmp_path)
    execution = materialize(tmp_path / "stage", source, digest)

    result = run_audited_child(
        execution,
        action=action,
        target=source_target,
        target2=destination,
        expect_denied=True,
    )

    assert result.returncode == 0, execution.stderr_path.read_text(
        encoding="utf-8", errors="strict"
    )
    if action == "winapi-create-file":
        assert not source_target.exists()
    else:
        assert destination is not None and not destination.exists()


@pytest.mark.parametrize(
    "action",
    ["chown", "chflags", "setxattr", "removexattr", "mknod"],
)
def test_audit_denies_supported_target_platform_extended_mutations(
    tmp_path: Path,
    action: str,
) -> None:
    if os.name != "nt" or not hasattr(os, action):
        pytest.skip(f"{action} is unsupported by the Windows target")
    outside = tmp_path / f"outside-{action}"
    if action != "mknod":
        outside.write_bytes(b"must-remain")
    source, digest = measurement_source(tmp_path)
    execution = materialize(tmp_path / "stage", source, digest)

    result = run_audited_child(
        execution,
        action=action,
        target=outside,
        expect_denied=True,
    )

    assert result.returncode == 0, execution.stderr_path.read_text(
        encoding="utf-8", errors="strict"
    )
    assert outside.exists() == (action != "mknod")


def test_audit_denies_supported_file_descriptor_mutation(
    tmp_path: Path,
) -> None:
    source, digest = measurement_source(tmp_path)
    execution = materialize(tmp_path / "stage", source, digest)
    target = execution.child_output_dir / "fd-mutation.txt"
    target.write_bytes(b"must-remain")

    result = run_audited_child(
        execution,
        action="truncate-fd",
        target=target,
        expect_denied=True,
    )

    assert result.returncode == 0, execution.stderr_path.read_text(
        encoding="utf-8", errors="strict"
    )
    assert target.read_bytes() == b"must-remain"


@pytest.mark.parametrize("action", ["rename", "symlink", "hardlink"])
def test_audit_denies_two_path_mutations_outside_write_roots(
    tmp_path: Path,
    action: str,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    source_target = outside / "source"
    source_target.write_text("keep", encoding="utf-8")
    destination = outside / "destination"
    measurement, digest = measurement_source(tmp_path)
    execution = materialize(tmp_path / "stage", measurement, digest)
    result = run_audited_child(
        execution,
        action=action,
        target=source_target,
        target2=destination,
        expect_denied=True,
    )
    assert result.returncode == 0
    assert source_target.read_text(encoding="utf-8") == "keep"
    assert not destination.exists()


def test_audit_denies_symlink_escape_when_supported(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "secret.txt"
    secret.write_text("secret", encoding="utf-8")
    source, digest = measurement_source(tmp_path)
    execution = materialize(tmp_path / "stage", source, digest)
    link = execution.cwd.parent / "work" / "escape"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"directory symlinks unavailable: {error}")
    result = run_audited_child(
        execution,
        action="read",
        target=link / "secret.txt",
        expect_denied=True,
    )
    assert result.returncode == 0
    assert b"secret" not in execution.stdout_path.read_bytes()


def test_audit_hook_is_installed_before_strict_child_code(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "truth-before-import"
    outside.write_text("secret", encoding="utf-8")
    source, digest = measurement_source(tmp_path)
    execution = materialize(tmp_path / "stage", source, digest)
    result = run_audited_child(
        execution,
        action="read",
        target=outside,
        expect_denied=True,
    )
    assert result.returncode == 0
    events = load_audit_events(execution.audit_log_path)
    assert events[0]["operation"] == "hook-installed"
    assert any(event.get("decision") == "deny" for event in events)


@pytest.mark.parametrize(
    ("action", "reentered_operation"),
    [
        ("reentry-open", "open"),
        ("reentry-process", "subprocess.Popen"),
    ],
)
def test_audit_governed_reentry_poisons_caught_child_execution(
    tmp_path: Path,
    action: str,
    reentered_operation: str,
) -> None:
    source, digest = measurement_source(tmp_path)
    execution = materialize(tmp_path / "stage", source, digest)
    forbidden = tmp_path / (
        "outside-secret.txt"
        if action == "reentry-open"
        else "nested-process-marker.txt"
    )
    if action == "reentry-open":
        forbidden.write_text(
            "forbidden-reentry-secret",
            encoding="utf-8",
        )

    result = run_audited_child(
        execution,
        action=action,
        target=execution.measurements_path,
        target2=forbidden,
    )

    assert result.returncode == 0, execution.stderr_path.read_text(
        encoding="utf-8", errors="strict"
    )
    stdout = execution.stdout_path.read_text(
        encoding="utf-8", errors="strict"
    )
    assert "REENTRY-LEAK=" not in stdout
    assert "REENTRY-DENIED-CAUGHT" in stdout
    assert "REENTRY-OUTER-DENIED-CAUGHT" in stdout
    if action == "reentry-process":
        assert not forbidden.exists()
    events = load_audit_events(execution.audit_log_path)
    assert any(
        event.get("operation") == "audit-reentry"
        and event.get("decision") == "deny"
        and event.get("reentered_operation") == reentered_operation
        for event in events
    )
    assert events[-1].get("operation") == "bootstrap-finished"
    assert events[-1].get("status") == "error"
    with pytest.raises(ValueError, match="denied|unsuccessful|terminal"):
        validate_audit_log(
            execution.audit_log_path,
            expected_policy_sha256=execution.audit_policy_sha256,
        )


def test_audit_fresh_import_creates_no_bytecode_or_denied_write(
    tmp_path: Path,
) -> None:
    source, digest = measurement_source(tmp_path)
    execution = materialize(tmp_path / "stage", source, digest)
    module = execution.cwd / "fresh_blind_module.py"
    module.write_text("VALUE = '盲态验证'\n", encoding="utf-8")
    result = run_audited_child(
        execution,
        action="import-fresh",
        target="fresh_blind_module",
    )
    assert result.returncode == 0, execution.stderr_path.read_text(
        encoding="utf-8", errors="strict"
    )
    assert not list(execution.cwd.rglob("*.pyc"))
    assert not list(execution.cwd.rglob("__pycache__"))
    assert not any(
        event.get("decision") == "deny"
        and event.get("operation") == "open"
        and "pyc" in str(event.get("resolved_path", "")).lower()
        for event in load_audit_events(execution.audit_log_path)
    )


def test_truth_loader_is_absent_from_strict_snapshot_and_import(
    tmp_path: Path,
) -> None:
    source, digest = measurement_source(tmp_path)
    execution = materialize(tmp_path / "stage", source, digest)
    result = run_audited_child(execution, action="import-truth")
    assert result.returncode == 0, execution.stderr_path.read_text(
        encoding="utf-8", errors="strict"
    )
    assert b"TRUTH-LOADER-UNAVAILABLE" in execution.stdout_path.read_bytes()
    inventory = {
        path.relative_to(execution.cwd).as_posix()
        for path in execution.cwd.rglob("*")
        if path.is_file()
    }
    assert "gsdiff/data/_artifact_truth.py" not in inventory
    assert not any(path.startswith("gsdiff/evaluation/") for path in inventory)
    assert "gsdiff/baselines/_evaluation.py" not in inventory


def test_strict_snapshot_keeps_safe_data_api_but_excludes_truth(
    tmp_path: Path,
) -> None:
    source, digest = measurement_source(tmp_path)
    execution = materialize(tmp_path / "stage", source, digest)
    data_root = execution.cwd / "gsdiff" / "data"
    assert (data_root / "__init__.py").is_file()
    assert (data_root / "_artifact_dataset.py").is_file()
    assert (data_root / "_artifact_outputs.py").is_file()
    assert not (data_root / "_artifact_truth.py").exists()


def test_staged_real_baseline_cli_imports_under_audit(
    tmp_path: Path,
) -> None:
    source, digest = measurement_source(tmp_path)
    execution = materialize(tmp_path / "stage", source, digest)
    delimiter = execution.argv.index("--")
    argv = (*execution.argv[: delimiter + 1], "--help")
    with (
        execution.stdout_path.open("wb") as stdout,
        execution.stderr_path.open("wb") as stderr,
    ):
        result = subprocess.run(
            argv,
            cwd=execution.cwd,
            env=dict(execution.env),
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            shell=False,
            check=False,
            timeout=30,
        )
    assert result.returncode == 0, execution.stderr_path.read_text(
        encoding="utf-8", errors="strict"
    )
    assert b"usage:" in execution.stdout_path.read_bytes().lower()
    summary = validate_audit_log(
        execution.audit_log_path,
        expected_policy_sha256=execution.audit_policy_sha256,
    )
    assert summary["terminal_status"] == "success"


def test_audit_utf8_stdio_and_terminal_log_round_trip_exactly(
    tmp_path: Path,
) -> None:
    source, digest = measurement_source(tmp_path)
    execution = materialize(tmp_path / "stage", source, digest)
    result = run_audited_child(execution, action="noop")
    assert result.returncode == 0
    assert execution.stdout_path.read_bytes() == "盲态验证\n".encode("utf-8")
    assert execution.stderr_path.read_bytes() == "盲态验证\n".encode("utf-8")
    events = load_audit_events(execution.audit_log_path)
    assert events[0]["operation"] == "hook-installed"
    assert [event["sequence"] for event in events] == list(range(len(events)))
    assert sum(
        event["operation"] == "bootstrap-finished" for event in events
    ) == 1
    assert events[-1]["operation"] == "bootstrap-finished"
    assert events[-1]["status"] == "success"


def test_audit_policy_hash_and_parent_validation_are_exact(
    tmp_path: Path,
) -> None:
    source, digest = measurement_source(tmp_path)
    execution = materialize(tmp_path / "stage", source, digest)
    assert execution.audit_policy_path == (
        tmp_path / "stage" / "parent" / "audit" / "policy.json"
    ).resolve()
    assert execution.audit_policy_sha256 == hashlib.sha256(
        execution.audit_policy_path.read_bytes()
    ).hexdigest()
    assert run_audited_child(execution, action="noop").returncode == 0
    summary = validate_audit_log(
        execution.audit_log_path,
        expected_policy_sha256=execution.audit_policy_sha256,
    )
    assert summary["event_count"] == len(
        load_audit_events(execution.audit_log_path)
    )
    assert summary["audit_log_sha256"] == hashlib.sha256(
        execution.audit_log_path.read_bytes()
    ).hexdigest()
    with pytest.raises(TypeError):
        summary["event_count"] = 0  # type: ignore[index]
    with pytest.raises(ValueError, match="policy"):
        validate_audit_log(
            execution.audit_log_path,
            expected_policy_sha256="0" * 64,
        )


def _valid_log_lines(policy_sha256: str) -> list[dict[str, object]]:
    return [
        {
            "sequence": 0,
            "timestamp_utc": "2026-07-28T00:00:00.000000Z",
            "operation": "hook-installed",
            "decision": "allow",
            "policy_sha256": policy_sha256,
        },
        {
            "sequence": 1,
            "timestamp_utc": "2026-07-28T00:00:01.000000Z",
            "operation": "bootstrap-finished",
            "decision": "allow",
            "status": "success",
        },
    ]


@pytest.mark.parametrize(
    "mutation",
    [
        "truncated",
        "duplicate-key",
        "duplicate-sequence",
        "out-of-order",
        "policy-mismatch",
        "missing-terminal",
    ],
)
def test_audit_validator_rejects_malformed_or_incomplete_logs(
    tmp_path: Path,
    mutation: str,
) -> None:
    policy_sha256 = "a" * 64
    lines = _valid_log_lines(policy_sha256)
    path = tmp_path / "audit.jsonl"
    if mutation == "truncated":
        path.write_text('{"sequence":', encoding="utf-8")
    elif mutation == "duplicate-key":
        path.write_text(
            '{"sequence":0,"sequence":1,"timestamp_utc":'
            '"2026-07-28T00:00:00.000000Z","operation":'
            '"hook-installed","decision":"allow","policy_sha256":"'
            + policy_sha256
            + '"}\n',
            encoding="utf-8",
        )
    else:
        if mutation == "duplicate-sequence":
            lines[1]["sequence"] = 0
        elif mutation == "out-of-order":
            lines[0]["sequence"] = 1
            lines[1]["sequence"] = 0
        elif mutation == "policy-mismatch":
            lines[0]["policy_sha256"] = "b" * 64
        elif mutation == "missing-terminal":
            lines.pop()
        path.write_text(
            "".join(
                json.dumps(line, separators=(",", ":")) + "\n"
                for line in lines
            ),
            encoding="utf-8",
        )
    with pytest.raises(ValueError):
        validate_audit_log(
            path, expected_policy_sha256=policy_sha256
        )


def test_audit_controlled_site_initialization_never_processes_pth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = tmp_path / "runtime"
    site_packages = runtime / "Lib" / "site-packages"
    site_packages.mkdir(parents=True)
    external = tmp_path / "external"
    external.mkdir()
    marker = tmp_path / "pth-executed"
    (site_packages / "malicious.pth").write_text(
        f"import pathlib; pathlib.Path({str(marker)!r}).write_text('bad')\n"
        f"{external}\n",
        encoding="utf-8",
    )
    code = tmp_path / "code"
    code.mkdir()
    spec = importlib.util.spec_from_file_location(
        "tested_method_child_bootstrap",
        SOURCE_ROOT
        / "scripts"
        / "experiments"
        / "method_child_bootstrap.py",
    )
    assert spec is not None and spec.loader is not None
    bootstrap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bootstrap)
    monkeypatch.setattr(sys, "path", [str(runtime / "python312.zip")])
    policy = {
        "python_runtime_root": str(runtime),
        "runtime_site_package_roots": [str(site_packages)],
    }
    bootstrap._install_controlled_sys_path(policy, code)
    assert not marker.exists()
    assert str(external) not in sys.path
    assert str(site_packages.resolve()) in sys.path
    assert str(code.resolve()) in sys.path


def test_audit_child_sys_path_contains_only_runtime_and_staged_code(
    tmp_path: Path,
) -> None:
    source, digest = measurement_source(tmp_path)
    execution = materialize(tmp_path / "stage", source, digest)
    result = run_audited_child(execution, action="sys-path")
    assert result.returncode == 0, execution.stderr_path.read_text(
        encoding="utf-8", errors="strict"
    )
    line = next(
        line
        for line in execution.stdout_path.read_text(
            encoding="utf-8", errors="strict"
        ).splitlines()
        if line.startswith("SYSPATH=")
    )
    child_paths = json.loads(line.removeprefix("SYSPATH="))
    runtime_root = PYTHON.resolve().parent
    for child_path in child_paths:
        candidate = Path(child_path).resolve(strict=False)
        assert (
            candidate.is_relative_to(runtime_root)
            or candidate.is_relative_to(execution.cwd)
        )
