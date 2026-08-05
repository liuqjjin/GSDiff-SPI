from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
from types import MappingProxyType
from dataclasses import FrozenInstanceError
from typing import get_args, get_type_hints

import pytest

import gsdiff.experiments.identity as identity


BASELINE_COMMIT = "c03420784bc92b4e9b9eef8330cbd9571ebebc68"


def _identity_kwargs(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "execution_class": "blind_method_child",
        "scientific_contract_id": "gsdiff-sim-v1",
        "scientific_contract_sha256": "a" * 64,
        "method_id": "gsdiff_tv",
        "target_id": "tank",
        "motion_id": "translation",
        "seed": 7,
        "config_sha256": "b" * 64,
        "dataset_identity_sha256": "c" * 64,
        "assets_sha256": {"target": "d" * 64},
        "checkpoints_sha256": {"diffusion": "e" * 64},
        "code_commit": "f" * 40,
        "dirty_worktree": False,
        "source_tree_hash": None,
        "dependencies_sha256": "1" * 64,
        "environment_lock_sha256": "2" * 64,
        "metric_version": "metrics-v1",
    }
    values.update(overrides)
    return values


def _make_identity(**overrides: object):
    return identity.build_run_identity(**_identity_kwargs(**overrides))


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "Identity Test")
    _git(repo, "config", "user.email", "identity@example.invalid")
    _git(repo, "config", "commit.gpgsign", "false")
    (repo / "src").mkdir()
    (repo / "src" / "main.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "docs").mkdir()
    (repo / "docs" / "readme.md").write_text("tracked docs\n", encoding="utf-8")
    (repo / ".gitignore").write_text("*.ignored.py\n", encoding="utf-8")
    _git(repo, "add", "--all")
    _git(repo, "commit", "-m", "initial")
    return repo


def _make_directory_link(link: Path, target: Path) -> None:
    try:
        os.symlink(target, link, target_is_directory=True)
        return
    except OSError:
        if os.name != "nt":
            raise
    subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        check=True,
        capture_output=True,
        text=True,
    )


def test_source_input_head_entry_annotation_matches_parser_output():
    return_hint = get_type_hints(identity._collect_source_inputs)["return"]

    assert get_args(return_hint)[2] == dict[str, tuple[str, str]]


def test_sha256_file_matches_standard_vector_and_detects_mutation(tmp_path: Path):
    path = tmp_path / "payload.bin"
    path.write_bytes(b"abc")

    assert identity.sha256_file(path) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )

    path.write_bytes(b"abd")
    assert identity.sha256_file(path) != (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_config_key_order_does_not_change_identity():
    left = {"solver": {"rho": 0.1, "type": "admm"}, "seed": 7}
    right = {"seed": 7, "solver": {"type": "admm", "rho": 0.1}}

    assert identity.resolved_config_sha256(left) == identity.resolved_config_sha256(
        right
    )


def test_config_accepts_and_normalizes_nested_read_only_mappings():
    config = MappingProxyType(
        {
            "solver": MappingProxyType(
                {"type": "admm", "schedule": (1.0, 0.1)}
            ),
            "seed": 7,
        }
    )
    expected = {
        "seed": 7,
        "solver": {"schedule": [1.0, 0.1], "type": "admm"},
    }

    assert identity.resolved_config_sha256(config) == hashlib.sha256(
        json.dumps(
            expected,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


@pytest.mark.parametrize("non_finite", [float("nan"), float("inf"), float("-inf")])
def test_resolved_config_rejects_nested_non_finite_numbers(non_finite: float):
    with pytest.raises(ValueError):
        identity.resolved_config_sha256(
            {"solver": {"schedule": [0.1, {"terminal": non_finite}]}}
        )


def test_run_identity_uses_only_canonical_payload_bytes():
    run_identity = _make_identity()
    expected_payload = {
        "assets_sha256": {"target": "d" * 64},
        "checkpoints_sha256": {"diffusion": "e" * 64},
        "code_commit": "f" * 40,
        "config_sha256": "b" * 64,
        "dataset_identity_sha256": "c" * 64,
        "dependencies_sha256": "1" * 64,
        "dirty_worktree": False,
        "environment_lock_sha256": "2" * 64,
        "execution_class": "blind_method_child",
        "method_id": "gsdiff_tv",
        "metric_version": "metrics-v1",
        "motion_id": "translation",
        "scientific_contract_id": "gsdiff-sim-v1",
        "scientific_contract_sha256": "a" * 64,
        "seed": 7,
        "source_tree_hash": None,
        "target_id": "tank",
    }
    expected_bytes = json.dumps(
        expected_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    expected_sha = hashlib.sha256(expected_bytes).hexdigest()

    assert run_identity.canonical_payload_json == expected_bytes
    assert run_identity.identity_sha256 == expected_sha
    assert run_identity.run_id == (
        f"gsdiff-sim-v1--gsdiff_tv--tank--translation--s7--{expected_sha[:8]}"
    )


def test_payload_returns_a_new_recursively_read_only_decoding():
    run_identity = _make_identity()

    first = run_identity.payload()
    second = run_identity.payload()

    assert isinstance(first, MappingProxyType)
    assert first == second
    assert first is not second
    assert first["assets_sha256"] is not second["assets_sha256"]
    with pytest.raises(TypeError):
        first["seed"] = 8  # type: ignore[index]
    with pytest.raises(TypeError):
        first["assets_sha256"]["target"] = "0" * 64  # type: ignore[index]


def test_public_canonical_json_serializes_read_only_payload_to_stored_bytes():
    run_identity = _make_identity()

    assert (
        identity.canonical_json_bytes(run_identity.payload())
        == run_identity.canonical_payload_json
    )


def test_run_identity_rejects_mutable_noncanonical_or_hash_mismatched_bytes():
    valid = _make_identity()

    with pytest.raises(TypeError):
        identity.RunIdentity(
            bytearray(valid.canonical_payload_json),  # type: ignore[arg-type]
            valid.identity_sha256,
            valid.run_id,
        )
    with pytest.raises(ValueError):
        identity.RunIdentity(
            valid.canonical_payload_json + b" ",
            hashlib.sha256(valid.canonical_payload_json + b" ").hexdigest(),
            valid.run_id,
        )
    with pytest.raises(ValueError):
        identity.RunIdentity(
            valid.canonical_payload_json,
            "0" * 64,
            valid.run_id,
        )
    with pytest.raises(FrozenInstanceError):
        valid.run_id = "changed"  # type: ignore[misc]


def test_run_identity_rejects_non_string_equality_spoofs():
    valid = _make_identity()

    class _EqualitySpoof:
        def __eq__(self, other: object) -> bool:
            return True

    spoof = _EqualitySpoof()
    with pytest.raises(TypeError):
        identity.RunIdentity(
            valid.canonical_payload_json,
            spoof,  # type: ignore[arg-type]
            spoof,  # type: ignore[arg-type]
        )


def _construct_direct_identity(payload: dict[str, object]):
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    payload_sha256 = hashlib.sha256(encoded).hexdigest()
    run_id = (
        f"{payload['scientific_contract_id']}--{payload['method_id']}--"
        f"{payload['target_id']}--{payload['motion_id']}--s{payload['seed']}--"
        f"{payload_sha256[:8]}"
    )
    return identity.RunIdentity(encoded, payload_sha256, run_id)


def _decoded_valid_payload() -> dict[str, object]:
    return json.loads(_make_identity().canonical_payload_json.decode("utf-8"))


def test_direct_run_identity_construction_cannot_bypass_blind_execution_gate():
    payload = _decoded_valid_payload()
    payload["execution_class"] = "compatibility_unblinded"

    with pytest.raises(ValueError, match="blind_method_child"):
        _construct_direct_identity(payload)


def test_direct_run_identity_construction_rejects_unknown_payload_field():
    payload = _decoded_valid_payload()
    payload["campaign_id"] = "primary-v1"

    with pytest.raises(ValueError, match="field|schema|unknown"):
        _construct_direct_identity(payload)


def test_direct_run_identity_construction_rejects_missing_payload_field():
    payload = _decoded_valid_payload()
    del payload["dependencies_sha256"]

    with pytest.raises(ValueError, match="field|schema|missing"):
        _construct_direct_identity(payload)


@pytest.mark.parametrize(
    ("dirty_worktree", "source_tree_hash"),
    [(True, None), (False, "a" * 64)],
)
def test_direct_run_identity_construction_enforces_dirty_source_pairing(
    dirty_worktree: bool, source_tree_hash: str | None
):
    payload = _decoded_valid_payload()
    payload["dirty_worktree"] = dirty_worktree
    payload["source_tree_hash"] = source_tree_hash

    with pytest.raises(ValueError, match="dirty|source_tree"):
        _construct_direct_identity(payload)


def test_caller_mutations_cannot_change_canonical_payload_or_hash():
    config = {"solver": {"rho": 0.1, "schedule": [1.0, 0.1]}}
    assets = {"target": "d" * 64}
    checkpoints = {"diffusion": "e" * 64}
    run_identity = _make_identity(
        config_sha256=identity.resolved_config_sha256(config),
        assets_sha256=assets,
        checkpoints_sha256=checkpoints,
    )
    original_bytes = run_identity.canonical_payload_json
    original_sha = run_identity.identity_sha256
    original_payload = run_identity.payload()

    config["solver"]["schedule"][0] = 99.0
    assets["target"] = "3" * 64
    assets["new"] = "4" * 64
    checkpoints.clear()

    assert run_identity.canonical_payload_json == original_bytes
    assert run_identity.identity_sha256 == original_sha
    assert run_identity.payload() == original_payload


def test_asset_and_checkpoint_mapping_order_does_not_change_identity():
    assets_left = {"target": "d" * 64, "mask": "3" * 64}
    assets_right = {"mask": "3" * 64, "target": "d" * 64}
    checkpoints_left = {"scene": "4" * 64, "diffusion": "e" * 64}
    checkpoints_right = {"diffusion": "e" * 64, "scene": "4" * 64}

    first = _make_identity(
        assets_sha256=assets_left,
        checkpoints_sha256=checkpoints_left,
    )
    second = _make_identity(
        assets_sha256=assets_right,
        checkpoints_sha256=checkpoints_right,
    )

    assert first.canonical_payload_json == second.canonical_payload_json
    assert first.identity_sha256 == second.identity_sha256


def test_checkpoint_hash_change_invalidates_identity():
    first = _make_identity(checkpoints_sha256={"diffusion": "a" * 64})
    second = _make_identity(checkpoints_sha256={"diffusion": "b" * 64})

    assert first.identity_sha256 != second.identity_sha256


def test_logical_config_and_campaign_names_do_not_enter_run_identity():
    resolved = {"solver": {"steps": 10}}
    dataset = {"target": "tank", "measurements": 2560}
    primary_logical_names = {
        "campaign_id": "primary-v1",
        "acquisition_config_id": "base",
        "method_config_id": "default",
    }
    supplement_logical_names = {
        "campaign_id": "supplement-grid-v1",
        "acquisition_config_id": "renamed-base",
        "method_config_id": "renamed-default",
    }
    assert primary_logical_names != supplement_logical_names
    shared = {
        "config_sha256": identity.resolved_config_sha256(resolved),
        "dataset_identity_sha256": identity.resolved_config_sha256(dataset),
    }

    primary_named_config = _make_identity(**shared)
    supplement_renamed_config = _make_identity(**shared)

    assert primary_named_config.identity_sha256 == supplement_renamed_config.identity_sha256
    assert "campaign_id" not in primary_named_config.payload()
    assert "acquisition_config_id" not in primary_named_config.payload()
    assert "method_config_id" not in primary_named_config.payload()
    for unsupported, value in primary_logical_names.items():
        with pytest.raises(TypeError, match="unexpected"):
            identity.build_run_identity(
                **_identity_kwargs(),
                **{unsupported: value},
            )


def test_resolved_config_and_dataset_content_changes_invalidate_identity():
    first = _make_identity(
        config_sha256=identity.resolved_config_sha256({"steps": 10}),
        dataset_identity_sha256=identity.resolved_config_sha256({"target": "tank"}),
    )
    changed_config = _make_identity(
        config_sha256=identity.resolved_config_sha256({"steps": 11}),
        dataset_identity_sha256=identity.resolved_config_sha256({"target": "tank"}),
    )
    changed_dataset = _make_identity(
        config_sha256=identity.resolved_config_sha256({"steps": 10}),
        dataset_identity_sha256=identity.resolved_config_sha256({"target": "digit5"}),
    )

    assert len({first.identity_sha256, changed_config.identity_sha256, changed_dataset.identity_sha256}) == 3


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("scientific_contract_id", "gsdiff-sim-v2"),
        ("scientific_contract_sha256", "3" * 64),
        ("config_sha256", "4" * 64),
        ("dataset_identity_sha256", "5" * 64),
        ("assets_sha256", {"target": "6" * 64}),
        ("checkpoints_sha256", {"diffusion": "6" * 64}),
        ("method_id", "gsdiff_diffusion"),
        ("target_id", "digit5"),
        ("motion_id", "rotation"),
        ("seed", 11),
        ("code_commit", "7" * 40),
        ("dependencies_sha256", "8" * 64),
        ("environment_lock_sha256", "9" * 64),
        ("metric_version", "metrics-v2"),
    ],
)
def test_each_identity_bearing_field_invalidates_identity(
    field: str, replacement: object
):
    original = _make_identity()
    changed = _make_identity(**{field: replacement})

    assert changed.identity_sha256 != original.identity_sha256


def test_dirty_source_cannot_share_clean_identity():
    clean = _make_identity(dirty_worktree=False, source_tree_hash=None)
    dirty = _make_identity(dirty_worktree=True, source_tree_hash="c" * 64)

    assert clean.identity_sha256 != dirty.identity_sha256


def test_two_distinct_dirty_source_hashes_produce_distinct_identities():
    first = _make_identity(dirty_worktree=True, source_tree_hash="c" * 64)
    second = _make_identity(dirty_worktree=True, source_tree_hash="d" * 64)

    assert first.identity_sha256 != second.identity_sha256


@pytest.mark.parametrize(
    ("dirty_worktree", "source_tree_hash"),
    [
        (True, None),
        (False, "c" * 64),
    ],
)
def test_dirty_flag_and_source_hash_must_agree(
    dirty_worktree: bool, source_tree_hash: str | None
):
    with pytest.raises(ValueError):
        _make_identity(
            dirty_worktree=dirty_worktree,
            source_tree_hash=source_tree_hash,
        )


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("scientific_contract_id", ""),
        ("scientific_contract_id", "Gsdiff"),
        ("method_id", "-method"),
        ("target_id", "target.name"),
        ("motion_id", "旋转"),
        ("metric_version", "Metrics-v1"),
    ],
)
def test_identity_ids_use_the_exact_ascii_lowercase_policy(field: str, invalid: str):
    with pytest.raises(ValueError):
        _make_identity(**{field: invalid})


@pytest.mark.parametrize(
    "field",
    [
        "scientific_contract_sha256",
        "config_sha256",
        "dataset_identity_sha256",
        "dependencies_sha256",
        "environment_lock_sha256",
    ],
)
@pytest.mark.parametrize("invalid", ["a" * 63, "A" * 64, "g" * 64, 3])
def test_identity_sha_fields_require_exact_lowercase_hex(
    field: str, invalid: object
):
    with pytest.raises((TypeError, ValueError)):
        _make_identity(**{field: invalid})


@pytest.mark.parametrize("invalid", ["f" * 39, "F" * 40, "g" * 40, 3])
def test_code_commit_requires_exact_full_lowercase_git_hex(invalid: object):
    with pytest.raises((TypeError, ValueError)):
        _make_identity(code_commit=invalid)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("assets_sha256", {"bad.name": "a" * 64}),
        ("assets_sha256", {"target": "A" * 64}),
        ("checkpoints_sha256", {"bad.name": "a" * 64}),
        ("checkpoints_sha256", {"scene": "a" * 63}),
    ],
)
def test_named_hash_mappings_validate_names_and_sha_values(
    field: str, value: object
):
    with pytest.raises((TypeError, ValueError)):
        _make_identity(**{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seed", True),
        ("seed", 7.0),
        ("dirty_worktree", 1),
        ("dirty_worktree", "false"),
    ],
)
def test_identity_scalar_types_are_exact(field: str, value: object):
    with pytest.raises((TypeError, ValueError)):
        _make_identity(**{field: value})


def test_compatibility_execution_is_rejected_before_identity_construction(
    monkeypatch: pytest.MonkeyPatch,
):
    constructed: list[object] = []

    class _ConstructionSentinel:
        def __init__(self, *args: object, **kwargs: object):
            constructed.append((args, kwargs))

    monkeypatch.setattr(identity, "RunIdentity", _ConstructionSentinel)

    with pytest.raises(ValueError, match="blind_method_child"):
        identity.build_run_identity(
            **_identity_kwargs(execution_class="compatibility_unblinded")
        )

    assert constructed == []


def test_execution_class_string_subclass_cannot_bypass_preconstruction_gate(
    monkeypatch: pytest.MonkeyPatch,
):
    constructed: list[object] = []

    class _CompatibilitySpoof(str):
        def __ne__(self, other: object) -> bool:
            return False

    class _ConstructionSentinel:
        def __init__(self, *args: object, **kwargs: object):
            constructed.append((args, kwargs))

    monkeypatch.setattr(identity, "RunIdentity", _ConstructionSentinel)

    with pytest.raises((TypeError, ValueError), match="blind_method_child"):
        identity.build_run_identity(
            **_identity_kwargs(
                execution_class=_CompatibilitySpoof("compatibility_unblinded")
            )
        )

    assert constructed == []


@pytest.mark.parametrize("execution_class", [None, "", "method_child_blind", "unknown"])
def test_only_exact_blind_method_child_execution_is_accepted(
    execution_class: object,
):
    with pytest.raises((TypeError, ValueError), match="blind_method_child"):
        _make_identity(execution_class=execution_class)


def test_git_state_reports_full_commit_branch_cleanliness_and_baseline(
    git_repo: Path,
):
    state = identity.git_state(git_repo, [Path("src")])

    assert state == {
        "commit": _git(git_repo, "rev-parse", "HEAD"),
        "branch": "main",
        "dirty": False,
        "baseline": BASELINE_COMMIT,
    }
    assert len(state["commit"]) == 40

    (git_repo / "src" / "main.py").write_text("VALUE = 2\n", encoding="utf-8")
    assert identity.git_state(git_repo, [Path("src")])["dirty"] is True


def test_git_state_disables_optional_locks_without_mutating_process_env(
    git_repo: Path, monkeypatch
):
    original = os.environ.get("GIT_OPTIONAL_LOCKS")
    observed: list[str | None] = []
    real_run = identity.subprocess.run

    def observing_run(*args, **kwargs):
        command = args[0]
        if command and command[0] == "git":
            environment = kwargs.get("env")
            observed.append(
                None
                if environment is None
                else environment.get("GIT_OPTIONAL_LOCKS")
            )
        return real_run(*args, **kwargs)

    monkeypatch.setattr(identity.subprocess, "run", observing_run)

    identity.git_state(git_repo, [Path("src")])

    assert observed
    assert set(observed) == {"0"}
    assert os.environ.get("GIT_OPTIONAL_LOCKS") == original


def test_git_state_does_not_let_replace_refs_hide_dirty_sources(git_repo: Path):
    claimed_commit = _git(git_repo, "rev-parse", "HEAD")
    source = git_repo / "src" / "main.py"
    source.write_text("VALUE = 2\n", encoding="utf-8")
    _git(git_repo, "add", "src/main.py")
    _git(git_repo, "commit", "-m", "replacement")
    replacement_commit = _git(git_repo, "rev-parse", "HEAD")
    _git(git_repo, "reset", "--soft", claimed_commit)
    _git(git_repo, "replace", claimed_commit, replacement_commit)

    assert _git(
        git_repo, "status", "--porcelain=v1", "--untracked-files=all"
    ) == ""

    state = identity.git_state(git_repo, [Path("src")])

    assert state["commit"] == claimed_commit
    assert state["dirty"] is True


def test_git_state_rejects_head_change_during_provenance_read(
    git_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    claimed_commit = _git(git_repo, "rev-parse", "HEAD")
    source = git_repo / "src" / "main.py"
    source.write_text("VALUE = 2\n", encoding="utf-8")
    _git(git_repo, "add", "src/main.py")
    _git(git_repo, "commit", "-m", "racing head")
    racing_commit = _git(git_repo, "rev-parse", "HEAD")
    _git(git_repo, "reset", "--hard", claimed_commit)
    original_git_bytes = identity._git_bytes
    changed = False

    def racing_git_bytes(repo: Path, *args: str) -> bytes:
        nonlocal changed
        payload = original_git_bytes(repo, *args)
        if args == ("rev-parse", "HEAD") and not changed:
            changed = True
            _git(git_repo, "reset", "--hard", racing_commit)
        return payload

    monkeypatch.setattr(identity, "_git_bytes", racing_git_bytes)

    with pytest.raises(ValueError, match="provenance|HEAD|changed"):
        identity.git_state(git_repo, [Path("src")])


def test_git_state_rejects_index_change_during_provenance_read(
    git_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source = git_repo / "src" / "main.py"
    original_git_bytes = identity._git_bytes
    changed = False

    def racing_git_bytes(repo: Path, *args: str) -> bytes:
        nonlocal changed
        payload = original_git_bytes(repo, *args)
        if args == ("ls-files", "--stage", "-z") and not changed:
            changed = True
            source.write_text("VALUE = 9\n", encoding="utf-8")
            _git(git_repo, "add", "src/main.py")
            source.write_text("VALUE = 1\n", encoding="utf-8")
        return payload

    monkeypatch.setattr(identity, "_git_bytes", racing_git_bytes)

    with pytest.raises(ValueError, match="provenance|index|changed"):
        identity.git_state(git_repo, [Path("src")])


def test_git_state_treats_ignored_untracked_input_inside_source_root_as_dirty(
    git_repo: Path,
):
    assert identity.git_state(git_repo, [Path("src")])["dirty"] is False
    (git_repo / "src" / "plugin.ignored.py").write_text(
        "PLUGIN = True\n", encoding="utf-8"
    )

    assert identity.git_state(git_repo, [Path("src")])["dirty"] is True


def test_git_state_ignores_literal_exclusion_inside_source_root(git_repo: Path):
    ignored_cache = git_repo / "src" / "__pycache__" / "plugin.ignored.py"
    ignored_cache.parent.mkdir()
    ignored_cache.write_text("CACHE = True\n", encoding="utf-8")

    assert identity.git_state(git_repo, [Path("src")])["dirty"] is False


def test_git_state_ignores_ignored_untracked_input_outside_source_roots(
    git_repo: Path,
):
    (git_repo / "docs" / "plugin.ignored.py").write_text(
        "PLUGIN = True\n", encoding="utf-8"
    )

    assert identity.git_state(git_repo, [Path("src")])["dirty"] is False


@pytest.mark.parametrize(
    ("set_flag", "clear_flag"),
    [
        ("--assume-unchanged", "--no-assume-unchanged"),
        ("--skip-worktree", "--no-skip-worktree"),
    ],
)
def test_git_state_detects_flag_hidden_source_content_and_presence(
    git_repo: Path,
    set_flag: str,
    clear_flag: str,
):
    roots = [Path("src")]
    source = git_repo / "src" / "main.py"
    original = source.read_bytes()
    clean_hash = identity.source_tree_sha256(git_repo, roots)

    _git(git_repo, "update-index", set_flag, "src/main.py")
    assert _git(
        git_repo, "status", "--porcelain=v1", "--untracked-files=all"
    ) == ""
    assert identity.source_tree_sha256(git_repo, roots) == clean_hash
    assert identity.git_state(git_repo, roots)["dirty"] is False

    source.write_text("VALUE = 999\n", encoding="utf-8")
    assert _git(
        git_repo, "status", "--porcelain=v1", "--untracked-files=all"
    ) == ""
    assert identity.source_tree_sha256(git_repo, roots) != clean_hash
    assert identity.git_state(git_repo, roots)["dirty"] is True

    source.write_bytes(original)
    assert identity.source_tree_sha256(git_repo, roots) == clean_hash
    assert identity.git_state(git_repo, roots)["dirty"] is False

    source.unlink()
    assert _git(
        git_repo, "status", "--porcelain=v1", "--untracked-files=all"
    ) == ""
    assert identity.source_tree_sha256(git_repo, roots) != clean_hash
    assert identity.git_state(git_repo, roots)["dirty"] is True

    source.write_bytes(original)
    _git(git_repo, "update-index", clear_flag, "src/main.py")
    assert identity.source_tree_sha256(git_repo, roots) == clean_hash
    assert identity.git_state(git_repo, roots)["dirty"] is False


@pytest.mark.parametrize(
    ("repo_name", "branch_name"),
    [("测试仓库", "分支"), ("repo😀", "feature😀")],
)
def test_unicode_git_repo_roots_use_explicit_utf8(
    tmp_path: Path, repo_name: str, branch_name: str
):
    repo = tmp_path / repo_name
    repo.mkdir()
    _git(repo, "init", "-b", branch_name)
    _git(repo, "config", "user.name", "Identity Test")
    _git(repo, "config", "user.email", "identity@example.invalid")
    _git(repo, "config", "commit.gpgsign", "false")
    (repo / "src").mkdir()
    (repo / "src" / "main.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo, "add", "--all")
    _git(repo, "commit", "-m", "initial")

    source_hash = identity.source_tree_sha256(repo, [Path("src")])
    state = identity.git_state(repo, [Path("src")])

    assert len(source_hash) == 64
    assert state["commit"] == _git(repo, "rev-parse", "HEAD")
    assert state["branch"] == branch_name
    assert state["dirty"] is False


def test_clean_source_tree_hash_is_deterministic_and_root_order_independent(
    git_repo: Path,
):
    first = identity.source_tree_sha256(
        git_repo, [Path("src"), git_repo / "docs"]
    )
    second = identity.source_tree_sha256(
        git_repo, [git_repo / "docs", git_repo / "src"]
    )

    assert first == second
    assert len(first) == 64
    assert first == first.lower()


def test_tracked_paths_outside_source_roots_do_not_change_hash(git_repo: Path):
    original = identity.source_tree_sha256(git_repo, [git_repo / "src"])

    (git_repo / "docs" / "readme.md").write_text("changed docs\n", encoding="utf-8")
    assert identity.source_tree_sha256(git_repo, [git_repo / "src"]) == original

    (git_repo / "src" / "main.py").write_text("VALUE = 2\n", encoding="utf-8")
    assert identity.source_tree_sha256(git_repo, [git_repo / "src"]) != original


def test_staged_source_content_changes_hash(git_repo: Path):
    original = identity.source_tree_sha256(git_repo, [git_repo / "src"])
    (git_repo / "src" / "main.py").write_text("VALUE = 2\n", encoding="utf-8")
    _git(git_repo, "add", "src/main.py")

    assert identity.source_tree_sha256(git_repo, [git_repo / "src"]) != original


def test_staged_index_content_remains_hashed_after_worktree_returns_to_head(
    git_repo: Path,
):
    source = git_repo / "src" / "main.py"
    head_content = source.read_bytes()
    original = identity.source_tree_sha256(git_repo, [git_repo / "src"])
    source.write_text("VALUE = 99\n", encoding="utf-8")
    _git(git_repo, "add", "src/main.py")
    source.write_bytes(head_content)

    assert source.read_bytes() == head_content
    assert _git(git_repo, "diff", "--cached", "--name-only") == "src/main.py"
    assert identity.source_tree_sha256(git_repo, [git_repo / "src"]) != original


def test_staged_deletion_remains_hashed_after_working_file_is_rebuilt(
    git_repo: Path,
):
    source = git_repo / "src" / "main.py"
    head_content = source.read_bytes()
    original = identity.source_tree_sha256(git_repo, [git_repo / "src"])
    _git(git_repo, "rm", "src/main.py")
    source.parent.mkdir(exist_ok=True)
    source.write_bytes(head_content)

    assert source.read_bytes() == head_content
    assert _git(git_repo, "diff", "--cached", "--name-only") == "src/main.py"
    assert identity.source_tree_sha256(git_repo, [git_repo / "src"]) != original


def test_unstaged_source_content_changes_hash(git_repo: Path):
    original = identity.source_tree_sha256(git_repo, [git_repo / "src"])
    (git_repo / "src" / "main.py").write_text("VALUE = 2\n", encoding="utf-8")

    assert identity.source_tree_sha256(git_repo, [git_repo / "src"]) != original


def test_source_rename_is_hashed_as_delete_and_add(git_repo: Path):
    original = identity.source_tree_sha256(git_repo, [git_repo / "src"])
    (git_repo / "src" / "main.py").rename(git_repo / "src" / "renamed.py")
    _git(git_repo, "add", "--all")

    assert identity.source_tree_sha256(git_repo, [git_repo / "src"]) != original


def test_source_deletion_changes_hash(git_repo: Path):
    original = identity.source_tree_sha256(git_repo, [git_repo / "src"])
    _git(git_repo, "rm", "src/main.py")

    assert identity.source_tree_sha256(git_repo, [git_repo / "src"]) != original


def test_executable_mode_changes_are_identity_bearing(git_repo: Path):
    source = git_repo / "src" / "main.py"
    original = identity.source_tree_sha256(git_repo, [git_repo / "src"])

    _git(git_repo, "update-index", "--chmod=+x", "--", "src/main.py")
    assert identity.source_tree_sha256(git_repo, [git_repo / "src"]) != original

    _git(git_repo, "update-index", "--chmod=-x", "--", "src/main.py")
    if os.name != "nt":
        source.chmod(source.stat().st_mode | stat.S_IXUSR)
        assert identity.source_tree_sha256(git_repo, [git_repo / "src"]) != original


def test_binary_source_content_changes_hash(git_repo: Path):
    binary = git_repo / "src" / "weights.bin"
    binary.write_bytes(b"\x00\xff\x10source")
    _git(git_repo, "add", "src/weights.bin")
    _git(git_repo, "commit", "-m", "add binary")
    original = identity.source_tree_sha256(git_repo, [git_repo / "src"])

    binary.write_bytes(b"\x00\xff\x10changed")

    assert identity.source_tree_sha256(git_repo, [git_repo / "src"]) != original


def test_ignored_untracked_source_inside_root_changes_hash(git_repo: Path):
    original = identity.source_tree_sha256(git_repo, [git_repo / "src"])
    (git_repo / "src" / "plugin.ignored.py").write_text(
        "PLUGIN = True\n", encoding="utf-8"
    )

    assert _git(
        git_repo,
        "check-ignore",
        "src/plugin.ignored.py",
    ) == "src/plugin.ignored.py"
    assert identity.source_tree_sha256(git_repo, [git_repo / "src"]) != original


def test_literal_artifact_cache_environment_trash_and_paper_outputs_are_excluded(
    git_repo: Path,
):
    original = identity.source_tree_sha256(git_repo, [git_repo])
    excluded_paths = [
        "artifacts/run/output.json",
        "results/run/output.json",
        "_trash/2026-07-27/source.py",
        "__pycache__/source.pyc",
        ".pytest_cache/v/cache/nodeids",
        ".mypy_cache/state.json",
        ".ruff_cache/state",
        ".cache/state",
        ".tox/py/python.exe",
        ".nox/tests/python.exe",
        ".hypothesis/examples/data",
        ".ipynb_checkpoints/notebook.py",
        ".venv/site-packages/module.py",
        "venv/site-packages/module.py",
        "env/site-packages/module.py",
        "ENV/site-packages/module.py",
        "__pypackages__/3.12/lib/module.py",
        ".pixi/envs/default/module.py",
        "paper/figure_data/main.csv",
        "paper/figures/main.pdf",
        "paper/tables/main.tex",
        "paper/build/main.aux",
        "paper/generated/main.tex",
    ]
    for relative in excluded_paths:
        path = git_repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"excluded")
        assert identity.source_tree_sha256(git_repo, [git_repo]) == original

    included_literal_neighbor = git_repo / "src" / "artifacts.py"
    included_literal_neighbor.write_text("VALUE = 1\n", encoding="utf-8")
    assert identity.source_tree_sha256(git_repo, [git_repo]) != original


def test_non_ascii_untracked_paths_are_sorted_as_utf8(git_repo: Path):
    first_root = git_repo / "src"
    second_root = git_repo / "extra"
    second_root.mkdir()
    (first_root / "测.py").write_text("VALUE = 1\n", encoding="utf-8")
    (second_root / "a.py").write_text("VALUE = 2\n", encoding="utf-8")

    first = identity.source_tree_sha256(git_repo, [first_root, second_root])
    second = identity.source_tree_sha256(git_repo, [second_root, first_root])

    assert first == second


def test_source_root_case_alias_cannot_create_a_second_windows_identity(
    git_repo: Path,
):
    if os.name != "nt":
        with pytest.raises(ValueError):
            identity.source_tree_sha256(git_repo, [Path("SRC")])
        return

    canonical = identity.source_tree_sha256(git_repo, [Path("src")])
    alias = identity.source_tree_sha256(git_repo, [Path("SRC")])
    assert alias == canonical

    source = git_repo / "src" / "main.py"
    head_content = source.read_bytes()
    source.write_text("VALUE = 99\n", encoding="utf-8")
    _git(git_repo, "add", "src/main.py")
    source.write_bytes(head_content)

    assert identity.source_tree_sha256(git_repo, [Path("SRC")]) != canonical


def test_case_only_direct_file_root_keeps_staged_index_content_on_windows(
    git_repo: Path,
):
    if os.name != "nt":
        pytest.skip("Windows filesystem identity regression")

    original = git_repo / "src" / "main.py"
    canonical = git_repo / "src" / "Main.py"
    _git(git_repo, "mv", "src/main.py", "src/Main.py")
    _git(git_repo, "commit", "-m", "canonical mixed-case source")
    canonical.rename(original)
    head_content = original.read_bytes()
    before = identity.source_tree_sha256(git_repo, [Path("src/main.py")])

    original.write_text("VALUE = 999\n", encoding="utf-8")
    _git(git_repo, "add", "-u")
    original.write_bytes(head_content)

    assert _git(git_repo, "diff", "--cached", "--name-only") == "src/Main.py"
    assert identity.source_tree_sha256(
        git_repo, [Path("src/main.py")]
    ) != before


def test_fully_deleted_ascii_case_alias_uses_unique_git_prefix_on_windows(
    git_repo: Path,
):
    if os.name != "nt":
        pytest.skip("Windows filesystem identity regression")

    _git(git_repo, "rm", "src/main.py")

    canonical = identity.source_tree_sha256(git_repo, [Path("src")])
    alias = identity.source_tree_sha256(git_repo, [Path("SRC")])
    assert alias == canonical


def test_ambiguous_case_colliding_git_paths_fail_closed_on_windows(
    git_repo: Path,
):
    if os.name != "nt":
        pytest.skip("Windows filesystem identity regression")

    object_id = subprocess.run(
        ["git", "hash-object", "-w", "--stdin"],
        cwd=git_repo,
        check=True,
        input=b"VALUE = 2\n",
        capture_output=True,
    ).stdout.decode("ascii").strip()
    _git(
        git_repo,
        "update-index",
        "--add",
        "--cacheinfo",
        f"100644,{object_id},src/Main.py",
    )

    with pytest.raises(ValueError, match="ambiguous|unique|colliding"):
        identity.source_tree_sha256(git_repo, [Path("src/main.py")])


def test_missing_source_root_does_not_use_unicode_casefold_alias(
    git_repo: Path,
):
    source = git_repo / "src" / "strasse" / "module.py"
    source.parent.mkdir()
    source.write_text("VALUE = 1\n", encoding="utf-8")
    _git(git_repo, "add", "src/strasse/module.py")
    _git(git_repo, "commit", "-m", "add strasse source")
    _git(git_repo, "rm", "src/strasse/module.py")

    assert not (git_repo / "src" / "stra\xdfe").exists()
    with pytest.raises(ValueError, match="does not exist"):
        identity.source_tree_sha256(git_repo, [Path("src/stra\xdfe")])


@pytest.mark.parametrize("mode", ["120000", "160000"])
def test_non_regular_git_modes_are_rejected(git_repo: Path, mode: str):
    if mode == "120000":
        object_id = subprocess.run(
            ["git", "hash-object", "-w", "--stdin"],
            cwd=git_repo,
            check=True,
            input=b"main.py",
            capture_output=True,
        ).stdout.decode("ascii").strip()
        path = "src/link"
    else:
        object_id = _git(git_repo, "rev-parse", "HEAD")
        path = "src/submodule"
    _git(
        git_repo,
        "update-index",
        "--add",
        "--cacheinfo",
        f"{mode},{object_id},{path}",
    )

    with pytest.raises(ValueError, match="mode|regular|symlink|submodule"):
        identity.source_tree_sha256(git_repo, [git_repo / "src"])


def test_internal_worktree_reparse_points_are_rejected(git_repo: Path):
    target = git_repo / "src" / "internal-target"
    target.mkdir()
    (target / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    _make_directory_link(git_repo / "src" / "internal-link", target)

    with pytest.raises(ValueError, match="reparse|link|regular"):
        identity.source_tree_sha256(git_repo, [git_repo / "src"])


def test_source_roots_must_be_nonempty_and_lexically_inside_repo(
    git_repo: Path, tmp_path: Path
):
    outside = tmp_path / "outside"
    outside.mkdir()

    with pytest.raises(ValueError):
        identity.source_tree_sha256(git_repo, [])
    with pytest.raises(ValueError):
        identity.source_tree_sha256(git_repo, [outside])
    with pytest.raises(ValueError):
        identity.source_tree_sha256(git_repo, [Path("../outside")])


def test_source_tree_rejects_git_metadata_directory_and_bare_repo(
    git_repo: Path, tmp_path: Path
):
    bare_repo = tmp_path / "bare.git"
    _git(tmp_path, "clone", "--bare", str(git_repo), str(bare_repo))

    for non_worktree in (git_repo / ".git", bare_repo):
        with pytest.raises(ValueError, match="worktree"):
            identity.source_tree_sha256(non_worktree, [Path("HEAD")])


def test_source_tree_accepts_linked_worktree_root(
    git_repo: Path, tmp_path: Path
):
    linked = tmp_path / "linked-worktree"
    _git(git_repo, "worktree", "add", "-b", "linked", str(linked))

    source_hash = identity.source_tree_sha256(linked, [Path("src")])

    assert len(source_hash) == 64


def test_escaping_source_root_link_is_rejected_before_traversal(
    git_repo: Path, tmp_path: Path
):
    outside = tmp_path / "outside-root"
    outside.mkdir()
    (outside / "secret.py").write_text("SECRET = True\n", encoding="utf-8")
    link = git_repo / "linked-source"
    _make_directory_link(link, outside)

    with pytest.raises(ValueError, match="escape|outside|repository"):
        identity.source_tree_sha256(git_repo, [link])


def test_escaping_nested_link_is_rejected_before_traversal(
    git_repo: Path, tmp_path: Path
):
    outside = tmp_path / "outside-child"
    outside.mkdir()
    (outside / "secret.py").write_text("SECRET = True\n", encoding="utf-8")
    _make_directory_link(git_repo / "src" / "linked-child", outside)

    with pytest.raises(ValueError, match="escape|outside|repository"):
        identity.source_tree_sha256(git_repo, [git_repo / "src"])


def test_environment_requirements_verification_hashes_unique_dependency_records(
    tmp_path: Path,
):
    requirements = tmp_path / "requirements-lock.txt"
    requirements.write_text("Beta==2.0\nalpha_pkg==1.0\n", encoding="utf-8")
    fingerprint = {
        "installed_distributions": [
            {"name": "alpha-pkg", "version": "1.0"},
            {"name": "alpha_pkg", "version": "1.0"},
            {"name": "beta", "version": "2.0"},
        ]
    }
    lock = {
        "schema_version": 1,
        "fingerprint": fingerprint,
        "fingerprint_sha256": identity.sha256_bytes(
            identity.canonical_json_bytes(fingerprint)
        ),
    }
    environment_lock = tmp_path / "environment-lock.json"
    environment_lock.write_bytes(identity.canonical_json_bytes(lock))

    hashes = identity.verify_environment_requirements(
        requirements, environment_lock, live_fingerprint=fingerprint
    )

    assert hashes == {
        "dependencies_sha256": hashlib.sha256(
            b'[{"name":"alpha-pkg","version":"1.0"},{"name":"beta","version":"2.0"}]'
        ).hexdigest(),
        "environment_lock_sha256": lock["fingerprint_sha256"],
    }


def test_environment_requirements_rejects_conflicting_duplicate_distribution(
    tmp_path: Path,
):
    requirements = tmp_path / "requirements-lock.txt"
    requirements.write_text("alpha==1\n", encoding="utf-8")
    fingerprint = {
        "installed_distributions": [
            {"name": "alpha", "version": "1"},
            {"name": "alpha", "version": "2"},
        ]
    }
    lock = {
        "schema_version": 1,
        "fingerprint": fingerprint,
        "fingerprint_sha256": identity.sha256_bytes(
            identity.canonical_json_bytes(fingerprint)
        ),
    }
    environment_lock = tmp_path / "environment-lock.json"
    environment_lock.write_bytes(identity.canonical_json_bytes(lock))

    with pytest.raises(ValueError, match="conflicting versions"):
        identity.verify_environment_requirements(
            requirements, environment_lock, live_fingerprint=fingerprint
        )


def test_environment_requirements_rejects_live_full_fingerprint_mismatch(
    tmp_path: Path,
):
    requirements = tmp_path / "requirements-lock.txt"
    requirements.write_text("alpha==1\n", encoding="utf-8")
    fingerprint = {"installed_distributions": [{"name": "alpha", "version": "1"}], "gpu": "locked"}
    lock = {
        "schema_version": 1,
        "fingerprint": fingerprint,
        "fingerprint_sha256": identity.sha256_bytes(identity.canonical_json_bytes(fingerprint)),
    }
    environment_lock = tmp_path / "environment-lock.json"
    environment_lock.write_bytes(identity.canonical_json_bytes(lock))

    with pytest.raises(ValueError, match="exactly"):
        identity.verify_environment_requirements(
            requirements,
            environment_lock,
            live_fingerprint={**fingerprint, "gpu": "different"},
        )


def test_environment_requirements_compares_live_fingerprint_as_canonical_json(
    tmp_path: Path,
):
    requirements = tmp_path / "requirements-lock.txt"
    requirements.write_text("alpha==1\n", encoding="utf-8")
    fingerprint = {
        "installed_distributions": [{"name": "alpha", "version": "1"}],
        "gpu": {"available": False},
    }
    lock = {
        "schema_version": 1,
        "fingerprint": fingerprint,
        "fingerprint_sha256": identity.sha256_bytes(
            identity.canonical_json_bytes(fingerprint)
        ),
    }
    environment_lock = tmp_path / "environment-lock.json"
    environment_lock.write_bytes(identity.canonical_json_bytes(lock))
    python_equal_but_json_distinct = {
        "installed_distributions": [{"name": "alpha", "version": "1"}],
        "gpu": {"available": 0},
    }
    assert python_equal_but_json_distinct == fingerprint

    with pytest.raises(ValueError, match="exactly"):
        identity.verify_environment_requirements(
            requirements,
            environment_lock,
            live_fingerprint=python_equal_but_json_distinct,
        )


def test_environment_requirements_rejects_requirement_projection_mismatch(
    tmp_path: Path,
):
    requirements = tmp_path / "requirements-lock.txt"
    requirements.write_text("alpha==2\n", encoding="utf-8")
    fingerprint = {
        "installed_distributions": [{"name": "alpha", "version": "1"}]
    }
    lock = {
        "schema_version": 1,
        "fingerprint": fingerprint,
        "fingerprint_sha256": identity.sha256_bytes(
            identity.canonical_json_bytes(fingerprint)
        ),
    }
    environment_lock = tmp_path / "environment-lock.json"
    environment_lock.write_bytes(identity.canonical_json_bytes(lock))

    with pytest.raises(ValueError, match="requirements lock"):
        identity.verify_environment_requirements(
            requirements,
            environment_lock,
            live_fingerprint=fingerprint,
        )


@pytest.mark.parametrize("schema_version", [True, "1"])
def test_environment_requirements_rejects_lock_schema_type_spoofs(
    tmp_path: Path,
    schema_version: object,
):
    requirements = tmp_path / "requirements-lock.txt"
    requirements.write_text("alpha==1\n", encoding="utf-8")
    fingerprint = {
        "installed_distributions": [{"name": "alpha", "version": "1"}]
    }
    lock = {
        "schema_version": schema_version,
        "fingerprint": fingerprint,
        "fingerprint_sha256": identity.sha256_bytes(
            identity.canonical_json_bytes(fingerprint)
        ),
    }
    environment_lock = tmp_path / "environment-lock.json"
    environment_lock.write_bytes(identity.canonical_json_bytes(lock))

    with pytest.raises(ValueError, match="schema version"):
        identity.verify_environment_requirements(
            requirements,
            environment_lock,
            live_fingerprint=fingerprint,
        )


def test_environment_requirements_rejects_stored_fingerprint_hash_mismatch(
    tmp_path: Path,
):
    requirements = tmp_path / "requirements-lock.txt"
    requirements.write_text("alpha==1\n", encoding="utf-8")
    fingerprint = {
        "installed_distributions": [{"name": "alpha", "version": "1"}]
    }
    lock = {
        "schema_version": 1,
        "fingerprint": fingerprint,
        "fingerprint_sha256": "0" * 64,
    }
    environment_lock = tmp_path / "environment-lock.json"
    environment_lock.write_bytes(identity.canonical_json_bytes(lock))

    with pytest.raises(ValueError, match="hash mismatch"):
        identity.verify_environment_requirements(
            requirements,
            environment_lock,
            live_fingerprint=fingerprint,
        )


def test_requirements_projection_rejects_normalized_conflicting_versions(
    tmp_path: Path,
):
    requirements = tmp_path / "requirements-lock.txt"
    requirements.write_text(
        "Alpha_Pkg==1\nalpha-pkg==2\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="conflicting versions"):
        identity.requirements_dependencies_sha256(requirements)


def test_requirements_projection_collapses_exact_normalized_duplicates(
    tmp_path: Path,
):
    requirements = tmp_path / "requirements-lock.txt"
    requirements.write_text(
        "Alpha_Pkg==1\nalpha-pkg==1\n",
        encoding="utf-8",
    )

    assert identity.requirements_dependencies_sha256(
        requirements
    ) == hashlib.sha256(
        b'[{"name":"alpha-pkg","version":"1"}]'
    ).hexdigest()


def test_environment_requirements_rejects_live_mapping_subclass(
    tmp_path: Path,
):
    class FingerprintSubclass(dict):
        pass

    requirements = tmp_path / "requirements-lock.txt"
    requirements.write_text("alpha==1\n", encoding="utf-8")
    fingerprint = {
        "installed_distributions": [{"name": "alpha", "version": "1"}]
    }
    lock = {
        "schema_version": 1,
        "fingerprint": fingerprint,
        "fingerprint_sha256": identity.sha256_bytes(
            identity.canonical_json_bytes(fingerprint)
        ),
    }
    environment_lock = tmp_path / "environment-lock.json"
    environment_lock.write_bytes(identity.canonical_json_bytes(lock))

    with pytest.raises(TypeError, match="unsupported JSON type"):
        identity.verify_environment_requirements(
            requirements,
            environment_lock,
            live_fingerprint=FingerprintSubclass(fingerprint),
        )


def test_identity_rejects_string_and_mapping_subclasses_at_cryptographic_boundary():
    class StringSubclass(str):
        pass

    class MappingSubclass(dict):
        pass

    with pytest.raises(TypeError, match="scientific_contract_id"):
        identity.build_run_identity(
            **{**_identity_kwargs(), "scientific_contract_id": StringSubclass("gsdiff-sim-v1")}
        )
    with pytest.raises(TypeError, match="assets_sha256"):
        identity.build_run_identity(
            **{**_identity_kwargs(), "assets_sha256": MappingSubclass()}
        )
    with pytest.raises(TypeError, match="code_commit"):
        identity.build_run_identity(
            **{**_identity_kwargs(), "code_commit": StringSubclass("c" * 40)}
        )


def test_resolved_config_rejects_custom_container_subclasses():
    class MappingSubclass(dict):
        pass

    class ListSubclass(list):
        pass

    with pytest.raises(TypeError):
        identity.resolved_config_sha256(MappingSubclass())
    with pytest.raises(TypeError):
        identity.resolved_config_sha256({"values": ListSubclass([1])})
