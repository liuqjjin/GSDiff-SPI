from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

import scripts.reproducibility.verify_implementation_provenance as verifier


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROVENANCE_PATH = (
    REPOSITORY_ROOT / "docs" / "reproducibility" / "implementation-provenance.json"
)


def _provenance():
    return json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))


def _write_provenance(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _git(repo, *arguments):
    return subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _commit(repo, message):
    _git(repo, "add", ".")
    _git(repo, "commit", "--allow-empty", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _controlled_history(tmp_path, monkeypatch, variant="valid"):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "debug/admm-vs-sgd")
    _git(repo, "config", "user.email", "tests@example.invalid")
    _git(repo, "config", "user.name", "Verifier Tests")

    anchors = {}
    for index in range(4):
        relative = f"docs/anchor-{index}.md"
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"immutable anchor {index}\n", encoding="utf-8")
        anchors[relative] = hashlib.sha256(path.read_bytes()).hexdigest()

    legacy = _commit(repo, "legacy")
    design = _commit(repo, "design")
    plan = _commit(repo, "plan baseline")
    if variant == "wrong_first_parent":
        _commit(repo, "unexpected commit before first Task-0 commit")
    first_task0 = _commit(repo, "first Task-0 commit")
    if variant == "wrong_terminal_parent":
        _commit(repo, "unexpected commit before Task-0 terminal")
    terminal_task0 = _commit(repo, "Task-0 terminal")
    if variant == "current_not_descendant":
        _git(repo, "switch", "-c", "alternate-current", first_task0)
    current = _commit(repo, "current")
    branch = _git(repo, "branch", "--show-current")

    monkeypatch.setattr(verifier, "EXPECTED_IMMUTABLE_INPUTS", anchors, raising=False)
    monkeypatch.setattr(verifier, "EXPECTED_LEGACY_BASELINE", legacy, raising=False)
    monkeypatch.setattr(verifier, "EXPECTED_APPROVED_DESIGN_COMMIT", design, raising=False)
    monkeypatch.setattr(verifier, "EXPECTED_PLAN_BASELINE", plan, raising=False)
    monkeypatch.setattr(verifier, "EXPECTED_FIRST_TASK0_COMMIT", first_task0, raising=False)
    monkeypatch.setattr(
        verifier, "EXPECTED_TASK0_TERMINAL_COMMIT", terminal_task0, raising=False
    )

    git_dir = str((repo / ".git").resolve())
    provenance = {
        "repository": {
            "worktree_path": str(repo.resolve()),
            "implementation_branch": branch,
            "starting_head": plan,
            "plan_baseline_commit": plan,
            "legacy_evidence_baseline": legacy,
            "approved_design_commit": design,
        },
        "worktree_detection": {
            "resolved_git_dir": git_dir,
            "resolved_git_common_dir": git_dir,
            "superproject_worktree": "",
            "registered_worktrees": [
                {
                    "path": str(repo.resolve()),
                    "head": plan,
                    "branch": f"refs/heads/{branch}",
                }
            ],
            "is_linked_worktree": False,
            "is_submodule": False,
        },
        "worktree_decision": {"mode": "in_place"},
        "immutable_inputs": [
            {"path": path, "sha256": digest} for path, digest in anchors.items()
        ],
    }
    provenance_path = repo / "provenance.json"
    _write_provenance(provenance_path, provenance)
    return repo, provenance_path, current


def test_strict_implementation_provenance_accepts_recorded_repository_state():
    summary = verifier.verify_implementation_provenance(
        PROVENANCE_PATH, repo_root=REPOSITORY_ROOT, strict=True
    )

    assert summary["immutable_inputs_verified"] == 4
    assert summary["current_commit"]


def test_implementation_provenance_rejects_missing_file(tmp_path):
    with pytest.raises(verifier.ProvenanceError, match="does not exist"):
        verifier.verify_implementation_provenance(
            tmp_path / "missing.json", repo_root=REPOSITORY_ROOT, strict=True
        )


def test_implementation_provenance_rejects_malformed_json(tmp_path):
    path = tmp_path / "provenance.json"
    path.write_text("{", encoding="utf-8")

    with pytest.raises(verifier.ProvenanceError, match="valid JSON"):
        verifier.verify_implementation_provenance(
            path, repo_root=REPOSITORY_ROOT, strict=True
        )


def test_strict_implementation_provenance_rejects_document_hash_mismatch(tmp_path):
    provenance = _provenance()
    provenance["immutable_inputs"][0]["sha256"] = "0" * 64
    path = tmp_path / "provenance.json"
    _write_provenance(path, provenance)

    with pytest.raises(verifier.ProvenanceError, match="SHA-256 mismatch"):
        verifier.verify_implementation_provenance(
            path, repo_root=REPOSITORY_ROOT, strict=True
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["repository"].__setitem__(
                "plan_baseline_commit", value["repository"]["legacy_evidence_baseline"]
            ),
            "plan_baseline_commit",
        ),
        (
            lambda value: value["repository"].__setitem__(
                "implementation_branch", "not-the-recorded-branch"
            ),
            "branch mismatch",
        ),
        (
            lambda value: value["repository"].__setitem__(
                "worktree_path", str(Path(value["repository"]["worktree_path"]) / "other")
            ),
            "worktree mismatch",
        ),
    ],
)
def test_strict_implementation_provenance_rejects_relationship_mismatch(
    tmp_path, mutation, message
):
    provenance = deepcopy(_provenance())
    mutation(provenance)
    path = tmp_path / "provenance.json"
    _write_provenance(path, provenance)

    with pytest.raises(verifier.ProvenanceError, match=message):
        verifier.verify_implementation_provenance(
            path, repo_root=REPOSITORY_ROOT, strict=True
        )


def test_strict_provenance_rejects_coordinated_anchor_substitution(tmp_path):
    provenance = _provenance()
    substituted_paths = [
        "README.md",
        "THEORY.md",
        "gsdiff/__init__.py",
        "requirements.txt",
    ]
    provenance["immutable_inputs"] = [
        {
            "path": relative,
            "sha256": hashlib.sha256(
                (REPOSITORY_ROOT / relative).read_bytes()
            ).hexdigest(),
        }
        for relative in substituted_paths
    ]
    path = tmp_path / "coordinated-substitution.json"
    _write_provenance(path, provenance)

    with pytest.raises(verifier.ProvenanceError, match="exact immutable"):
        verifier.verify_implementation_provenance(
            path, repo_root=REPOSITORY_ROOT, strict=True
        )


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        (
            "legacy_evidence_baseline",
            "24c1959599d9d775114d068f6de41ef2e31b5e36",
            "legacy_evidence_baseline",
        ),
        (
            "approved_design_commit",
            "c03420784bc92b4e9b9eef8330cbd9571ebebc68",
            "approved_design_commit",
        ),
    ],
)
def test_strict_provenance_rejects_exact_commit_substitution(
    tmp_path, field, replacement, message
):
    provenance = _provenance()
    provenance["repository"][field] = replacement
    path = tmp_path / "commit-substitution.json"
    _write_provenance(path, provenance)

    with pytest.raises(verifier.ProvenanceError, match=message):
        verifier.verify_implementation_provenance(
            path, repo_root=REPOSITORY_ROOT, strict=True
        )


@pytest.mark.parametrize(
    "replacement",
    [
        "d61aec434caaaecb334f7486a450efd4d6fcc2e5",
        "2e53688a07ea619e6409a60cbaba53f5ca6cb385",
    ],
)
def test_strict_provenance_rejects_coordinated_start_and_plan_substitution(
    tmp_path, replacement
):
    provenance = _provenance()
    provenance["repository"]["starting_head"] = replacement
    provenance["repository"]["plan_baseline_commit"] = replacement
    path = tmp_path / "baseline-substitution.json"
    _write_provenance(path, provenance)

    with pytest.raises(verifier.ProvenanceError, match="plan_baseline_commit"):
        verifier.verify_implementation_provenance(
            path, repo_root=REPOSITORY_ROOT, strict=True
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.pop("worktree_decision"), "worktree_decision"),
        (
            lambda value: value["worktree_decision"].__setitem__(
                "mode", "linked_worktree"
            ),
            "in_place",
        ),
        (
            lambda value: value["worktree_detection"].__setitem__(
                "is_linked_worktree", True
            ),
            "linked-worktree",
        ),
        (
            lambda value: value["worktree_detection"].pop("is_linked_worktree"),
            "is_linked_worktree",
        ),
        (
            lambda value: value["worktree_detection"].__setitem__(
                "is_submodule", True
            ),
            "submodule",
        ),
        (
            lambda value: value["worktree_detection"].pop("is_submodule"),
            "is_submodule",
        ),
    ],
)
def test_strict_provenance_rejects_missing_or_forged_worktree_facts(
    tmp_path, mutation, message
):
    provenance = _provenance()
    mutation(provenance)
    path = tmp_path / "worktree-facts.json"
    _write_provenance(path, provenance)

    with pytest.raises(verifier.ProvenanceError, match=message):
        verifier.verify_implementation_provenance(
            path, repo_root=REPOSITORY_ROOT, strict=True
        )


def test_strict_provenance_accepts_controlled_exact_history(
    tmp_path, monkeypatch
):
    repo, path, current = _controlled_history(tmp_path, monkeypatch)

    summary = verifier.verify_implementation_provenance(
        path, repo_root=repo, strict=True
    )

    assert summary["current_commit"] == current


@pytest.mark.parametrize(
    ("variant", "message"),
    [
        ("wrong_first_parent", "first Task-0"),
        ("wrong_terminal_parent", "Task-0 terminal"),
        ("current_not_descendant", "Task-0 terminal.*current HEAD"),
    ],
)
def test_strict_provenance_rejects_task0_parent_and_ancestry_mismatch(
    tmp_path, monkeypatch, variant, message
):
    repo, path, _ = _controlled_history(tmp_path, monkeypatch, variant=variant)

    with pytest.raises(verifier.ProvenanceError, match=message):
        verifier.verify_implementation_provenance(path, repo_root=repo, strict=True)
