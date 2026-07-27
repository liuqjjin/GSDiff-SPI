from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROVENANCE_PATH = (
    REPOSITORY_ROOT
    / "docs"
    / "reproducibility"
    / "implementation-provenance.json"
)
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
EXPECTED_IMMUTABLE_INPUTS = {
    "docs/superpowers/specs/2026-07-27-gsdiff-correctness-publication-design.md": (
        "60c0a32d8ecf6b54734544a1862b939940a0ff5215546f224e5dab889de6d12a"
    ),
    "docs/superpowers/plans/2026-07-27-gsdiff-correctness-reproducibility.md": (
        "3e789c278f74fad707f343e793c5eae4b3c808ff89779b16a79a0c86044f68b7"
    ),
    "docs/superpowers/plans/2026-07-27-gsdiff-experiments-artifacts.md": (
        "a267afd2b270742bb7364eff8e1c23d92a57e310de30ec953bc0a1f5436bcf0e"
    ),
    "docs/superpowers/plans/2026-07-27-gsdiff-publication-package.md": (
        "a52be20bc7b456de0c77219f45c00e389968a61e9370789a9fa1499c579f830f"
    ),
}
EXPECTED_LEGACY_BASELINE = "c03420784bc92b4e9b9eef8330cbd9571ebebc68"
EXPECTED_APPROVED_DESIGN_COMMIT = "24c1959599d9d775114d068f6de41ef2e31b5e36"
EXPECTED_PLAN_BASELINE = "abca49b36439efc6cb607a45c1601e50d84d6656"
EXPECTED_FIRST_TASK0_COMMIT = "d61aec434caaaecb334f7486a450efd4d6fcc2e5"
EXPECTED_TASK0_TERMINAL_COMMIT = "2e53688a07ea619e6409a60cbaba53f5ca6cb385"


class ProvenanceError(RuntimeError):
    pass


def _load_provenance(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise ProvenanceError(f"implementation provenance does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProvenanceError(
            f"implementation provenance is not valid JSON: {path}"
        ) from exc
    if not isinstance(value, dict):
        raise ProvenanceError("implementation provenance must be a JSON object")
    required = {
        "repository",
        "worktree_decision",
        "worktree_detection",
        "immutable_inputs",
    }
    missing = sorted(required - set(value))
    if missing:
        raise ProvenanceError(
            "implementation provenance missing fields: " + ", ".join(missing)
        )
    return value


def _git(
    repo_root: Path, arguments: list[str], *, text: bool = True
) -> str | bytes:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=text,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        raise ProvenanceError(
            f"Git verification failed for: git {' '.join(arguments)}"
        ) from exc
    return result.stdout.strip() if text else result.stdout


def _same_path(left: str | Path, right: str | Path) -> bool:
    return os.path.normcase(str(Path(left).resolve())) == os.path.normcase(
        str(Path(right).resolve())
    )


def _verify_commit(repo_root: Path, commit: object, field: str) -> str:
    if not isinstance(commit, str) or not _COMMIT_PATTERN.fullmatch(commit):
        raise ProvenanceError(f"{field} is not a full 40-character commit")
    _git(repo_root, ["cat-file", "-e", f"{commit}^{{commit}}"])
    return commit


def _verify_ancestor(repo_root: Path, ancestor: str, descendant: str, label: str) -> None:
    try:
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except subprocess.CalledProcessError as exc:
        raise ProvenanceError(f"commit relationship mismatch: {label}") from exc
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        raise ProvenanceError(f"Git verification failed: {label}") from exc


def _require_exact_field(
    payload: dict[str, object], field: str, expected: object
) -> object:
    if field not in payload:
        raise ProvenanceError(f"missing required field: {field}")
    if payload[field] != expected:
        raise ProvenanceError(
            f"{field} does not match the exact Task-0 provenance anchor "
            f"{expected!r}"
        )
    return payload[field]


def _verify_direct_parent(
    repo_root: Path, commit: str, expected_parent: str, label: str
) -> None:
    parent = str(_git(repo_root, ["rev-parse", f"{commit}^"]))
    if parent != expected_parent:
        raise ProvenanceError(
            f"{label} parent mismatch: expected {expected_parent}, observed {parent}"
        )


def verify_implementation_provenance(
    path: Path | str = DEFAULT_PROVENANCE_PATH,
    *,
    repo_root: Path | str = REPOSITORY_ROOT,
    strict: bool,
) -> dict[str, object]:
    provenance = _load_provenance(Path(path))
    root = Path(repo_root).resolve()
    repository = provenance["repository"]
    immutable_inputs = provenance["immutable_inputs"]
    if not isinstance(repository, dict) or not isinstance(immutable_inputs, list):
        raise ProvenanceError("invalid repository or immutable_inputs payload")
    if len(immutable_inputs) != 4:
        raise ProvenanceError("immutable_inputs must contain exactly four documents")

    baseline = _verify_commit(
        root,
        _require_exact_field(
            repository, "plan_baseline_commit", EXPECTED_PLAN_BASELINE
        ),
        "plan_baseline_commit",
    )
    starting_head = _verify_commit(
        root,
        _require_exact_field(repository, "starting_head", EXPECTED_PLAN_BASELINE),
        "starting_head",
    )
    legacy = _verify_commit(
        root,
        _require_exact_field(
            repository, "legacy_evidence_baseline", EXPECTED_LEGACY_BASELINE
        ),
        "legacy_evidence_baseline",
    )
    design = _verify_commit(
        root,
        _require_exact_field(
            repository, "approved_design_commit", EXPECTED_APPROVED_DESIGN_COMMIT
        ),
        "approved_design_commit",
    )

    recorded_inputs = {}
    for item in immutable_inputs:
        if not isinstance(item, dict):
            raise ProvenanceError("immutable input entries must be objects")
        relative = item.get("path")
        recorded_hash = item.get("sha256")
        if not isinstance(relative, str) or not isinstance(recorded_hash, str):
            raise ProvenanceError("immutable input path and sha256 must be strings")
        if not re.fullmatch(r"[0-9a-f]{64}", recorded_hash):
            raise ProvenanceError(f"invalid SHA-256 for immutable input: {relative}")
        if relative in recorded_inputs:
            raise ProvenanceError(f"duplicate immutable input: {relative}")
        recorded_inputs[relative] = recorded_hash

    if set(recorded_inputs) != set(EXPECTED_IMMUTABLE_INPUTS):
        raise ProvenanceError(
            "exact immutable input paths do not match the Task-0 anchors"
        )
    for relative, expected_hash in EXPECTED_IMMUTABLE_INPUTS.items():
        recorded_hash = recorded_inputs[relative]
        if recorded_hash != expected_hash:
            raise ProvenanceError(
                f"SHA-256 mismatch for exact immutable input: {relative}"
            )
        document_path = (root / relative).resolve()
        try:
            document_path.relative_to(root)
        except ValueError as exc:
            raise ProvenanceError(f"immutable input escapes repository: {relative}") from exc
        if not document_path.is_file():
            raise ProvenanceError(f"immutable input does not exist: {relative}")
        current_hash = hashlib.sha256(document_path.read_bytes()).hexdigest()
        if current_hash != expected_hash:
            raise ProvenanceError(f"SHA-256 mismatch for immutable input: {relative}")
        baseline_payload = _git(root, ["show", f"{baseline}:{relative}"], text=False)
        baseline_hash = hashlib.sha256(baseline_payload).hexdigest()
        if baseline_hash != expected_hash:
            raise ProvenanceError(
                f"baseline SHA-256 mismatch for immutable input: {relative}"
            )

    current_commit = str(_git(root, ["rev-parse", "HEAD"]))
    if strict:
        first_task0 = _verify_commit(
            root, EXPECTED_FIRST_TASK0_COMMIT, "first Task-0 provenance commit"
        )
        terminal_task0 = _verify_commit(
            root, EXPECTED_TASK0_TERMINAL_COMMIT, "Task-0 terminal commit"
        )
        _verify_direct_parent(
            root, first_task0, baseline, "first Task-0 provenance commit"
        )
        _verify_direct_parent(
            root, terminal_task0, first_task0, "Task-0 terminal commit"
        )
        _verify_ancestor(
            root,
            terminal_task0,
            current_commit,
            "Task-0 terminal commit -> current HEAD",
        )

        actual_root = str(_git(root, ["rev-parse", "--show-toplevel"]))
        if "worktree_path" not in repository:
            raise ProvenanceError("missing required field: worktree_path")
        if not _same_path(repository.get("worktree_path", ""), actual_root):
            raise ProvenanceError(
                "worktree mismatch between recorded and current repository"
            )
        actual_branch = str(_git(root, ["branch", "--show-current"]))
        if "implementation_branch" not in repository:
            raise ProvenanceError("missing required field: implementation_branch")
        if repository.get("implementation_branch") != actual_branch:
            raise ProvenanceError(
                "branch mismatch between recorded and current repository"
            )

        _verify_ancestor(root, legacy, starting_head, "legacy baseline -> starting_head")
        _verify_ancestor(root, design, starting_head, "approved design -> starting_head")

        detection = provenance["worktree_detection"]
        decision = provenance["worktree_decision"]
        if not isinstance(detection, dict) or not isinstance(decision, dict):
            raise ProvenanceError(
                "worktree_decision and worktree_detection must be objects"
            )
        _require_exact_field(decision, "mode", "in_place")
        if "is_linked_worktree" not in detection:
            raise ProvenanceError("missing required field: is_linked_worktree")
        if detection["is_linked_worktree"] is not False:
            raise ProvenanceError("recorded checkout must be a normal non-linked-worktree")
        if "is_submodule" not in detection:
            raise ProvenanceError("missing required field: is_submodule")
        if detection["is_submodule"] is not False:
            raise ProvenanceError("recorded checkout must not be a submodule")
        if detection.get("superproject_worktree") != "":
            raise ProvenanceError("recorded superproject_worktree must be empty")

        registered = detection.get("registered_worktrees")
        if (
            not isinstance(registered, list)
            or len(registered) != 1
            or not isinstance(registered[0], dict)
        ):
            raise ProvenanceError(
                "recorded worktree path/branch relationship mismatch"
            )
        registered_entry = registered[0]
        if (
            not _same_path(registered_entry.get("path", ""), repository["worktree_path"])
            or registered_entry.get("head") != starting_head
            or registered_entry.get("branch")
            != f"refs/heads/{repository['implementation_branch']}"
        ):
            raise ProvenanceError(
                "recorded worktree path/branch relationship mismatch"
            )

        actual_superproject = str(
            _git(root, ["rev-parse", "--show-superproject-working-tree"])
        )
        if actual_superproject:
            raise ProvenanceError("current checkout is a submodule")
        actual_git_dir = str(
            _git(root, ["rev-parse", "--path-format=absolute", "--git-dir"])
        )
        actual_common_dir = str(
            _git(root, ["rev-parse", "--path-format=absolute", "--git-common-dir"])
        )
        if not _same_path(actual_git_dir, actual_common_dir):
            raise ProvenanceError("current checkout is a linked-worktree")
        if not _same_path(detection.get("resolved_git_dir", ""), actual_git_dir):
            raise ProvenanceError("worktree Git directory mismatch")
        if not _same_path(detection.get("resolved_git_common_dir", ""), actual_common_dir):
            raise ProvenanceError("worktree common Git directory mismatch")

    return {
        "current_commit": current_commit,
        "immutable_inputs_verified": len(recorded_inputs),
        "strict": strict,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify immutable implementation provenance and Git relationships."
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=DEFAULT_PROVENANCE_PATH,
        help="implementation-provenance.json path",
    )
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        summary = verify_implementation_provenance(
            args.path, repo_root=REPOSITORY_ROOT, strict=args.strict
        )
    except ProvenanceError as exc:
        print(f"implementation_provenance_verification=failed: {exc}", file=sys.stderr)
        return 1
    print(
        "implementation_provenance_verification=passed "
        f"strict={str(args.strict).lower()} "
        f"immutable_inputs={summary['immutable_inputs_verified']} "
        f"current_commit={summary['current_commit']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
