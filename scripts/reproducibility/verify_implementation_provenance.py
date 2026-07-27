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
    required = {"repository", "worktree_detection", "immutable_inputs"}
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
        root, repository.get("plan_baseline_commit"), "plan_baseline_commit"
    )
    starting_head = _verify_commit(root, repository.get("starting_head"), "starting_head")
    if baseline != starting_head:
        raise ProvenanceError(
            "starting_head must equal plan_baseline_commit for the recorded baseline"
        )

    verified_paths = set()
    for item in immutable_inputs:
        if not isinstance(item, dict):
            raise ProvenanceError("immutable input entries must be objects")
        relative = item.get("path")
        recorded_hash = item.get("sha256")
        if not isinstance(relative, str) or not isinstance(recorded_hash, str):
            raise ProvenanceError("immutable input path and sha256 must be strings")
        if not re.fullmatch(r"[0-9a-f]{64}", recorded_hash):
            raise ProvenanceError(f"invalid SHA-256 for immutable input: {relative}")
        document_path = (root / relative).resolve()
        try:
            document_path.relative_to(root)
        except ValueError as exc:
            raise ProvenanceError(f"immutable input escapes repository: {relative}") from exc
        if relative in verified_paths:
            raise ProvenanceError(f"duplicate immutable input: {relative}")
        verified_paths.add(relative)
        if not document_path.is_file():
            raise ProvenanceError(f"immutable input does not exist: {relative}")
        current_hash = hashlib.sha256(document_path.read_bytes()).hexdigest()
        if current_hash != recorded_hash:
            raise ProvenanceError(f"SHA-256 mismatch for immutable input: {relative}")
        baseline_payload = _git(root, ["show", f"{baseline}:{relative}"], text=False)
        baseline_hash = hashlib.sha256(baseline_payload).hexdigest()
        if baseline_hash != recorded_hash:
            raise ProvenanceError(
                f"baseline SHA-256 mismatch for immutable input: {relative}"
            )

    current_commit = str(_git(root, ["rev-parse", "HEAD"]))
    if strict:
        actual_root = str(_git(root, ["rev-parse", "--show-toplevel"]))
        if not _same_path(repository.get("worktree_path", ""), actual_root):
            raise ProvenanceError(
                "worktree mismatch between recorded and current repository"
            )
        actual_branch = str(_git(root, ["branch", "--show-current"]))
        if repository.get("implementation_branch") != actual_branch:
            raise ProvenanceError(
                "branch mismatch between recorded and current repository"
            )

        legacy = _verify_commit(
            root,
            repository.get("legacy_evidence_baseline"),
            "legacy_evidence_baseline",
        )
        design = _verify_commit(
            root, repository.get("approved_design_commit"), "approved_design_commit"
        )
        _verify_ancestor(root, legacy, starting_head, "legacy baseline -> starting_head")
        _verify_ancestor(root, design, starting_head, "approved design -> starting_head")
        _verify_ancestor(root, starting_head, current_commit, "starting_head -> current HEAD")

        detection = provenance["worktree_detection"]
        if not isinstance(detection, dict):
            raise ProvenanceError("worktree_detection must be an object")
        actual_git_dir = str(_git(root, ["rev-parse", "--path-format=absolute", "--git-dir"]))
        actual_common_dir = str(
            _git(root, ["rev-parse", "--path-format=absolute", "--git-common-dir"])
        )
        if not _same_path(detection.get("resolved_git_dir", ""), actual_git_dir):
            raise ProvenanceError("worktree Git directory mismatch")
        if not _same_path(detection.get("resolved_git_common_dir", ""), actual_common_dir):
            raise ProvenanceError("worktree common Git directory mismatch")

    return {
        "current_commit": current_commit,
        "immutable_inputs_verified": len(immutable_inputs),
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
