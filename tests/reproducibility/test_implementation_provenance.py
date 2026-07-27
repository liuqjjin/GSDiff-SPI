from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

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
            "starting_head",
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
