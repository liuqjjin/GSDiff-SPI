from __future__ import annotations

from pathlib import Path
import math
import sys

import pytest

from gsdiff.experiments.identity import (
    canonical_json_bytes,
    collect_runtime_metadata,
    sha256_bytes,
)


def test_runtime_metadata_has_reproducibility_fields():
    meta = collect_runtime_metadata()

    assert Path(meta["python_executable"]).resolve() == Path(sys.executable).resolve()
    assert meta["python_version"]
    assert meta["torch_version"]
    assert "cuda_version" in meta
    assert meta["os"]


def test_canonical_json_bytes_is_stable_and_compact():
    left = {"unicode": "测", "nested": {"z": 2, "a": 1}}
    right = {"nested": {"a": 1, "z": 2}, "unicode": "测"}

    assert canonical_json_bytes(left) == canonical_json_bytes(right)
    assert canonical_json_bytes(left) == (
        '{"nested":{"a":1,"z":2},"unicode":"测"}'.encode("utf-8")
    )


@pytest.mark.parametrize("non_finite", [math.nan, math.inf, -math.inf])
def test_canonical_json_bytes_rejects_non_finite_numbers(non_finite):
    with pytest.raises(ValueError):
        canonical_json_bytes({"value": non_finite})


def test_sha256_bytes_matches_standard_vector_and_detects_mutation():
    original = b"abc"
    mutated = b"abd"

    assert sha256_bytes(original) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
    assert sha256_bytes(original) != sha256_bytes(mutated)
