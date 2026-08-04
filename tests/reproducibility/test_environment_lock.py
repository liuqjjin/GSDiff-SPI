from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import gsdiff.experiments.identity as identity
from gsdiff.experiments.identity import (
    canonical_json_bytes,
    collect_environment_fingerprint,
    sha256_bytes,
)
import pytest

import scripts.reproducibility.verify_environment_lock as verifier


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class _Distribution:
    def __init__(self, name: str, version: str):
        self.metadata = {"Name": name}
        self.version = version


def _fingerprint_hash(value: object) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _normalized_requirements(path: Path) -> dict[str, str]:
    requirements = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        name, separator, version = line.partition("==")
        if separator:
            normalized = name.lower().replace("_", "-").replace(".", "-")
            assert normalized not in requirements
            requirements[normalized] = version
    return requirements


def test_authoritative_locks_use_vendored_recinr_without_stale_distributions():
    vendored_modules = {
        "recinr": REPOSITORY_ROOT / "gsdiff" / "baselines" / "recinr.py",
        "recinr-model": (
            REPOSITORY_ROOT / "gsdiff" / "baselines" / "recinr_model.py"
        ),
    }
    assert all(path.is_file() for path in vendored_modules.values())

    environment_lock = json.loads(
        (
            REPOSITORY_ROOT
            / "docs"
            / "reproducibility"
            / "environment-lock.json"
        ).read_text(encoding="utf-8")
    )
    requirements = _normalized_requirements(
        REPOSITORY_ROOT / "requirements-lock.txt"
    )
    installed = {
        record["name"]
        for record in environment_lock["fingerprint"]["installed_distributions"]
    }
    stale_distributions = {
        "inr-spi",
        "recinr",
        "recinr-rebuild",
        "build",
        "pyproject-hooks",
    }

    assert stale_distributions.isdisjoint(requirements)
    assert stale_distributions.isdisjoint(installed)


def test_distribution_records_are_normalized_and_sorted(monkeypatch):
    records = [
        _Distribution("Zed_Package", "2.0"),
        _Distribution("alpha.package", "1.0"),
        _Distribution("zed-package", "1.0"),
    ]
    monkeypatch.setattr(
        identity.importlib_metadata, "distributions", lambda: iter(records)
    )
    first = collect_environment_fingerprint()
    monkeypatch.setattr(
        identity.importlib_metadata, "distributions", lambda: iter(reversed(records))
    )
    second = collect_environment_fingerprint()

    expected = [
        {"name": "alpha-package", "version": "1.0"},
        {"name": "zed-package", "version": "1.0"},
        {"name": "zed-package", "version": "2.0"},
    ]
    assert first["installed_distributions"] == expected
    assert second["installed_distributions"] == expected
    assert canonical_json_bytes(first) == canonical_json_bytes(second)


def test_fingerprint_includes_required_runtime_dimensions():
    fingerprint = collect_environment_fingerprint()

    assert fingerprint["python"]["implementation"]
    assert fingerprint["python"]["abi"]["cache_tag"]
    assert "soabi" in fingerprint["python"]["abi"]
    assert fingerprint["platform"]["system"]
    assert fingerprint["platform"]["machine"]
    assert fingerprint["pytorch"]["version"]
    assert "cuda_build" in fingerprint["pytorch"]
    assert "driver_version" in fingerprint["gpu"]
    assert "devices" in fingerprint["gpu"]
    assert "CUBLAS_WORKSPACE_CONFIG" in fingerprint["numerical_environment"]
    assert "OMP_NUM_THREADS" in fingerprint["numerical_environment"]


def test_dependency_abi_and_numerical_environment_mutations_change_hash(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    fingerprint = collect_environment_fingerprint()
    original_hash = _fingerprint_hash(fingerprint)

    dependency_mutation = deepcopy(fingerprint)
    dependency_mutation["installed_distributions"][0]["version"] += ".changed"
    abi_mutation = deepcopy(fingerprint)
    abi_mutation["python"]["abi"]["cache_tag"] += "-changed"
    numerical_env_mutation = deepcopy(fingerprint)
    numerical_env_mutation["numerical_environment"]["OMP_NUM_THREADS"] = "4"

    assert _fingerprint_hash(dependency_mutation) != original_hash
    assert _fingerprint_hash(abi_mutation) != original_hash
    assert _fingerprint_hash(numerical_env_mutation) != original_hash


def test_unknown_secret_bearing_environment_variables_are_not_captured(monkeypatch):
    secrets = {
        "AWS_SECRET_ACCESS_KEY": "do-not-record-aws",
        "DATABASE_URL": "do-not-record-db",
        "GSDIFF_TEST_SECRET": "do-not-record-custom",
    }
    for key, value in secrets.items():
        monkeypatch.setenv(key, value)

    encoded = canonical_json_bytes(collect_environment_fingerprint()).decode("utf-8")

    for key, value in secrets.items():
        assert key not in encoded
        assert value not in encoded


def _write_lock(path, lock):
    path.write_bytes(canonical_json_bytes(lock) + b"\n")


def _current_lock():
    fingerprint = collect_environment_fingerprint()
    return verifier.make_environment_lock(fingerprint), fingerprint


def _minimal_fingerprint(distributions):
    return {
        "gpu": {},
        "installed_distributions": distributions,
        "numerical_environment": {},
        "platform": {},
        "python": {},
        "pytorch": {},
    }


def test_strict_cli_rejects_self_consistent_json_with_requirements_mismatch(
    tmp_path, monkeypatch, capsys
):
    fingerprint = _minimal_fingerprint(
        [{"name": "example-package", "version": "1.0"}]
    )
    environment_path = tmp_path / "environment-lock.json"
    requirements_path = tmp_path / "requirements-lock.txt"
    _write_lock(environment_path, verifier.make_environment_lock(fingerprint))
    requirements_path.write_text("example-package==2.0\n", encoding="utf-8")
    monkeypatch.setattr(
        verifier, "collect_environment_fingerprint", lambda: fingerprint
    )

    exit_code = verifier.main(
        [
            str(environment_path),
            "--strict",
            "--requirements-lock",
            str(requirements_path),
        ]
    )

    assert exit_code == 1
    assert "requirements lock mismatch" in capsys.readouterr().err


def test_strict_cli_accepts_matching_requirements_environment_and_live_runtime(
    tmp_path, monkeypatch, capsys
):
    fingerprint = _minimal_fingerprint(
        [{"name": "example-package", "version": "1.0"}]
    )
    environment_path = tmp_path / "environment-lock.json"
    requirements_path = tmp_path / "requirements-lock.txt"
    _write_lock(environment_path, verifier.make_environment_lock(fingerprint))
    requirements_path.write_text("Example_Package==1.0\n", encoding="utf-8")
    monkeypatch.setattr(
        verifier, "collect_environment_fingerprint", lambda: fingerprint
    )

    exit_code = verifier.main(
        [
            str(environment_path),
            "--strict",
            "--requirements-lock",
            str(requirements_path),
        ]
    )

    assert exit_code == 0
    assert "environment_lock_verification=passed" in capsys.readouterr().out


def test_environment_lock_accepts_valid_current_fingerprint(tmp_path, monkeypatch):
    lock, fingerprint = _current_lock()
    path = tmp_path / "environment-lock.json"
    _write_lock(path, lock)
    monkeypatch.setattr(verifier, "collect_environment_fingerprint", lambda: fingerprint)

    summary = verifier.verify_environment_lock(path, strict=True)

    assert summary["fingerprint_sha256"] == lock["fingerprint_sha256"]


def test_environment_lock_writer_uses_canonical_json(tmp_path):
    _, fingerprint = _current_lock()
    path = tmp_path / "environment-lock.json"

    lock = verifier.write_environment_lock(path, fingerprint=fingerprint)

    assert path.read_bytes() == canonical_json_bytes(lock) + b"\n"
    assert verifier.verify_environment_lock(path, strict=False)


def test_environment_lock_rejects_missing_file(tmp_path):
    with pytest.raises(verifier.EnvironmentLockError, match="does not exist"):
        verifier.verify_environment_lock(
            tmp_path / "missing-environment-lock.json", strict=True
        )


def test_environment_lock_rejects_malformed_json(tmp_path):
    path = tmp_path / "environment-lock.json"
    path.write_text("{", encoding="utf-8")

    with pytest.raises(verifier.EnvironmentLockError, match="valid JSON"):
        verifier.verify_environment_lock(path, strict=True)


def test_environment_lock_rejects_invalid_payload(tmp_path):
    path = tmp_path / "environment-lock.json"
    path.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")

    with pytest.raises(verifier.EnvironmentLockError, match="fingerprint"):
        verifier.verify_environment_lock(path, strict=True)


def test_environment_lock_rejects_boolean_schema_version(tmp_path):
    lock, _ = _current_lock()
    lock["schema_version"] = True
    path = tmp_path / "environment-lock.json"
    _write_lock(path, lock)

    with pytest.raises(verifier.EnvironmentLockError, match="expected"):
        verifier.verify_environment_lock(path, strict=False)


def test_environment_lock_rejects_hash_mismatch(tmp_path):
    lock, _ = _current_lock()
    lock["fingerprint_sha256"] = "0" * 64
    path = tmp_path / "environment-lock.json"
    _write_lock(path, lock)

    with pytest.raises(verifier.EnvironmentLockError, match="hash mismatch"):
        verifier.verify_environment_lock(path, strict=False)


@pytest.mark.parametrize(
    ("mutate", "field_name"),
    [
        (
            lambda value: value["installed_distributions"][0].__setitem__(
                "version", "changed"
            ),
            "installed_distributions",
        ),
        (
            lambda value: value["python"]["abi"].__setitem__("cache_tag", "changed"),
            "python",
        ),
        (
            lambda value: value["numerical_environment"].__setitem__(
                "OMP_NUM_THREADS", "changed"
            ),
            "numerical_environment",
        ),
    ],
)
def test_strict_environment_lock_rejects_current_state_mismatch(
    tmp_path, monkeypatch, mutate, field_name
):
    lock, current = _current_lock()
    mutate(lock["fingerprint"])
    lock["fingerprint_sha256"] = _fingerprint_hash(lock["fingerprint"])
    path = tmp_path / "environment-lock.json"
    _write_lock(path, lock)
    monkeypatch.setattr(verifier, "collect_environment_fingerprint", lambda: current)

    with pytest.raises(verifier.EnvironmentLockError, match=field_name):
        verifier.verify_environment_lock(path, strict=True)


def test_strict_environment_lock_uses_canonical_json_equality(
    tmp_path, monkeypatch
):
    lock, current = _current_lock()
    lock["fingerprint"]["gpu"]["available"] = False
    lock["fingerprint_sha256"] = _fingerprint_hash(lock["fingerprint"])
    current["gpu"]["available"] = 0
    assert current == lock["fingerprint"]
    path = tmp_path / "environment-lock.json"
    _write_lock(path, lock)
    monkeypatch.setattr(
        verifier,
        "collect_environment_fingerprint",
        lambda: current,
    )

    with pytest.raises(verifier.EnvironmentLockError, match="gpu"):
        verifier.verify_environment_lock(path, strict=True)
