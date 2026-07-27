from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Sequence

from jsonschema import Draft202012Validator

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gsdiff.experiments.identity import (  # noqa: E402
    canonical_json_bytes,
    collect_environment_fingerprint,
    sha256_bytes,
)


DEFAULT_LOCK_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "reproducibility"
    / "environment-lock.json"
)

_LOCK_SCHEMA = {
    "type": "object",
    "required": ["schema_version", "fingerprint", "fingerprint_sha256"],
    "additionalProperties": False,
    "properties": {
        "schema_version": {"const": 1},
        "fingerprint": {
            "type": "object",
            "required": [
                "gpu",
                "installed_distributions",
                "numerical_environment",
                "platform",
                "python",
                "pytorch",
            ],
        },
        "fingerprint_sha256": {
            "type": "string",
            "pattern": "^[0-9a-f]{64}$",
        },
    },
}


class EnvironmentLockError(RuntimeError):
    pass


def make_environment_lock(
    fingerprint: dict[str, object] | None = None,
) -> dict[str, object]:
    fingerprint = deepcopy(fingerprint or collect_environment_fingerprint())
    return {
        "fingerprint": fingerprint,
        "fingerprint_sha256": sha256_bytes(canonical_json_bytes(fingerprint)),
        "schema_version": 1,
    }


def write_environment_lock(
    path: Path | str = DEFAULT_LOCK_PATH,
    *,
    fingerprint: dict[str, object] | None = None,
) -> dict[str, object]:
    destination = Path(path)
    lock = make_environment_lock(fingerprint)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(canonical_json_bytes(lock) + b"\n")
    return lock


def _load_lock(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise EnvironmentLockError(f"environment lock does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EnvironmentLockError(
            f"environment lock is not valid JSON: {path}"
        ) from exc
    errors = sorted(Draft202012Validator(_LOCK_SCHEMA).iter_errors(value), key=str)
    if errors:
        details = "; ".join(error.message for error in errors)
        raise EnvironmentLockError(f"invalid environment lock payload: {details}")
    return value


def verify_environment_lock(
    path: Path | str = DEFAULT_LOCK_PATH, *, strict: bool
) -> dict[str, object]:
    lock = _load_lock(Path(path))
    stored_fingerprint = lock["fingerprint"]
    recomputed_hash = sha256_bytes(canonical_json_bytes(stored_fingerprint))
    if lock["fingerprint_sha256"] != recomputed_hash:
        raise EnvironmentLockError(
            "environment lock hash mismatch: stored fingerprint_sha256 does not "
            "match the canonical fingerprint payload"
        )

    if strict:
        current_fingerprint = collect_environment_fingerprint()
        if current_fingerprint != stored_fingerprint:
            fields = sorted(
                key
                for key in set(current_fingerprint) | set(stored_fingerprint)
                if current_fingerprint.get(key) != stored_fingerprint.get(key)
            )
            raise EnvironmentLockError(
                "current environment mismatch in: " + ", ".join(fields)
            )

    return {
        "fingerprint_sha256": recomputed_hash,
        "strict": strict,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify the canonical GSDiff-SPI environment lock."
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=DEFAULT_LOCK_PATH,
        help="environment-lock.json path",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="also require the current runtime to match the locked fingerprint",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="replace the target with a canonical lock for the current runtime",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.write:
            lock = write_environment_lock(args.path)
            print(
                "environment_lock_written="
                f"{args.path} fingerprint_sha256={lock['fingerprint_sha256']}"
            )
        else:
            summary = verify_environment_lock(args.path, strict=args.strict)
            print(
                "environment_lock_verification=passed "
                f"strict={str(args.strict).lower()} "
                f"fingerprint_sha256={summary['fingerprint_sha256']}"
            )
    except EnvironmentLockError as exc:
        print(f"environment_lock_verification=failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
