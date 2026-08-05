"""Bind the fixed publication phase set into one canonical results lock."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    repository_root = str(REPO_ROOT)
    if repository_root not in sys.path:
        sys.path.insert(0, repository_root)

from gsdiff.experiments.aggregation import publish_json_atomic
from gsdiff.experiments.contracts import (
    load_phase_evidence_contract,
    load_publication_lock_contract,
    load_statistics_contract,
)
from gsdiff.experiments.path_roles import PathRole, require_disjoint_path_roles
from gsdiff.experiments.versioned_json import validate_versioned_json
from scripts.experiments.aggregate_campaign import (
    materialize_authoritative_phase,
)
from scripts.experiments.verify_campaign import (
    load_canonical_document,
    verify_physical_aggregate_document,
)


_COMMON_FIELDS = (
    "code_commit",
    "metric_version",
    "dependencies_sha256",
    "environment_lock_sha256",
    "source_snapshot_sha256",
    "source_projection_sha256",
    "requested_runtime_device",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publication-contract", type=Path, required=True)
    parser.add_argument("--phase-id", action="append", required=True)
    parser.add_argument("--aggregate", type=Path, action="append", required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument(
        "--phase-evidence-contract",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument(
        "--statistics-contract",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        arguments = _parser().parse_args(argv)
        path_roles = _lock_path_roles(arguments)
        require_disjoint_path_roles(*path_roles)
        publication_contract = load_publication_lock_contract(
            arguments.publication_contract
        )
        lengths = {
            len(arguments.phase_id),
            len(arguments.aggregate),
            len(arguments.phase_evidence_contract),
            len(arguments.statistics_contract),
        }
        if len(lengths) != 1:
            raise ValueError("each phase requires one aggregate and both contracts")
        if len(arguments.phase_id) != len(set(arguments.phase_id)):
            raise ValueError("duplicate aggregate phase")
        required_phase_ids = set(publication_contract.required_phase_counts)
        if set(arguments.phase_id) != required_phase_ids:
            raise ValueError("aggregate phases do not match publication contract")
        phases: dict[str, dict[str, object]] = {}
        input_paths: dict[str, tuple[Path, Path, Path]] = {}
        method_provenance: dict[
            tuple[str, str],
            tuple[str, tuple[tuple[str, str], ...]],
        ] = {}
        common: dict[str, str] | None = None
        for phase_id, aggregate_path, phase_evidence_path, statistics_path in zip(
            arguments.phase_id,
            arguments.aggregate,
            arguments.phase_evidence_contract,
            arguments.statistics_contract,
            strict=True,
        ):
            plan, source_protocol = materialize_authoritative_phase(phase_id)
            statistics_contract = load_statistics_contract(
                statistics_path,
                expected_plan=plan,
                source_protocol=source_protocol,
            )
            phase_evidence_contract = load_phase_evidence_contract(
                phase_evidence_path,
                expected_plan=plan,
                expected_statistics_contract_sha256=(
                    statistics_contract.canonical_sha256
                ),
            )
            document, payload = load_canonical_document(aggregate_path)
            completion = verify_physical_aggregate_document(
                document,
                artifact_root=arguments.artifact_root,
                phase_evidence_contract=phase_evidence_contract,
                statistics_contract=statistics_contract,
            )
            if completion["phase_id"] != phase_id:
                raise ValueError("verified aggregate phase disagrees with CLI phase")
            expected_count = publication_contract.required_phase_counts[phase_id]
            if completion["record_count"] != expected_count:
                raise ValueError("verified aggregate count disagrees with publication")
            records = document.get("records")
            if type(records) is not list or not records or type(records[0]) is not dict:
                raise ValueError("verified aggregate records are unavailable")
            _bind_method_provenance(records, method_provenance)
            signature = {field: str(records[0][field]) for field in _COMMON_FIELDS}
            if common is None:
                common = signature
            elif signature != common:
                raise ValueError("phase aggregates contain mixed evidence")
            phases[phase_id] = {
                "phase_id": phase_id,
                "phase_sha256": completion["phase_sha256"],
                "execution_profile": completion["execution_profile"],
                "aggregate_sha256": hashlib.sha256(payload).hexdigest(),
                "phase_evidence_contract_sha256": completion[
                    "phase_evidence_contract_sha256"
                ],
                "statistics_contract_sha256": completion[
                    "statistics_contract_sha256"
                ],
                "record_count": completion["record_count"],
            }
            input_paths[phase_id] = (
                aggregate_path,
                phase_evidence_path,
                statistics_path,
            )
        if set(phases) != required_phase_ids:
            raise ValueError("verified aggregate phases are incomplete")
        assert common is not None

        phase_order = list(publication_contract.required_phase_counts)
        regeneration_command = [
            "python",
            "scripts/experiments/lock_results.py",
            "--publication-contract",
            str(arguments.publication_contract),
            "--artifact-root",
            str(arguments.artifact_root),
        ]
        for phase_id in phase_order:
            aggregate_path, phase_evidence_path, statistics_path = input_paths[
                phase_id
            ]
            regeneration_command.extend(["--phase-id", phase_id])
            regeneration_command.extend(["--aggregate", str(aggregate_path)])
            regeneration_command.extend(
                ["--phase-evidence-contract", str(phase_evidence_path)]
            )
            regeneration_command.extend(
                ["--statistics-contract", str(statistics_path)]
            )
        regeneration_command.extend(["--output", str(arguments.output)])
        total_records = sum(
            int(phases[phase_id]["record_count"]) for phase_id in phase_order
        )
        if total_records != publication_contract.required_total_records:
            raise ValueError("verified total disagrees with publication contract")
        document = validate_versioned_json(
            {
                "schema_version": "results-lock-v1",
                "status": "complete",
                "publication_lock_contract_sha256": (
                    publication_contract.canonical_sha256
                ),
                **common,
                "phases": [phases[phase_id] for phase_id in phase_order],
                "total_records": total_records,
                "regeneration_command": regeneration_command,
            },
            "results-lock-v1",
        )
        require_disjoint_path_roles(*path_roles)
        publish_json_atomic(
            arguments.output,
            document,
            schema_version="results-lock-v1",
        )
        return 0
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"results lock refused: {type(error).__name__}", file=sys.stderr)
        return 1


def _bind_method_provenance(
    records: list[object],
    observed: dict[
        tuple[str, str],
        tuple[str, tuple[tuple[str, str], ...]],
    ],
) -> None:
    """Require one config/checkpoint provenance per method grain across phases."""
    for record in records:
        if type(record) is not dict:
            raise ValueError("verified aggregate record is unavailable")
        key = record.get("key")
        if type(key) is not dict:
            raise ValueError("verified aggregate logical key is unavailable")
        method_id = key.get("method_id")
        method_config_id = key.get("method_config_id")
        method_config_sha256 = record.get("method_config_sha256")
        checkpoints_sha256 = record.get("checkpoints_sha256")
        if (
            type(method_id) is not str
            or not method_id
            or type(method_config_id) is not str
            or not method_config_id
            or type(method_config_sha256) is not str
            or type(checkpoints_sha256) is not dict
            or any(
                type(logical_id) is not str
                or not logical_id
                or type(sha256) is not str
                for logical_id, sha256 in checkpoints_sha256.items()
            )
        ):
            raise ValueError("verified aggregate method provenance is invalid")
        grain = (method_id, method_config_id)
        provenance = (
            method_config_sha256,
            tuple(sorted(checkpoints_sha256.items())),
        )
        prior = observed.get(grain)
        if prior is None:
            observed[grain] = provenance
        elif prior != provenance:
            raise ValueError("phase aggregates contain mixed method provenance")


def _lock_path_roles(arguments: argparse.Namespace) -> tuple[PathRole, ...]:
    roles: list[PathRole] = [
        PathRole(
            "publication lock contract",
            arguments.publication_contract,
            "read",
        ),
        PathRole("artifact root", arguments.artifact_root, "read"),
        PathRole(
            "ablation protocol",
            REPO_ROOT / "configs" / "protocols" / "ablations-v1.yaml",
            "read",
        ),
        PathRole(
            "primary protocol",
            REPO_ROOT / "configs" / "protocols" / "primary-v1.yaml",
            "read",
        ),
        PathRole("schema directory", REPO_ROOT / "schemas", "read"),
        PathRole(
            "requirements lock",
            REPO_ROOT / "requirements-lock.txt",
            "read",
        ),
        PathRole(
            "environment lock",
            REPO_ROOT / "docs" / "reproducibility" / "environment-lock.json",
            "read",
        ),
        PathRole("runtime environment", Path(sys.prefix), "read"),
    ]
    for index, path in enumerate(arguments.aggregate):
        roles.append(PathRole(f"aggregate {index}", path, "read"))
    for index, path in enumerate(arguments.phase_evidence_contract):
        roles.append(PathRole(f"phase evidence contract {index}", path, "read"))
    for index, path in enumerate(arguments.statistics_contract):
        roles.append(PathRole(f"statistics contract {index}", path, "read"))
    roles.append(PathRole("results lock", arguments.output, "write"))
    return tuple(roles)


if __name__ == "__main__":
    raise SystemExit(main())
