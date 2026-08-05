"""Deterministic finite summaries and seed-cluster paired statistics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, is_dataclass
import hashlib
import math
from numbers import Real
import re
from typing import Any

import numpy as np

from .identity import canonical_json_bytes


_METRICS = (
    "psnr_global_affine",
    "ssim_global_affine",
    "nrmse_global_affine_l2",
    "psnr_legacy_per_frame_minmax",
)
_METRIC_SET = frozenset(_METRICS)
_SHA256 = re.compile(r"^[0-9a-f]{64}$", re.ASCII)
_RECORD_FIELDS = (
    "phase_id",
    "scientific_contract_sha256",
    "acquisition_config_id",
    "method_config_id",
    "method_id",
    "target_id",
    "motion_id",
    "seed",
    "dataset_identity_sha256",
    "metrics",
)
_LOGICAL_KEY_FIELDS = frozenset(
    {
        "phase_id",
        "acquisition_config_id",
        "method_config_id",
        "method_id",
        "target_id",
        "motion_id",
        "seed",
    }
)


@dataclass(frozen=True)
class PairedComparison:
    comparison_id: str
    method_id: str
    comparator_id: str
    metric: str
    method_config_id: str = "default"
    comparator_config_id: str = "default"


def aggregate_seed_metrics(
    records: Sequence[Mapping[str, object] | Any],
    *,
    required_seeds: Sequence[int],
    comparisons: Sequence[PairedComparison] = (),
    n_bootstrap: int = 10_000,
    bootstrap_seed: int = 20260727,
) -> dict[str, object]:
    """Aggregate complete metric records under the frozen statistics contract."""
    seeds = _required_seed_tuple(required_seeds)
    _require_exact_int("n_bootstrap", n_bootstrap, minimum=1)
    _require_exact_int("bootstrap_seed", bootstrap_seed, minimum=0)
    normalized_comparisons = _validated_comparisons(comparisons)
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise TypeError("records must be a sequence")
    normalized_records = tuple(_record_view(record) for record in records)
    if not normalized_records:
        raise ValueError("records must not be empty")
    phase_ids = {str(record["phase_id"]) for record in normalized_records}
    if len(phase_ids) != 1:
        raise ValueError("statistics records must belong to one phase")
    phase_id = next(iter(phase_ids))

    summaries = _build_summaries(normalized_records, seeds)
    paired_effects = _build_paired_effects(
        normalized_records,
        seeds,
        normalized_comparisons,
        n_bootstrap=n_bootstrap,
        bootstrap_seed=bootstrap_seed,
    )
    document: dict[str, object] = {
        "schema_version": "experiment-statistics-v1",
        "phase_id": phase_id,
        "required_seeds": list(seeds),
        "n_bootstrap": n_bootstrap,
        "bootstrap_seed": bootstrap_seed,
        "summaries": summaries,
        "paired_effects": paired_effects,
    }
    canonical_json_bytes(document)
    return document


def _required_seed_tuple(required_seeds: Sequence[int]) -> tuple[int, ...]:
    if isinstance(required_seeds, (str, bytes)) or not isinstance(
        required_seeds, Sequence
    ):
        raise TypeError("required_seeds must be a sequence")
    values: list[int] = []
    for seed in required_seeds:
        _require_exact_int("required_seeds entry", seed)
        values.append(seed)
    if not values:
        raise ValueError("required_seeds must not be empty")
    if len(values) != len(set(values)):
        raise ValueError("required_seeds contains a duplicate")
    return tuple(sorted(values))


def _validated_comparisons(
    comparisons: Sequence[PairedComparison],
) -> tuple[PairedComparison, ...]:
    if isinstance(comparisons, (str, bytes)) or not isinstance(
        comparisons, Sequence
    ):
        raise TypeError("comparisons must be a sequence")
    normalized: list[PairedComparison] = []
    seen_ids: set[str] = set()
    for comparison in comparisons:
        if type(comparison) is not PairedComparison:
            raise TypeError("comparison must be an exact PairedComparison")
        for field in (
            "comparison_id",
            "method_id",
            "comparator_id",
            "method_config_id",
            "comparator_config_id",
        ):
            _require_id(f"comparison {field}", getattr(comparison, field))
        if comparison.metric not in _METRIC_SET:
            raise ValueError("comparison metric is not a supported metric")
        if comparison.comparison_id in seen_ids:
            raise ValueError("duplicate comparison_id")
        if (
            comparison.method_id == comparison.comparator_id
            and comparison.method_config_id == comparison.comparator_config_id
        ):
            raise ValueError("paired comparison sides must be distinct")
        seen_ids.add(comparison.comparison_id)
        normalized.append(comparison)
    return tuple(
        sorted(
            normalized,
            key=lambda item: (
                item.comparison_id,
                item.method_id,
                item.method_config_id,
                item.comparator_id,
                item.comparator_config_id,
                item.metric,
            ),
        )
    )


def _record_view(record: Mapping[str, object] | Any) -> dict[str, object]:
    raw = {field: _record_field(record, field) for field in _RECORD_FIELDS}
    for field in (
        "phase_id",
        "acquisition_config_id",
        "method_config_id",
        "method_id",
        "target_id",
        "motion_id",
    ):
        raw[field] = _require_id(field, raw[field])
    raw["scientific_contract_sha256"] = _require_sha256(
        "scientific_contract_sha256",
        raw["scientific_contract_sha256"],
    )
    raw["dataset_identity_sha256"] = _require_sha256(
        "dataset_identity_sha256",
        raw["dataset_identity_sha256"],
    )
    _require_exact_int("record seed", raw["seed"])
    metrics = raw["metrics"]
    if not isinstance(metrics, Mapping):
        raise TypeError("record metrics must be a mapping")
    normalized_metrics: dict[str, float] = {}
    for metric in _METRICS:
        if metric not in metrics:
            raise ValueError(f"record metrics is missing {metric!r}")
        normalized_metrics[metric] = _finite_number(
            f"record metric {metric}", metrics[metric]
        )
    raw["metrics"] = normalized_metrics
    return raw


def _record_field(record: object, name: str) -> object:
    if isinstance(record, Mapping):
        direct = record.get(name) if name in record else None
        key = record.get("key")
        has_nested = (
            name in _LOGICAL_KEY_FIELDS
            and isinstance(key, Mapping)
            and name in key
        )
        if name in record and has_nested and direct != key[name]:
            raise ValueError(f"record has conflicting {name!r} values")
        if name in record:
            return direct
        if has_nested:
            return key[name]
        raise ValueError(f"record is missing {name!r}")
    if is_dataclass(record) and not isinstance(record, type):
        if hasattr(record, name):
            return getattr(record, name)
        key = getattr(record, "key", None)
        if name in _LOGICAL_KEY_FIELDS and is_dataclass(key) and hasattr(key, name):
            return getattr(key, name)
        raise ValueError(f"record is missing {name!r}")
    raise TypeError("each record must be a mapping or dataclass instance")


def _build_summaries(
    records: Sequence[Mapping[str, object]],
    required_seeds: tuple[int, ...],
) -> list[dict[str, object]]:
    groups: dict[tuple[str, ...], dict[int, Mapping[str, float]]] = {}
    for record in records:
        key = _summary_key(record)
        seed = record["seed"]
        assert type(seed) is int
        seed_values = groups.setdefault(key, {})
        if seed in seed_values:
            raise ValueError("duplicate record for summary group and seed")
        metrics = record["metrics"]
        assert isinstance(metrics, Mapping)
        seed_values[seed] = metrics  # type: ignore[assignment]

    required_set = set(required_seeds)
    summaries: list[dict[str, object]] = []
    for key in sorted(groups):
        seed_values = groups[key]
        observed = set(seed_values)
        if observed != required_set:
            missing = sorted(required_set - observed)
            extra = sorted(observed - required_set)
            raise ValueError(
                "summary seed coverage mismatch: "
                f"missing={missing}, extra={extra}"
            )
        for metric in sorted(_METRICS):
            values = [seed_values[seed][metric] for seed in required_seeds]
            mean = _finite_mean(values, noun="summary mean")
            summaries.append(
                {
                    "scientific_contract_sha256": key[0],
                    "acquisition_config_id": key[1],
                    "target_id": key[2],
                    "motion_id": key[3],
                    "method_id": key[4],
                    "method_config_id": key[5],
                    "metric": metric,
                    "per_seed": [
                        {"seed": seed, "value": seed_values[seed][metric]}
                        for seed in required_seeds
                    ],
                    "n": len(values),
                    "mean": mean,
                    "sample_sd": _sample_sd(values, mean),
                }
            )
    return summaries


def _summary_key(record: Mapping[str, object]) -> tuple[str, ...]:
    return (
        str(record["scientific_contract_sha256"]),
        str(record["acquisition_config_id"]),
        str(record["target_id"]),
        str(record["motion_id"]),
        str(record["method_id"]),
        str(record["method_config_id"]),
    )


def _build_paired_effects(
    records: Sequence[Mapping[str, object]],
    required_seeds: tuple[int, ...],
    comparisons: Sequence[PairedComparison],
    *,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> list[dict[str, object]]:
    effects: list[dict[str, object]] = []
    for comparison in comparisons:
        selected = [
            record
            for record in records
            if _comparison_side(record, comparison) is not None
        ]
        if not selected:
            raise ValueError(
                f"paired comparison {comparison.comparison_id!r} has no records"
            )
        contracts = sorted(
            {str(record["scientific_contract_sha256"]) for record in selected}
        )
        for contract in contracts:
            effects.append(
                _paired_effect_for_contract(
                    [
                        record
                        for record in selected
                        if record["scientific_contract_sha256"] == contract
                    ],
                    required_seeds,
                    comparison,
                    contract,
                    n_bootstrap=n_bootstrap,
                    bootstrap_seed=bootstrap_seed,
                )
            )
    return sorted(
        effects,
        key=lambda item: (
            item["comparison_id"],
            item["scientific_contract_sha256"],
            item["method_id"],
            item["method_config_id"],
            item["comparator_id"],
            item["comparator_config_id"],
            item["metric"],
        ),
    )


def _paired_effect_for_contract(
    records: Sequence[Mapping[str, object]],
    required_seeds: tuple[int, ...],
    comparison: PairedComparison,
    contract: str,
    *,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    cells: dict[tuple[str, str, str, int], dict[str, Mapping[str, object]]] = {}
    for record in records:
        side = _comparison_side(record, comparison)
        assert side is not None
        seed = record["seed"]
        assert type(seed) is int
        key = (
            str(record["acquisition_config_id"]),
            str(record["target_id"]),
            str(record["motion_id"]),
            seed,
        )
        sides = cells.setdefault(key, {})
        if side in sides:
            raise ValueError("duplicate paired side at exact pairing grain")
        sides[side] = record

    effects_by_seed: dict[int, list[tuple[tuple[str, str, str], float]]] = {
        seed: [] for seed in required_seeds
    }
    for cell_key in sorted(cells):
        sides = cells[cell_key]
        if set(sides) != {"method", "comparator"}:
            raise ValueError("missing paired comparison side")
        method = sides["method"]
        comparator = sides["comparator"]
        if method["dataset_identity_sha256"] != comparator[
            "dataset_identity_sha256"
        ]:
            raise ValueError("paired methods use different dataset identities")
        seed = cell_key[3]
        if seed not in effects_by_seed:
            raise ValueError("paired comparison contains an extra seed")
        method_metrics = method["metrics"]
        comparator_metrics = comparator["metrics"]
        assert isinstance(method_metrics, Mapping)
        assert isinstance(comparator_metrics, Mapping)
        method_value = method_metrics[comparison.metric]
        comparator_value = comparator_metrics[comparison.metric]
        effect = method_value - comparator_value
        effect = _finite_result("paired effect", effect)
        effects_by_seed[seed].append((cell_key[:3], effect))

    expected_cells: set[tuple[str, str, str]] | None = None
    per_seed: list[dict[str, object]] = []
    seed_means: list[float] = []
    for seed in required_seeds:
        rows = sorted(effects_by_seed[seed])
        observed_cells = {cell for cell, _effect in rows}
        if not rows:
            raise ValueError("missing paired cells for required seed")
        if expected_cells is None:
            expected_cells = observed_cells
        elif observed_cells != expected_cells:
            raise ValueError("paired cells are not balanced across seeds")
        seed_mean = _finite_mean(
            [effect for _cell, effect in rows],
            noun="paired seed mean",
        )
        seed_means.append(seed_mean)
        per_seed.append(
            {
                "seed": seed,
                "paired_cells": len(rows),
                "mean_effect": seed_mean,
            }
        )

    headline_mean = _finite_mean(seed_means, noun="paired headline mean")
    statistic_key = {
        "comparison_id": comparison.comparison_id,
        "scientific_contract_sha256": contract,
        "method_id": comparison.method_id,
        "method_config_id": comparison.method_config_id,
        "comparator_id": comparison.comparator_id,
        "comparator_config_id": comparison.comparator_config_id,
        "metric": comparison.metric,
    }
    return {
        **statistic_key,
        "effect_direction": "method_minus_comparator",
        "metric_direction": (
            "lower_is_better"
            if comparison.metric == "nrmse_global_affine_l2"
            else "higher_is_better"
        ),
        "per_seed": per_seed,
        "n": len(seed_means),
        "mean": headline_mean,
        "sample_sd": _sample_sd(seed_means, headline_mean),
        "bootstrap_ci": (
            None
            if len(seed_means) < 2
            else _bootstrap_ci(
                seed_means,
                n_bootstrap=n_bootstrap,
                bootstrap_seed=bootstrap_seed,
                statistic_key=statistic_key,
            )
        ),
    }


def _comparison_side(
    record: Mapping[str, object], comparison: PairedComparison
) -> str | None:
    identity = (record["method_id"], record["method_config_id"])
    if identity == (comparison.method_id, comparison.method_config_id):
        return "method"
    if identity == (comparison.comparator_id, comparison.comparator_config_id):
        return "comparator"
    return None


def _bootstrap_ci(
    seed_means: Sequence[float],
    *,
    n_bootstrap: int,
    bootstrap_seed: int,
    statistic_key: Mapping[str, object],
) -> list[float]:
    entropy_document = {
        "domain": "aggregate-bootstrap-v1",
        "bootstrap_seed": bootstrap_seed,
        "statistic_key": dict(statistic_key),
    }
    digest = hashlib.sha256(canonical_json_bytes(entropy_document)).digest()
    words = [
        int.from_bytes(digest[index : index + 4], "big")
        for index in range(0, len(digest), 4)
    ]
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(words)))
    sample_size = len(seed_means)
    estimates: list[float] = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, sample_size, size=sample_size)
        estimate = _finite_mean(
            [seed_means[int(index)] for index in indices],
            noun="bootstrap estimate",
        )
        estimates.append(estimate)
    return [
        _type7_percentile(estimates, 0.025),
        _type7_percentile(estimates, 0.975),
    ]


def _type7_percentile(values: Sequence[float], probability: float) -> float:
    """Return an explicit Hyndman-Fan type-7 percentile."""
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("percentile values must be a sequence")
    if not values:
        raise ValueError("percentile values must not be empty")
    p = _finite_number("percentile probability", probability)
    if not 0.0 <= p <= 1.0:
        raise ValueError("percentile probability must be in [0, 1]")
    ordered = sorted(
        _finite_number("percentile value", value) for value in values
    )
    h = (len(ordered) - 1) * p
    lower = math.floor(h)
    upper = math.ceil(h)
    fraction = h - lower
    if lower == upper:
        return ordered[lower]
    interpolated = ordered[lower] + fraction * (
        ordered[upper] - ordered[lower]
    )
    return _finite_result("percentile endpoint", interpolated)


def _finite_mean(values: Sequence[float], *, noun: str) -> float:
    if not values:
        raise ValueError(f"{noun} requires at least one value")
    try:
        total = math.fsum(values)
    except OverflowError as error:
        raise ValueError(f"{noun} overflowed") from error
    return _finite_result(noun, total / len(values))


def _sample_sd(values: Sequence[float], mean: float) -> float | None:
    if len(values) < 2:
        return None
    deviations = [value - mean for value in values]
    if not all(math.isfinite(value) for value in deviations):
        raise ValueError("sample SD deviation overflowed")
    scale = max(abs(value) for value in deviations)
    if scale == 0.0:
        return 0.0
    scaled_sum = math.fsum((value / scale) ** 2 for value in deviations)
    result = scale * math.sqrt(scaled_sum / (len(values) - 1))
    return _finite_result("sample SD", result)


def _finite_number(noun: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{noun} must be a real number and not bool")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{noun} must be finite")
    return _normalize_zero(result)


def _finite_result(noun: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{noun} is not a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{noun} is not finite or overflowed")
    return _normalize_zero(result)


def _normalize_zero(value: float) -> float:
    return 0.0 if value == 0.0 else value


def _require_exact_int(noun: str, value: object, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise TypeError(f"{noun} must be an exact integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{noun} must be at least {minimum}")
    return value


def _require_id(noun: str, value: object) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{noun} must be a nonempty string")
    return value


def _require_sha256(noun: str, value: object) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{noun} must be a lowercase SHA-256")
    return value


__all__ = ["PairedComparison", "aggregate_seed_metrics"]
