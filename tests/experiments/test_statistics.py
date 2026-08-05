from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass
import importlib
import math
from types import ModuleType

import pytest

from gsdiff.experiments.identity import canonical_json_bytes


METRIC_NAMES = (
    "psnr_global_affine",
    "ssim_global_affine",
    "nrmse_global_affine_l2",
    "psnr_legacy_per_frame_minmax",
)
CONTRACT_SHA = "a" * 64


def _statistics() -> ModuleType:
    return importlib.import_module("gsdiff.experiments.statistics")


def _metrics(
    psnr: object = 1.0,
    *,
    ssim: object = 0.5,
    nrmse: object = 0.25,
    legacy: object = 2.0,
) -> dict[str, object]:
    return {
        "psnr_global_affine": psnr,
        "ssim_global_affine": ssim,
        "nrmse_global_affine_l2": nrmse,
        "psnr_legacy_per_frame_minmax": legacy,
    }


def _record(
    *,
    method_id: str = "method-a",
    method_config_id: str = "default",
    target_id: str = "target-a",
    motion_id: str = "motion-a",
    acquisition_config_id: str = "base",
    seed: object = 1,
    dataset_identity_sha256: str | None = None,
    metrics: dict[str, object] | None = None,
) -> dict[str, object]:
    dataset_hash = dataset_identity_sha256 or (
        f"{int(seed):064x}" if type(seed) is int else "b" * 64
    )
    return {
        "phase_id": "phase-v1",
        "scientific_contract_sha256": CONTRACT_SHA,
        "acquisition_config_id": acquisition_config_id,
        "method_config_id": method_config_id,
        "method_id": method_id,
        "target_id": target_id,
        "motion_id": motion_id,
        "seed": seed,
        "dataset_identity_sha256": dataset_hash,
        "metrics": _metrics() if metrics is None else metrics,
    }


@dataclass(frozen=True)
class _AttributeRecord:
    phase_id: str
    scientific_contract_sha256: str
    acquisition_config_id: str
    method_config_id: str
    method_id: str
    target_id: str
    motion_id: str
    seed: int
    dataset_identity_sha256: str
    metrics: dict[str, object]


def _as_attribute_record(record: dict[str, object]) -> _AttributeRecord:
    return _AttributeRecord(**record)  # type: ignore[arg-type]


def _summary(
    document: dict[str, object],
    *,
    metric: str,
    method_id: str = "method-a",
    target_id: str = "target-a",
) -> dict[str, object]:
    matches = [
        item
        for item in document["summaries"]  # type: ignore[union-attr]
        if item["metric"] == metric
        and item["method_id"] == method_id
        and item["target_id"] == target_id
    ]
    assert len(matches) == 1
    return matches[0]


def _effect(
    document: dict[str, object], comparison_id: str
) -> dict[str, object]:
    matches = [
        item
        for item in document["paired_effects"]  # type: ignore[union-attr]
        if item["comparison_id"] == comparison_id
    ]
    assert len(matches) == 1
    return matches[0]


def _comparison(
    comparison_id: str = "comparison-a",
    *,
    metric: str = "psnr_global_affine",
):
    return _statistics().PairedComparison(
        comparison_id=comparison_id,
        method_id="method-a",
        comparator_id="method-b",
        metric=metric,
    )


def _paired_records(
    *,
    seeds: tuple[int, ...] = (1, 2),
    target_id: str = "target-a",
    method_values: tuple[float, ...] = (3.0, 5.0),
    comparator_values: tuple[float, ...] = (1.0, 2.0),
    metric: str = "psnr_global_affine",
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for seed, method_value, comparator_value in zip(
        seeds, method_values, comparator_values, strict=True
    ):
        dataset_hash = f"{seed:064x}"
        method_metrics = _metrics()
        comparator_metrics = _metrics()
        method_metrics[metric] = method_value
        comparator_metrics[metric] = comparator_value
        records.extend(
            [
                _record(
                    method_id="method-a",
                    target_id=target_id,
                    seed=seed,
                    dataset_identity_sha256=dataset_hash,
                    metrics=method_metrics,
                ),
                _record(
                    method_id="method-b",
                    target_id=target_id,
                    seed=seed,
                    dataset_identity_sha256=dataset_hash,
                    metrics=comparator_metrics,
                ),
            ]
        )
    return records


def test_public_api_is_frozen_and_computes_known_mean_and_ddof1():
    statistics = _statistics()
    comparison = statistics.PairedComparison(
        "comparison-a",
        "method-a",
        "method-b",
        "psnr_global_affine",
    )
    with pytest.raises(FrozenInstanceError):
        comparison.metric = "ssim_global_affine"

    records = [
        _as_attribute_record(_record(seed=3, metrics=_metrics(3.0))),
        _record(seed=1, metrics=_metrics(1.0)),
        _record(seed=2, metrics=_metrics(2.0)),
    ]
    document = statistics.aggregate_seed_metrics(
        records,
        required_seeds=[3, 1, 2],
    )

    assert document["schema_version"] == "experiment-statistics-v1"
    assert document["required_seeds"] == [1, 2, 3]
    assert document["n_bootstrap"] == 10_000
    assert document["bootstrap_seed"] == 20260727
    assert len(document["summaries"]) == 4
    assert document["paired_effects"] == []
    summary = _summary(document, metric="psnr_global_affine")
    assert summary["per_seed"] == [
        {"seed": 1, "value": 1.0},
        {"seed": 2, "value": 2.0},
        {"seed": 3, "value": 3.0},
    ]
    assert summary["n"] == 3
    assert summary["mean"] == 2.0
    assert summary["sample_sd"] == 1.0
    canonical_json_bytes(document)


def test_real_complete_metric_records_are_supported() -> None:
    from gsdiff.experiments.aggregation import (
        CompleteMetricRecord,
        LogicalRunKey,
    )

    records = []
    for seed, psnr in ((1, 1.0), (2, 3.0)):
        records.append(
            CompleteMetricRecord(
                key=LogicalRunKey(
                    phase_id="phase-v1",
                    acquisition_config_id="base",
                    method_config_id="default",
                    method_id="method-a",
                    target_id="target-a",
                    motion_id="motion-a",
                    seed=seed,
                ),
                scientific_contract_id="contract-v1",
                scientific_contract_sha256=CONTRACT_SHA,
                method_config_sha256="a" * 64,
                checkpoints_sha256={},
                dataset_identity_sha256=f"{seed:064x}",
                run_identity_sha256="b" * 64,
                manifest_sha256="c" * 64,
                metrics_sha256="d" * 64,
                metric_version="metrics-v1",
                code_commit="e" * 40,
                dependencies_sha256="f" * 64,
                environment_lock_sha256="0" * 64,
                source_snapshot_sha256="1" * 64,
                source_projection_sha256="2" * 64,
                requested_runtime_device="cuda:0",
                execution_profile="publication-v1",
                metrics=_metrics(psnr),
            )
        )

    document = _statistics().aggregate_seed_metrics(
        records,
        required_seeds=[1, 2],
    )

    assert document["phase_id"] == "phase-v1"
    assert _summary(document, metric="psnr_global_affine")["mean"] == 2.0


def test_empty_or_mixed_phase_records_are_rejected() -> None:
    statistics = _statistics()

    with pytest.raises(ValueError, match="records must not be empty"):
        statistics.aggregate_seed_metrics([], required_seeds=[1])

    left = _record(method_id="method-a", seed=1)
    right = _record(method_id="method-b", seed=1)
    right["phase_id"] = "other-phase-v1"
    with pytest.raises(ValueError, match="one phase"):
        statistics.aggregate_seed_metrics(
            [left, right],
            required_seeds=[1],
        )


def test_one_seed_emits_null_sample_sd_and_bootstrap_ci():
    statistics = _statistics()
    records = _paired_records(
        seeds=(7,),
        method_values=(2.0,),
        comparator_values=(1.0,),
    )

    document = statistics.aggregate_seed_metrics(
        records,
        required_seeds=[7],
        comparisons=[_comparison()],
        n_bootstrap=8,
    )

    assert _summary(document, metric="psnr_global_affine")["sample_sd"] is None
    effect = _effect(document, "comparison-a")
    assert effect["per_seed"] == [
        {"seed": 7, "paired_cells": 1, "mean_effect": 1.0}
    ]
    assert effect["n"] == 1
    assert effect["mean"] == 1.0
    assert effect["sample_sd"] is None
    assert effect["bootstrap_ci"] is None


@pytest.mark.parametrize(
    ("records", "required_seeds", "match"),
    [
        ([_record(seed=1)], [1, 2], "missing|coverage"),
        ([_record(seed=1), _record(seed=2)], [1], "extra|coverage"),
        ([_record(seed=1), _record(seed=1)], [1], "duplicate"),
        ([_record(seed=1)], [1, 1], "required_seeds|duplicate"),
    ],
)
def test_required_seed_coverage_is_exact(
    records: list[dict[str, object]],
    required_seeds: list[int],
    match: str,
):
    with pytest.raises(ValueError, match=match):
        _statistics().aggregate_seed_metrics(
            records,
            required_seeds=required_seeds,
        )


@pytest.mark.parametrize(
    ("case", "kwargs"),
    [
        ("required-bool", {"required_seeds": [True]}),
        ("required-float", {"required_seeds": [1.0]}),
        ("bootstrap-count-bool", {"required_seeds": [1], "n_bootstrap": True}),
        ("bootstrap-count-zero", {"required_seeds": [1], "n_bootstrap": 0}),
        ("bootstrap-seed-bool", {"required_seeds": [1], "bootstrap_seed": False}),
        ("bootstrap-seed-negative", {"required_seeds": [1], "bootstrap_seed": -1}),
    ],
)
def test_integer_control_inputs_reject_bool_and_invalid_bounds(
    case: str, kwargs: dict[str, object]
):
    del case
    with pytest.raises((TypeError, ValueError)):
        _statistics().aggregate_seed_metrics([_record(seed=1)], **kwargs)


@pytest.mark.parametrize(
    "records",
    [
        [_record(seed=True)],
        [_record(seed=1, metrics=_metrics(True))],
        [_record(seed=1, metrics=_metrics(float("nan")))],
        [_record(seed=1, metrics=_metrics(float("inf")))],
        [
            _record(seed=1, metrics=_metrics(1e308)),
            _record(seed=2, metrics=_metrics(1e308)),
        ],
    ],
)
def test_records_reject_bool_nonfinite_and_summary_overflow(
    records: list[dict[str, object]],
):
    required = [1, 2] if len(records) == 2 else [1]
    with pytest.raises((TypeError, ValueError, OverflowError)):
        _statistics().aggregate_seed_metrics(
            records,
            required_seeds=required,
        )


@pytest.mark.parametrize("mutation", ["dataset-mismatch", "missing", "duplicate"])
def test_pairing_rejects_dataset_mismatch_missing_and_duplicate_side(
    mutation: str,
):
    records = _paired_records(
        seeds=(1,),
        method_values=(2.0,),
        comparator_values=(1.0,),
    )
    if mutation == "dataset-mismatch":
        records[1]["dataset_identity_sha256"] = "f" * 64
    elif mutation == "missing":
        records.pop()
    else:
        records.append(dict(records[0]))

    with pytest.raises(ValueError, match="dataset|missing|duplicate|pair"):
        _statistics().aggregate_seed_metrics(
            records,
            required_seeds=[1],
            comparisons=[_comparison()],
            n_bootstrap=8,
        )


def test_pairing_rejects_effect_overflow():
    records = _paired_records(
        seeds=(1,),
        method_values=(1e308,),
        comparator_values=(-1e308,),
    )

    with pytest.raises(ValueError, match="finite|overflow|effect"):
        _statistics().aggregate_seed_metrics(
            records,
            required_seeds=[1],
            comparisons=[_comparison()],
            n_bootstrap=8,
        )


def test_nrmse_effect_uses_method_minus_comparator_and_declares_direction():
    records = _paired_records(
        seeds=(1, 2),
        method_values=(1.0, 2.0),
        comparator_values=(4.0, 6.0),
        metric="nrmse_global_affine_l2",
    )

    document = _statistics().aggregate_seed_metrics(
        records,
        required_seeds=[1, 2],
        comparisons=[
            _comparison(metric="nrmse_global_affine_l2")
        ],
        n_bootstrap=8,
        bootstrap_seed=4,
    )

    effect = _effect(document, "comparison-a")
    assert effect["effect_direction"] == "method_minus_comparator"
    assert effect["metric_direction"] == "lower_is_better"
    assert effect["per_seed"] == [
        {"seed": 1, "paired_cells": 1, "mean_effect": -3.0},
        {"seed": 2, "paired_cells": 1, "mean_effect": -4.0},
    ]
    assert effect["mean"] == -3.5


def test_bootstrap_resamples_seed_clusters_and_uses_type7_percentiles():
    records: list[dict[str, object]] = []
    for target_id in ("target-a", "target-b"):
        records.extend(
            _paired_records(
                seeds=(1, 2),
                target_id=target_id,
                method_values=(0.0, 10.0),
                comparator_values=(0.0, 0.0),
            )
        )

    document = _statistics().aggregate_seed_metrics(
        records,
        required_seeds=[1, 2],
        comparisons=[_comparison("cluster")],
        n_bootstrap=16,
        bootstrap_seed=40,
    )

    effect = _effect(document, "cluster")
    assert effect["per_seed"] == [
        {"seed": 1, "paired_cells": 2, "mean_effect": 0.0},
        {"seed": 2, "paired_cells": 2, "mean_effect": 10.0},
    ]
    assert effect["mean"] == 5.0
    assert effect["sample_sd"] == pytest.approx(math.sqrt(50.0))
    assert effect["bootstrap_ci"] == [0.0, 8.125]
    assert effect["bootstrap_ci"] != [0.9375, 7.5]


def test_input_order_and_unrelated_statistics_do_not_change_bytes_or_rng():
    statistics = _statistics()
    records = _paired_records()
    comparison_z = _comparison("z-statistic")
    comparison_a = _comparison("a-statistic", metric="ssim_global_affine")

    baseline = statistics.aggregate_seed_metrics(
        records,
        required_seeds=[2, 1],
        comparisons=[comparison_z],
        n_bootstrap=32,
        bootstrap_seed=7,
    )
    repeated = statistics.aggregate_seed_metrics(
        list(reversed(records)),
        required_seeds=[1, 2],
        comparisons=[comparison_a, comparison_z],
        n_bootstrap=32,
        bootstrap_seed=7,
    )
    reordered = statistics.aggregate_seed_metrics(
        records,
        required_seeds=[2, 1],
        comparisons=[comparison_z, comparison_a],
        n_bootstrap=32,
        bootstrap_seed=7,
    )

    assert _effect(baseline, "z-statistic") == _effect(
        repeated, "z-statistic"
    )
    assert canonical_json_bytes(repeated) == canonical_json_bytes(reordered)


def test_type7_known_vector_and_negative_zero_normalization():
    statistics = _statistics()
    assert statistics._type7_percentile([0.0, 10.0, 20.0, 30.0], 0.25) == 7.5
    assert statistics._type7_percentile([0.0, 10.0, 20.0, 30.0], 0.975) == 29.25

    document = statistics.aggregate_seed_metrics(
        [_record(seed=1, metrics=_metrics(-0.0, ssim=-0.0, nrmse=-0.0, legacy=-0.0))],
        required_seeds=[1],
    )
    assert b"-0.0" not in canonical_json_bytes(document)


def test_statistics_never_emit_p_values_or_significance_claims():
    document = _statistics().aggregate_seed_metrics(
        _paired_records(),
        required_seeds=[1, 2],
        comparisons=[_comparison()],
        n_bootstrap=8,
    )

    def visit(value: object) -> None:
        if isinstance(value, dict):
            forbidden = {"p", "p_value", "p-value", "pvalue", "significance"}
            assert forbidden.isdisjoint(key.lower() for key in value)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(document)
