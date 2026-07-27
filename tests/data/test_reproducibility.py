import numpy as np

from gsdiff.data.simulation import generate_spi_data


TRAIN_ARRAYS = (
    "canonical",
    "gt_frames",
    "patterns",
    "measurements",
    "frame_idx",
    "t_grid",
)
HOLDOUT_ARRAYS = (
    "eval_patterns",
    "eval_measurements",
    "eval_frame_idx",
)
ALL_ARRAYS = TRAIN_ARRAYS + HOLDOUT_ARRAYS

EXPECTED_ARRAY_CONTRACTS = {
    "canonical": ((32, 40), np.dtype(np.float32)),
    "gt_frames": ((4, 32, 40), np.dtype(np.float32)),
    "patterns": ((64, 32, 40), np.dtype(np.float32)),
    "measurements": ((64,), np.dtype(np.float32)),
    "frame_idx": ((64,), np.dtype(np.int64)),
    "t_grid": ((4,), np.dtype(np.float32)),
    "eval_patterns": ((16, 32, 40), np.dtype(np.float32)),
    "eval_measurements": ((16,), np.dtype(np.float32)),
    "eval_frame_idx": ((16,), np.dtype(np.int64)),
}


def _generate(seed, holdout_extra):
    return generate_spi_data(
        H=32,
        W=40,
        T=4,
        K=64,
        pattern_type="bernoulli",
        snr_db=25.0,
        holdout_extra=holdout_extra,
        seed=seed,
    )


def _training_noise(data):
    clean = np.array(
        [
            np.sum(data.patterns[k] * data.gt_frames[data.frame_idx[k]])
            for k in range(data.K)
        ],
        dtype=np.float64,
    )
    return data.measurements.astype(np.float64) - clean


def test_seed_7_repeats_every_public_array_byte_exactly():
    first = _generate(seed=7, holdout_extra=16)
    second = _generate(seed=7, holdout_extra=16)

    for name in ALL_ARRAYS:
        assert np.array_equal(getattr(first, name), getattr(second, name)), name
        expected_shape, expected_dtype = EXPECTED_ARRAY_CONTRACTS[name]
        assert getattr(first, name).shape == expected_shape
        assert getattr(first, name).dtype == expected_dtype


def test_adding_holdout_keeps_every_training_array_byte_exact():
    training_only = _generate(seed=7, holdout_extra=0)
    with_holdout = _generate(seed=7, holdout_extra=16)

    for name in TRAIN_ARRAYS:
        assert np.array_equal(
            getattr(training_only, name), getattr(with_holdout, name)
        ), name
    for name in HOLDOUT_ARRAYS:
        assert getattr(training_only, name) is None
        assert getattr(with_holdout, name) is not None


def test_seed_11_changes_stochastic_arrays_but_not_deterministic_targets():
    seed_7 = _generate(seed=7, holdout_extra=16)
    seed_11 = _generate(seed=11, holdout_extra=16)

    for name in ("canonical", "gt_frames", "frame_idx", "t_grid"):
        assert np.array_equal(getattr(seed_7, name), getattr(seed_11, name)), name

    for name in ("patterns", "measurements", *HOLDOUT_ARRAYS[:2]):
        assert not np.array_equal(getattr(seed_7, name), getattr(seed_11, name)), name
    assert not np.array_equal(_training_noise(seed_7), _training_noise(seed_11))

    for data in (seed_7, seed_11):
        assert (data.H, data.W, data.T, data.K) == (32, 40, 4, 64)
        for name, (expected_shape, expected_dtype) in EXPECTED_ARRAY_CONTRACTS.items():
            array = getattr(data, name)
            assert array.shape == expected_shape
            assert array.dtype == expected_dtype
