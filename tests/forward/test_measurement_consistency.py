import numpy as np
import torch

from gsdiff.data.simulation import _measure_interpolated_np
from gsdiff.forward.spi import SPIForwardModel


def _video():
    values = np.array(
        [
            [[0.2, -0.7, 1.1], [1.9, 0.4, -1.3]],
            [[-0.6, 1.4, 0.8], [0.3, -1.7, 2.2]],
            [[1.5, 0.1, -0.9], [-0.2, 2.4, 0.7]],
        ],
        dtype=np.float64,
    )
    return values, torch.from_numpy(values[:, None])


def _patterns(K):
    flat = np.arange(K * 6, dtype=np.float64).reshape(K, 2, 3)
    return ((flat % 7) - 2.5) / 4.0


def test_measure_matches_explicit_numpy_inner_products_with_unused_frame():
    video_np, video = _video()
    patterns_np = _patterns(5)
    frame_idx_np = np.array([0, 2, 0, 2, 2], dtype=np.int64)

    actual = SPIForwardModel.measure(
        video, torch.from_numpy(patterns_np), torch.from_numpy(frame_idx_np)
    )
    expected = np.array(
        [
            np.sum(patterns_np[k].reshape(-1) * video_np[f].reshape(-1))
            for k, f in enumerate(frame_idx_np)
        ],
        dtype=np.float64,
    )

    assert 1 not in frame_idx_np
    assert actual.shape == (5,)
    assert actual.dtype == torch.float64
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-10, atol=1e-12)


def test_measure_interpolated_matches_numpy_reference_and_final_boundary():
    video_np, video = _video()
    patterns_np = _patterns(7)

    actual = SPIForwardModel.measure_interpolated(
        video, torch.from_numpy(patterns_np)
    )
    expected = _measure_interpolated_np(patterns_np, video_np, K=7, T=3)
    expected_final = np.sum(patterns_np[-1] * video_np[-1])

    assert actual.shape == (7,)
    assert actual.dtype == torch.float64
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(actual[-1].item(), expected_final, rtol=1e-10, atol=1e-12)


def test_measure_interpolated_k1_uses_first_frame():
    video_np, video = _video()
    patterns_np = _patterns(1)

    actual = SPIForwardModel.measure_interpolated(
        video, torch.from_numpy(patterns_np)
    )
    expected = _measure_interpolated_np(patterns_np, video_np, K=1, T=3)
    expected_first = np.sum(patterns_np[0] * video_np[0])

    assert actual.shape == (1,)
    assert actual.dtype == torch.float64
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(actual[0].item(), expected_first, rtol=1e-10, atol=1e-12)
