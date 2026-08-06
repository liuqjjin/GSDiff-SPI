from __future__ import annotations

import torch
import pytest

from gsdiff.prior import training_v2 as v2


@pytest.mark.cuda
def test_full_size_one_batch_preflight_runs_only_on_cuda_zero(tmp_path) -> None:
    assert torch.cuda.is_available(), "the required real-CUDA gate cannot be skipped"
    result = v2.run_preflight(
        save_directory=tmp_path,
        device=torch.device("cuda:0"),
    )
    assert result == {
        "batch_shape": [8, 1, 20, 64, 64],
        "device": "cuda:0",
        "durable_writes": 0,
        "optimizer_steps": 1,
    }
    assert list(tmp_path.iterdir()) == []
