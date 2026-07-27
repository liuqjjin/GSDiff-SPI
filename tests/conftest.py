from __future__ import annotations

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def deterministic_test_seed():
    np.random.seed(20260727)
    torch.manual_seed(20260727)


@pytest.fixture
def cpu_device():
    return torch.device("cpu")
