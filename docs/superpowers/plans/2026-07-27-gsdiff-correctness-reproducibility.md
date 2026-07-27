# GSDiff-SPI Correctness and Reproducibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Checkboxes
> define the immutable execution sequence; record progress in
> `docs/reproducibility/implementation-log.md`, not by editing this plan.

**Goal:** Correct the load-bearing numerical defects, establish deterministic tests, and expose immutable data and metric interfaces that later experiment and publication plans can trust.

**Architecture:** The existing scene, motion, forward, solver, and prior modules remain the algorithmic core. Small pure helpers make the weighted operators, schedules, geometry, metrics, data serialization, and runtime identity directly testable; `train.py` consumes these helpers while preserving explicitly labelled legacy outputs.

**Tech Stack:** Python 3.12, PyTorch 2.8, NumPy, SciPy, scikit-image, PyYAML, pytest 9, Windows PowerShell, CUDA 12.8 for marked smoke tests.

## Global Constraints

- Authoritative interpreter: `D:\conda\envs\spi\python.exe`; never invoke the Windows Store `python.exe` stub.
- Repository baseline preserved as legacy evidence: `c03420784bc92b4e9b9eef8330cbd9571ebebc68`.
- Every production behavior change must follow observed RED → minimal GREEN → refactor.
- CPU numerical tests use float64 and deterministic seeds; CUDA tests use the `cuda` marker.
- Weighted TV adjoint relative error must be below `1e-10` for `alpha ∈ {0, 0.05, 0.3, 1, 2}`.
- Primary image metrics use one affine calibration for the complete video; independent per-frame min-max PSNR is legacy-only.
- No GPU campaign may start during this plan.
- No ignored legacy result may be relabelled as a post-fix result.
- Each task is committed independently after its focused tests and the accumulated CPU suite pass.
- Every native command is fail-closed: immediately inspect its exit code with
  the `Invoke-Checked` pattern in the completion gate. A later successful
  command may never mask an earlier failure; compact snippets do not waive this
  rule.
- Before each task commit, append that task's RED/GREEN commands, outcomes, and
  prior task commit to `docs/reproducibility/implementation-log.md` and stage
  the log with the task. After Task 10, make one final log-only commit recording
  Task 10's commit, then run the completion gate.

---

## File Structure

Create:

- `pytest.ini` — test discovery and the `cuda` marker.
- `requirements-dev.txt` — test-only dependencies.
- `requirements-lock.txt` — exact tested runtime distributions.
- `tests/conftest.py` — deterministic helpers and tiny dummy models.
- `tests/test_runtime_contract.py` — runtime metadata and CPU/CUDA separation.
- `tests/test_cuda_smoke.py` — real CUDA forward/backward/proximal smoke.
- `tests/prior/test_tv3d.py` — weighted adjoint, proximal, and energy definitions.
- `tests/prior/test_diffusion_schedule.py` — exact annealing endpoints.
- `tests/solver/test_sgd_parameters.py` — frozen gradients, clipping, and LR floors.
- `tests/solver/test_gradient_groups.py` — independent scene/motion clipping in both solvers.
- `tests/scene/test_numerical_gradients.py` — finite-difference scene and motion checks.
- `tests/scene/test_rectangular_geometry.py` — non-square coordinates and FOV masks.
- `tests/forward/test_measurement_consistency.py` — direct/batched measurement agreement.
- `tests/data/test_reproducibility.py` — deterministic patterns, noise, and holdout.
- `tests/evaluation/test_metrics.py` — global-affine and legacy metrics.
- `tests/data/test_artifacts.py` — bit-exact dataset round trip.
- `tests/experiments/test_identity.py` — canonical hashing and runtime identity.
- `tests/reproducibility/test_environment_lock.py` — strict fingerprint checks.
- `tests/reproducibility/test_implementation_provenance.py` — immutable plan
  and starting-state checks.
- `tests/reproducibility/test_pytest_junit.py` — reject stale, malformed,
  empty, failed, or skipped required-test reports.
- `gsdiff/evaluation/__init__.py` — evaluation package exports.
- `gsdiff/evaluation/metrics.py` — `metrics-v1`.
- `gsdiff/solver/gradients.py` — active-parameter and group-clipping helpers.
- `gsdiff/data/artifacts.py` — separate immutable acquisition/truth
  serialization.
- `gsdiff/experiments/__init__.py` — experiment primitives.
- `gsdiff/experiments/identity.py` — canonical hashes and runtime metadata.
- `docs/reproducibility/runtime.md` — exact environment and commands.
- `docs/reproducibility/environment-lock.json` — canonical environment fingerprint.
- `docs/reproducibility/implementation-provenance.json` — exact plan/worktree
  starting state.
- `docs/reproducibility/implementation-log.md` — progress without editing the
  approved plans.
- `scripts/reproducibility/verify_environment_lock.py` — fail-closed
  environment verification.
- `scripts/reproducibility/verify_implementation_provenance.py` — plan and
  starting-state verification.
- `scripts/reproducibility/verify_pytest_junit.py` — prove marked tests
  actually executed.

Modify:

- `gsdiff/prior/tv.py`
- `gsdiff/prior/diffusion.py`
- `gsdiff/solver/sgd.py`
- `gsdiff/solver/admm.py`
- `gsdiff/baselines/inr.py`
- `gsdiff/baselines/recinr.py`
- `gsdiff/baselines/common.py`
- `gsdiff/utils.py`
- `train.py`
- `requirements.txt`
- `README.md`
- `THEORY.md`
- `CLAUDE.md`

## Task 0: Obtain Isolation Consent and Record the Starting State

No production implementation starts until this gate is complete.

- [ ] **Step 1: Freeze the approved document baseline before branching**

The root checkout first commits these exact immutable inputs:

- `docs/superpowers/specs/2026-07-27-gsdiff-correctness-publication-design.md`;
- `docs/superpowers/plans/2026-07-27-gsdiff-correctness-reproducibility.md`;
- `docs/superpowers/plans/2026-07-27-gsdiff-experiments-artifacts.md`;
- `docs/superpowers/plans/2026-07-27-gsdiff-publication-package.md`.

Record that commit as `plan_baseline_commit`. Do not edit plan checkboxes during
execution: all progress, deviations, RED/GREEN evidence, and task commits go to
`docs/reproducibility/implementation-log.md`. A plan change requires a reviewed
new plan-baseline commit and provenance update before work resumes.

- [ ] **Step 2: Obtain the user's explicit worktree preference**

Detect whether the current checkout is already a linked worktree by comparing
the resolved Git directory and common directory, with the submodule guard. If
it is a normal checkout and the user has not declared a preference, ask whether
to create an isolated worktree. Do not infer consent from approval of the
scientific specification.

- [ ] **Step 3: Create or enter the approved isolated workspace**

Use the native worktree mechanism if the environment provides one; otherwise
use a Git worktree only after verifying the project-local worktree directory is
ignored. Create it from `plan_baseline_commit` and use a `codex/` branch name.
If the user declines, record that decision and work in place.

- [ ] **Step 4: Record exact provenance**

Record:

- legacy evidence baseline `c03420784bc92b4e9b9eef8330cbd9571ebebc68`;
- approved design commit `24c1959599d9d775114d068f6de41ef2e31b5e36`;
- implementation branch, worktree path, starting HEAD, remotes, and clean
  status;
- `plan_baseline_commit` and SHA-256 of the design plus all three exact plan
  paths listed above;
- the user's worktree decision and timestamp.

Write this to `docs/reproducibility/implementation-provenance.json`, validate
it, and commit it with the initially empty implementation log before changing
production code.

Commit authorization in this plan does not imply authorization to push, merge,
delete a branch, publish a release, or permanently delete quarantine.

- [ ] **Step 5: Run the baseline**

Install no dependency yet. Run the repository's existing import/compile and any
discoverable tests with `D:\conda\envs\spi\python.exe`. If the baseline fails,
report the exact failure and ask whether to diagnose it before implementation,
as required by the worktree workflow.

## Task 1: Establish the Runtime and Test Contract

**Files:**

- Create: `pytest.ini`
- Create: `requirements-dev.txt`
- Create: `requirements-lock.txt`
- Create: `tests/conftest.py`
- Create: `tests/test_runtime_contract.py`
- Create: `tests/test_cuda_smoke.py`
- Create: `gsdiff/experiments/__init__.py`
- Create: `gsdiff/experiments/identity.py`
- Create: `docs/reproducibility/runtime.md`
- Create: `docs/reproducibility/environment-lock.json`
- Create: `scripts/reproducibility/verify_environment_lock.py`
- Create: `scripts/reproducibility/verify_implementation_provenance.py`
- Create: `scripts/reproducibility/verify_pytest_junit.py`
- Create: `tests/reproducibility/test_environment_lock.py`
- Create: `tests/reproducibility/test_implementation_provenance.py`

**Interfaces:**

- Consumes: Python and PyTorch runtime state.
- Produces:

```python
def collect_runtime_metadata() -> dict[str, object]
def collect_environment_fingerprint() -> dict[str, object]
def canonical_json_bytes(value: object) -> bytes
def sha256_bytes(payload: bytes) -> str
```

- [ ] **Step 1: Create pytest configuration and shared deterministic fixtures**

Add:

```ini
[pytest]
testpaths = tests
addopts = -ra
markers =
    cuda: requires a CUDA-capable PyTorch runtime
```

Add `requirements-dev.txt`:

```text
-r requirements.txt
pytest==9.1.0
jsonschema==4.25.1
```

Add `tests/conftest.py`:

```python
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
```

- [ ] **Step 2: Install and verify the declared development dependencies**

After creating `requirements-dev.txt`, install it into the approved
authoritative environment and verify dependency consistency before the first
pytest invocation:

```powershell
D:\conda\envs\spi\python.exe -m pip install -r requirements-dev.txt
D:\conda\envs\spi\python.exe -m pip check
```

Record the pre/post distribution fingerprints in the implementation log.
Generate the exact requirements/environment locks only after this installation
succeeds.

- [ ] **Step 3: Write the failing runtime and fingerprint tests**

```python
from pathlib import Path
import sys

from gsdiff.experiments.identity import collect_runtime_metadata


def test_runtime_metadata_has_reproducibility_fields():
    meta = collect_runtime_metadata()
    assert Path(meta["python_executable"]).resolve() == Path(sys.executable).resolve()
    assert meta["python_version"]
    assert meta["torch_version"]
    assert "cuda_version" in meta
    assert meta["os"]
```

Add RED tests for `collect_environment_fingerprint()`:

- distribution ordering/case normalization is deterministic;
- Python ABI, platform, PyTorch/CUDA build, GPU driver/device, and allowlisted
  numerical environment variables are present;
- changing one dependency version, ABI field, or numerical environment value
  changes the canonical hash;
- unknown secret-bearing environment variables are never captured.

- [ ] **Step 4: Run the focused tests and observe RED**

Run:

```powershell
D:\conda\envs\spi\python.exe -m pytest tests/test_runtime_contract.py tests/reproducibility/test_environment_lock.py -q
```

Expected: collection fails because `gsdiff.experiments.identity` does not yet
exist.

- [ ] **Step 5: Implement the runtime, fingerprint, and hashing helpers**

Create `gsdiff/experiments/identity.py`:

```python
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from typing import Any

import torch


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def collect_runtime_metadata() -> dict[str, object]:
    return {
        "python_executable": os.path.realpath(sys.executable),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "os": platform.platform(),
    }


def collect_environment_fingerprint() -> dict[str, object]:
    """Return the canonical, secret-free environment-lock payload."""
    # Sort normalized installed distribution name/version records; include
    # Python implementation/ABI, platform, PyTorch/CUDA build, GPU
    # driver/device, and only the documented numerical environment allowlist.
    ...
```

Export all four helpers from `gsdiff/experiments/__init__.py`. Implement and
test canonical lock writing plus strict comparison in
`verify_environment_lock.py`; no verifier may trust a recorded hash without
recomputing the current fingerprint.

- [ ] **Step 6: Document and freeze the exact runtime**

`docs/reproducibility/runtime.md` must record:

```text
Interpreter: D:\conda\envs\spi\python.exe
Install: D:\conda\envs\spi\python.exe -m pip install -r requirements-dev.txt
CPU tests: D:\conda\envs\spi\python.exe -m pytest -m "not cuda" -q
CUDA tests: D:\conda\envs\spi\python.exe -m pytest -m cuda -q
```

Include the output of:

```powershell
D:\conda\envs\spi\python.exe -c "import platform,torch; print(platform.python_version()); print(torch.__version__); print(torch.version.cuda); print(torch.cuda.get_device_name(0))"
```

Generate a sorted exact distribution lock from the authoritative interpreter
and a canonical `environment-lock.json` containing installed package
name/version, Python ABI, PyTorch/CUDA build, GPU driver/device, OS, and
relevant numerical environment variables. Hash the canonical record for run
identity. `requirements.txt` remains the human minimum; publication experiments
match the exact fingerprint.

The hard-coded `D:\conda\envs\spi\python.exe` check belongs only to this
workstation's documented preflight command. Portable unit tests compare
metadata with the interpreter that actually launched pytest so a clean clone or
CI environment can reproduce the contract using its own locked interpreter.
`verify_environment_lock.py --strict` recomputes the canonical fingerprint and
fails on any dependency, ABI, numerical-environment, or lock-hash mismatch.
`verify_implementation_provenance.py` recomputes the four immutable document
hashes and verifies the recorded baseline/worktree relationship.

- [ ] **Step 7: Add a real CUDA smoke test**

Mark one tiny test with `@pytest.mark.cuda`. On CUDA it builds a `4×7`,
three-frame Gaussian+SE(2) forward model, evaluates measurements, and performs
backward. Assert finite outputs and gradients on the CUDA device. The corrected
TV proximal is added to this smoke test only after Task 2 reaches GREEN. Skip
only when CUDA is genuinely unavailable; the final GPU campaign gate requires
the test report to contain at least one executed CUDA test and zero skips, not
merely exit zero with all tests skipped.

Implement `verify_pytest_junit.py` with a required report path and
`--created-after-utc` freshness boundary. Unit tests reject missing/malformed
XML, zero tests, failures/errors, excessive skips, and a syntactically valid but
stale report.

- [ ] **Step 8: Verify GREEN, strict verifiers, and commit**

Run:

```powershell
D:\conda\envs\spi\python.exe -m pytest -m "not cuda" -q
D:\conda\envs\spi\python.exe -m pytest -m cuda -q
D:\conda\envs\spi\python.exe scripts\reproducibility\verify_environment_lock.py --strict
D:\conda\envs\spi\python.exe scripts\reproducibility\verify_implementation_provenance.py --strict
git diff --check
```

Expected: the focused test passes and `git diff --check` is silent.

Commit:

```powershell
git add pytest.ini requirements-dev.txt requirements-lock.txt tests/conftest.py tests/test_runtime_contract.py tests/test_cuda_smoke.py tests/reproducibility gsdiff/experiments scripts/reproducibility docs/reproducibility/runtime.md docs/reproducibility/environment-lock.json docs/reproducibility/implementation-log.md
git commit -m "chore: establish reproducible test runtime"
```

## Task 2: Correct the Weighted 3D-TV Adjoint

**Files:**

- Create: `tests/prior/test_tv3d.py`
- Modify: `tests/test_cuda_smoke.py`
- Modify: `gsdiff/prior/tv.py`

**Interfaces:**

- Consumes: video `[T,H,W]`, dual field `[T,H,W,3]`, scalar `alpha`.
- Produces:

```python
def _gradient3d(video: torch.Tensor, alpha: float) -> torch.Tensor
def _divergence3d(field: torch.Tensor, alpha: float) -> torch.Tensor
```

- [ ] **Step 1: Write the failing weighted-adjoint test**

```python
import pytest
import torch

from gsdiff.prior.tv import _divergence3d, _gradient3d


@pytest.mark.parametrize("alpha", [0.0, 0.05, 0.3, 1.0, 2.0])
def test_weighted_gradient_divergence_are_negative_adjoint(alpha):
    torch.manual_seed(0)
    x = torch.randn(4, 5, 7, dtype=torch.float64)
    p = torch.randn(4, 5, 7, 3, dtype=torch.float64)
    p[-1, :, :, 0] = 0
    p[:, -1, :, 1] = 0
    p[:, :, -1, 2] = 0
    lhs = (_gradient3d(x, alpha) * p).sum()
    rhs = -(x * _divergence3d(p, alpha)).sum()
    relative_error = (lhs - rhs).abs() / lhs.abs().clamp_min(1e-15)
    assert relative_error.item() < 1e-10
```

- [ ] **Step 2: Observe RED**

Run:

```powershell
D:\conda\envs\spi\python.exe -m pytest tests/prior/test_tv3d.py::test_weighted_gradient_divergence_are_negative_adjoint -q
```

Expected: import fails because the helpers do not exist.

- [ ] **Step 3: Implement the two operators**

Add to `gsdiff/prior/tv.py`:

```python
def _gradient3d(video, alpha):
    T, H, W = video.shape
    grad = video.new_zeros(T, H, W, 3)
    grad[:-1, :, :, 0] = float(alpha) * (video[1:] - video[:-1])
    grad[:, :-1, :, 1] = video[:, 1:] - video[:, :-1]
    grad[:, :, :-1, 2] = video[:, :, 1:] - video[:, :, :-1]
    return grad


def _divergence3d(field, alpha):
    T, H, W, _ = field.shape
    div = field.new_zeros(T, H, W)
    a = float(alpha)
    div[1:] += a * (field[1:, :, :, 0] - field[:-1, :, :, 0])
    div[0] += a * field[0, :, :, 0]
    div[:, 1:] += field[:, 1:, :, 1] - field[:, :-1, :, 1]
    div[:, 0] += field[:, 0, :, 1]
    div[:, :, 1:] += field[:, :, 1:, 2] - field[:, :, :-1, 2]
    div[:, :, 0] += field[:, :, 0, 2]
    return div
```

Refactor both divergence blocks and the forward-gradient block in
`TVPrior3D._chambolle3d` to call these helpers. Do not retain a second copy of
the temporal divergence. Replace the 2D Chambolle implementation's
`torch.zeros(...)` buffers with `img.new_zeros(...)` as well, so the
`alpha=0` float64 equivalence test compares the same numerical precision rather
than silently down-casting the framewise reference.

- [ ] **Step 4: Add the `alpha=0` proximal equivalence test**

```python
from gsdiff.prior.tv import TVPrior, TVPrior3D


def test_alpha_zero_matches_framewise_2d_tv():
    video = torch.rand(3, 1, 8, 10, dtype=torch.float64)
    expected = TVPrior(max_iter=20).proximal(video, weight=0.08)
    actual = TVPrior3D(max_iter=20, temporal_weight=0.0).proximal(
        video, weight=0.08
    )
    assert expected.dtype == video.dtype
    assert actual.dtype == video.dtype
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
```

- [ ] **Step 5: Verify GREEN and the accumulated suite**

Extend the CUDA smoke only now: run one corrected `TVPrior3D.proximal` call on a
tiny CUDA tensor and assert finite output, unchanged device, and unchanged
dtype.

Run:

```powershell
D:\conda\envs\spi\python.exe -m pytest tests/prior/test_tv3d.py -q
D:\conda\envs\spi\python.exe -m pytest -m "not cuda" -q
D:\conda\envs\spi\python.exe -m pytest -m cuda -q
```

Expected: all parametrized adjoint cases and the proximal equivalence pass.

- [ ] **Step 6: Commit**

```powershell
git add gsdiff/prior/tv.py tests/prior/test_tv3d.py tests/test_cuda_smoke.py docs/reproducibility/implementation-log.md
git commit -m "fix: correct weighted 3d tv adjoint"
```

## Task 3: Make the TV Definition Internally Consistent

**Files:**

- Modify: `gsdiff/prior/tv.py`
- Modify: `gsdiff/solver/sgd.py`
- Modify: `gsdiff/solver/admm.py`
- Modify: `tests/prior/test_tv3d.py`
- Create: `tests/solver/test_tv_regularizer.py`

**Interfaces:**

- Consumes: `[T,1,H,W]` video and temporal weight.
- Produces two deliberately distinct objectives:

```python
def isotropic_tv2d_sum(video: torch.Tensor) -> torch.Tensor
def isotropic_tv3d_sum(
    video: torch.Tensor,
    temporal_weight: float,
) -> torch.Tensor
def anisotropic_tv_mean(
    video: torch.Tensor,
    temporal_weight: float = 0.0,
) -> torch.Tensor
```

The z-step proximal prior and its reported energy use the pointwise isotropic
sum. The differentiable theta-step regularizer retains the historical
componentwise anisotropic mean. The implementation must name this distinction
instead of silently changing the optimized objective.

- [ ] **Step 1: Write the failing isotropic-energy test**

```python
import math
import pytest
import torch

from gsdiff.prior.tv import TVPrior, TVPrior3D


def test_tv3d_energy_uses_pointwise_isotropic_norm():
    x = torch.zeros(2, 1, 2, 2)
    x[0, 0, 0, 1] = 1.0
    x[0, 0, 1, 0] = 1.0
    energy = TVPrior3D(temporal_weight=0.0).energy(x)
    assert math.isclose(energy, 2.0 + math.sqrt(2.0), rel_tol=1e-6)


def test_tv2d_energy_uses_pointwise_isotropic_norm():
    x = torch.zeros(1, 1, 2, 2)
    x[0, 0, 0, 1] = 1.0
    x[0, 0, 1, 0] = 1.0
    energy = TVPrior().energy(x)
    assert math.isclose(energy, 2.0 + math.sqrt(2.0), rel_tol=1e-6)
```

Run this test and record the current anisotropic value as the expected RED
failure.

- [ ] **Step 2: Write the failing shared-regularizer test**

```python
import torch

from gsdiff.prior.tv import (
    anisotropic_tv_mean,
    isotropic_tv2d_sum,
    isotropic_tv3d_sum,
)


def test_theta_and_z_tv_objectives_are_explicitly_distinct():
    x = torch.zeros(2, 1, 2, 2)
    x[0, 0, 0, 1] = 1.0
    x[0, 0, 1, 0] = 1.0
    assert isotropic_tv3d_sum(x, temporal_weight=0.5).item() == pytest.approx(
        math.sqrt(2.0) + 2.0 * math.sqrt(1.25)
    )
    assert anisotropic_tv_mean(x, temporal_weight=0.5).item() == pytest.approx(
        1.25
    )


def test_tv3d_alpha_zero_energy_matches_independent_tv2d():
    x = torch.randn(4, 1, 5, 7, dtype=torch.float64)
    expected = sum(isotropic_tv2d_sum(frame[None]) for frame in x)
    actual = isotropic_tv3d_sum(x, temporal_weight=0.0)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
```

Expected RED: the named helpers do not exist and the current `energy()` uses a
componentwise absolute sum.

- [ ] **Step 3: Implement the shared differentiable regularizer**

Use `_gradient3d(video[:, 0], temporal_weight)` for the isotropic sum and
implement the anisotropic mean from the absolute component differences using
the existing valid-edge convention. `TVPrior.energy()` must call
`isotropic_tv2d_sum`; `TVPrior3D.energy()` must call `isotropic_tv3d_sum`.

Keep `sgd.tv_loss` and `admm._soft_tv` at their old signatures, but make each a
thin compatibility wrapper around `anisotropic_tv_mean`. Do not change either
solver's coefficient or reduction in this task.

Add a proximal regression:

```python
@pytest.mark.parametrize("weight", [0.01, 0.2])
def test_tv3d_alpha_zero_prox_matches_independent_tv2d(weight):
    x = torch.randn(4, 1, 5, 7, dtype=torch.float64)
    prior3d = TVPrior3D(temporal_weight=0.0, max_iter=30)
    prior2d = TVPrior(max_iter=30)
    expected = torch.stack([prior2d.proximal(f[None], weight)[0] for f in x])
    actual = prior3d.proximal(x, weight)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
```

- [ ] **Step 4: Update mathematical documentation**

Correct `README.md`, `THEORY.md`, and `CLAUDE.md` so they state the two
definitions precisely: the z-step proximal/energy is pointwise isotropic sum;
the theta-step soft regularizer is componentwise anisotropic mean. Remove any
claim that treats them as interchangeable.

- [ ] **Step 5: Verify and commit**

Run:

```powershell
D:\conda\envs\spi\python.exe -m pytest tests/prior/test_tv3d.py tests/solver/test_tv_regularizer.py -q
D:\conda\envs\spi\python.exe -m pytest -m "not cuda" -q
git diff --check
```

Commit:

```powershell
git add gsdiff/prior/tv.py gsdiff/solver/sgd.py gsdiff/solver/admm.py tests/prior/test_tv3d.py tests/solver/test_tv_regularizer.py README.md THEORY.md CLAUDE.md docs/reproducibility/implementation-log.md
git commit -m "fix: make tv definitions match implementations"
```

## Task 4: Make Diffusion Annealing Endpoints Exact

**Files:**

- Create: `tests/prior/test_diffusion_schedule.py`
- Modify: `gsdiff/prior/diffusion.py`
- Modify: `train.py`

**Interfaces:**

- Consumes: zero-based call index, number of calls, start/end sigma.
- Produces:

```python
def log_annealed_sigma(index: int, count: int,
                       sigma_start: float, sigma_end: float) -> float
```

`DiffusionPrior` also exposes:

```python
@property
def last_sigma(self) -> float | None
```

- [ ] **Step 1: Write failing pure-function tests**

```python
import pytest

from gsdiff.prior.diffusion import log_annealed_sigma


def test_one_step_schedule_uses_end_sigma():
    assert log_annealed_sigma(0, 1, 0.3, 0.05) == pytest.approx(0.05)


def test_multistep_schedule_hits_requested_endpoints():
    values = [log_annealed_sigma(i, 5, 0.3, 0.05) for i in range(5)]
    assert values[0] == 0.3
    assert values[-1] == 0.05
    assert all(a > b for a, b in zip(values, values[1:]))
```

Expected RED: import failure.

- [ ] **Step 2: Implement the pure schedule**

```python
def log_annealed_sigma(index, count, sigma_start, sigma_end):
    if count < 1:
        raise ValueError("count must be >= 1")
    if not 0 <= index < count:
        raise IndexError(f"index {index} outside [0, {count})")
    if count == 1:
        return float(sigma_end)
    if index == 0:
        return float(sigma_start)
    if index == count - 1:
        return float(sigma_end)
    t = index / (count - 1)
    return math.exp(
        (1.0 - t) * math.log(sigma_start) + t * math.log(sigma_end)
    )
```

Make `DiffusionPrior._current_sigma()` call this helper with
`self._call_count` and `self._n_steps`.

- [ ] **Step 3: Test actual call accounting without loading a checkpoint**

```python
from gsdiff.prior.diffusion import DiffusionPrior


def test_prior_current_sigma_uses_exact_last_outer_step():
    prior = DiffusionPrior.__new__(DiffusionPrior)
    prior.sigma_start = 0.3
    prior.sigma_end = 0.05
    prior._n_steps = 4
    prior._call_count = 0
    seen = []
    for _ in range(4):
        seen.append(prior._current_sigma())
        prior._call_count += 1
    assert seen == pytest.approx([0.3, 0.16509636, 0.09085603, 0.05])
```

- [ ] **Step 4: Test the sigma actually consumed by `proximal()`**

Construct a `DiffusionPrior` instance with a zero-output dummy denoiser so no
checkpoint or CUDA device is needed. Test `set_n_steps(0)` raises `ValueError`,
`set_n_steps(1)` makes the one call consume `sigma_end`, and four successive
calls set `last_sigma` to the exact four scheduled values. Test the internal
DDIM ladder contains both its requested start and `sigma_min` endpoints.

Inside `proximal()`, set `_last_sigma` immediately after computing the current
sigma and before incrementing `_call_count`. Initialize `_last_sigma = None` in
`__init__` and reset it in `set_n_steps()`.

- [ ] **Step 5: Correct training logs**

In `train.py`, record `prior.last_sigma` after the current z-step. Do not call
the private `_current_sigma()` after `proximal()` to label the just-completed
iteration. Store the consumed value in machine-readable history as
`sigma_used`.

- [ ] **Step 6: Verify and commit**

Run:

```powershell
D:\conda\envs\spi\python.exe -m pytest tests/prior/test_diffusion_schedule.py -q
D:\conda\envs\spi\python.exe -m pytest -m "not cuda" -q
```

Commit:

```powershell
git add gsdiff/prior/diffusion.py train.py tests/prior/test_diffusion_schedule.py docs/reproducibility/implementation-log.md
git commit -m "fix: make diffusion schedule endpoints exact"
```

## Task 5: Isolate Frozen Parameters, Gradient Clipping, and LR Floors

**Files:**

- Create: `gsdiff/solver/gradients.py`
- Create: `tests/solver/test_sgd_parameters.py`
- Create: `tests/solver/test_gradient_groups.py`
- Modify: `gsdiff/solver/sgd.py`
- Modify: `gsdiff/solver/admm.py`

**Interfaces:**

- Consumes: explicit scene/motion parameter groups and total step count.
- Produces:

```python
def freeze_parameters(parameters: Iterable[torch.nn.Parameter]) -> None
def active_parameters(
    parameters: Iterable[torch.nn.Parameter],
) -> list[torch.nn.Parameter]
def clip_grad_groups(
    groups: Iterable[Iterable[torch.nn.Parameter]],
    max_norm: float,
) -> list[torch.Tensor]
def cosine_multiplier(step: int, total_steps: int,
                      final_ratio: float = 0.1) -> float
```

- [ ] **Step 1: Add tiny differentiable forward fixtures**

In `tests/solver/test_sgd_parameters.py`, define a scene and motion module with
one scalar parameter each and a forward object whose output depends on both.
The fixture must expose `.scene`, `.motion`, `.parameters()`, `.H`, `.W`, and
the same call signature used by `SGDSolver`.

- [ ] **Step 2: Write the failing frozen-gradient test**

```python
def test_frozen_motion_has_no_grad_and_cannot_change_scene_clipping(dummy_solver_inputs):
    solver = make_solver(dummy_solver_inputs, freeze_motion=True)
    solver.step()
    motion_params = list(solver.fwd.motion.parameters())
    assert all(not p.requires_grad for p in motion_params)
    assert all(p.grad is None for p in motion_params)
```

Expected RED: frozen motion parameters still require gradients and receive
accumulated gradients.

- [ ] **Step 3: Prove frozen motion cannot affect the scene update**

Create two otherwise identical dummy forwards whose motion branch would
produce gradient norms of `1` and `1e6`. With `freeze_motion=True`, the scene
parameter after one step must be byte-identical in both runs. This directly
guards against inactive motion gradients entering clipping.

- [ ] **Step 4: Implement active-parameter handling**

When `freeze_motion` is true, call:

```python
freeze_parameters(mp)
```

The helper sets `requires_grad_(False)` and `parameter.grad = None`. Store
`_scene_params` and `_motion_params` explicitly in both solvers. Do not infer
logical groups later from a flattened module iterator.

- [ ] **Step 5: Write failing group-specific clipping tests**

For SGD and ADMM, construct scene and motion gradients with very different
norms. Assert clipping the motion group cannot rescale the scene group and vice
versa. Assert frozen/empty groups are ignored and returned norms preserve group
order.

Expected RED: both solvers currently pass a flattened scene+motion list to one
`clip_grad_norm_` call.

- [ ] **Step 6: Implement independent clipping**

`clip_grad_groups` calls `clip_grad_norm_` once per nonempty active group.
Both solvers call it with `[self._scene_params, self._motion_params]`. During
SGD motion warmup, zeroed scene gradients remain zero; they must not be
renormalized by motion gradients.

- [ ] **Step 7: Write the failing group-specific LR-floor tests**

```python
def test_sgd_parameter_groups_end_at_ten_percent_of_their_own_lr(
    dummy_solver_inputs,
):
    solver = make_solver(
        dummy_solver_inputs, lr_scene=0.009, lr_motion=0.15, n_steps=4
    )
    for _ in range(4):
        solver.step()
    lrs = [group["lr"] for group in solver.optimizer.param_groups]
    assert lrs == pytest.approx([0.0009, 0.015])
```

Add the equivalent ADMM test over its exact `n_outer * n_inner` schedule.

Expected RED: the motion group in both solvers ends at the scalar scene floor.

- [ ] **Step 8: Implement a multiplicative cosine schedule**

```python
def cosine_multiplier(step, total_steps, final_ratio=0.1):
    capped = min(max(step, 0), total_steps)
    phase = capped / total_steps
    return final_ratio + (1.0 - final_ratio) * 0.5 * (
        1.0 + math.cos(math.pi * phase)
    )
```

Use `LambdaLR` with the same multiplier for each group so each group preserves
its own base LR. Define and test exact step-count semantics so the last
declared optimizer step reaches `final_ratio` without an off-by-one.

- [ ] **Step 9: Verify and commit**

Run:

```powershell
D:\conda\envs\spi\python.exe -m pytest tests/solver/test_sgd_parameters.py tests/solver/test_gradient_groups.py -q
D:\conda\envs\spi\python.exe -m pytest -m "not cuda" -q
```

Commit:

```powershell
git add gsdiff/solver/gradients.py gsdiff/solver/sgd.py gsdiff/solver/admm.py tests/solver/test_sgd_parameters.py tests/solver/test_gradient_groups.py docs/reproducibility/implementation-log.md
git commit -m "fix: isolate solver parameter groups"
```

## Task 6: Align Rectangular Geometry and Field-of-View Boundaries

**Files:**

- Create: `tests/scene/test_rectangular_geometry.py`
- Modify: `gsdiff/baselines/inr.py`
- Modify: `gsdiff/baselines/recinr.py`

**Interfaces:**

- Consumes: pixel coordinates `(y,x)`, image shape `(H,W)`, SE(2) transform.
- Produces:

```python
def normalize_pixel_coordinates(
    coordinates: torch.Tensor, center: torch.Tensor, H: int, W: int
) -> torch.Tensor
```

- [ ] **Step 1: Write the failing corner-normalization test**

```python
from gsdiff.baselines.inr import normalize_pixel_coordinates


def test_rectangular_corners_map_to_unit_square():
    coords = torch.tensor([[0.0, 0.0], [31.0, 63.0]])
    center = torch.tensor([15.5, 31.5])
    actual = normalize_pixel_coordinates(coords, center, H=32, W=64)
    expected = torch.tensor([[-1.0, -1.0], [1.0, 1.0]])
    torch.testing.assert_close(actual, expected)
```

Expected RED: helper does not exist.

- [ ] **Step 2: Implement per-axis normalization**

```python
def normalize_pixel_coordinates(coordinates, center, H, W):
    half = coordinates.new_tensor([(H - 1) / 2.0, (W - 1) / 2.0])
    return (coordinates - center) / half
```

Use this helper in `INRForwardModel.render_video()` and `.norm_grid()`.

- [ ] **Step 3: Write the failing out-of-FOV masking test**

Use a constant-one canonical scene and a translation that moves every inverse
sample outside the normalized square. Assert that the rendered pixels are
zero, not border replicated and not SIREN extrapolation.

- [ ] **Step 4: Implement one physical boundary convention**

After inverse warping:

```python
inside = (x_norm.abs() <= 1.0).all(dim=-1)
queried = self.scene.query(x_norm.reshape(-1, 2)).view(T, self.H, self.W)
inten = queried * inside.to(queried.dtype)
```

Apply the physical zero boundary only in the shared forward mask above; do not
also change canonical classes' direct-query padding defaults without a
separate need. Store `Hc` and `Wc` on `GridCanonical`, and reshape prefit
targets to `(Hc, Wc)` after an exact size check. In `build_scene`, make absent
`Hc/Wc` fall back to requested `H/W` for `grid`, and pass requested `H/W` into
`lowrank`. Make `ReCINRCanonicalScene` store `(gh, gw)` rather than one square
`gh`; a scalar `grid_size` denotes the short-side resolution and the other
dimension preserves `H:W`, while an explicit `(gh, gw)` tuple is accepted.

- [ ] **Step 5: Verify ReCINR rectangular construction**

Drive construction through the real `build_scene` and `INRForwardModel` path.
Add parametrized tests for `siren`, `grid`, `lowrank`, and `recinr_se2` at
`32×64`, plus the Gaussian forward renderer. Each must return
`[T,1,32,64]`, obey the same FOV mask, and produce finite gradients.

- [ ] **Step 6: Verify and commit**

Run:

```powershell
D:\conda\envs\spi\python.exe -m pytest tests/scene/test_rectangular_geometry.py -q
D:\conda\envs\spi\python.exe -m pytest -m "not cuda" -q
```

Commit:

```powershell
git add gsdiff/baselines/inr.py gsdiff/baselines/recinr.py tests/scene/test_rectangular_geometry.py docs/reproducibility/implementation-log.md
git commit -m "fix: align rectangular inr geometry and boundaries"
```

## Task 7: Lock Forward, Gradient, and Data Reproducibility Contracts

**Files:**

- Create: `tests/forward/test_measurement_consistency.py`
- Create: `tests/scene/test_numerical_gradients.py`
- Create: `tests/data/test_reproducibility.py`
- Modify if RED requires it: `gsdiff/scene/gaussian2d.py`
- Modify if RED requires it: `gsdiff/motion/se2.py`
- Modify if RED requires it: `gsdiff/forward/spi.py`
- Modify if RED requires it: `gsdiff/data/simulation.py`
- Modify if RED requires it: `gsdiff/data/patterns.py`

**Interfaces:**

- Consumes: generated `SPIData`, a rendered video, and tiny float64 scene/motion
  models.
- Produces: evidence that one seed means one measurement dataset and that the
  differentiable forward path has the declared derivatives.

- [ ] **Step 1: Test fixed-seed byte identity**

Generate the same `32×40`, `T=4`, `K=64`, Bernoulli, `25 dB`,
`holdout_extra=16` dataset twice with seed 7. Assert `np.array_equal` for the
canonical image, frames, patterns, measurements, frame indices, time grid,
holdout patterns, holdout measurements, and holdout indices. Verify adding the
holdout does not alter any training array.

- [ ] **Step 2: Test seed sensitivity**

Repeat with seed 11 and assert stochastic patterns, noise, and holdout differ
while deterministic target/shape properties and dtypes remain correct.

- [ ] **Step 3: Test direct and interpolated measurement agreement**

On tiny non-square arrays, compare `SPIForwardModel.measure()` against explicit
NumPy inner products and compare `measure_interpolated()` against
`_measure_interpolated_np`. Include `K=1`, the final-frame boundary, and a
frame with no assigned measurement. Use float64 reference arithmetic and
`rtol=1e-10`, `atol=1e-12`.

- [ ] **Step 4: Write float64 finite-difference gradient tests**

Use CPU, `H=4`, `W=7`, `T=3`, `M=1`, deterministic parameters, and central
differences with a step near `1e-6`. Compare autograd with finite differences
for:

- one Gaussian center coordinate;
- one translation-velocity coordinate;
- angular velocity;
- an end-to-end measurement loss depending on both scene and motion.

Require finite values and relative error below `1e-5`, with a documented
absolute fallback near zero. The tests must traverse
`GaussianScene2D.render`, `SE2Motion.transform_centers`,
`SE2Motion.transform_covariances`, and `SPIForwardModel`.

Expected RED: pixel grids, identity matrices, measurement buffers, and
interpolation time indices currently contain hard-coded float32 tensors, so a
float64 tiny problem cannot retain one dtype end to end.

- [ ] **Step 5: Propagate dtype/device without changing physics**

All grids, identity matrices, time indices, and output buffers inherit the
participating tensor's dtype/device. Examples:

```python
torch.arange(H, device=dev, dtype=centers_t.dtype)
torch.eye(2, device=Sigma_t.device, dtype=Sigma_t.dtype)
video.new_empty(K)
```

When a module is converted with `.double()`, registered buffers and every
internally created tensor must follow. Preserve `(y,x)` coordinates, row-major
flattening, loop order, interpolation equations, and random algorithms.

- [ ] **Step 6: Run focused tests and fix only observed roots**

```powershell
D:\conda\envs\spi\python.exe -m pytest tests/forward/test_measurement_consistency.py tests/scene/test_numerical_gradients.py tests/data/test_reproducibility.py -q
```

If a test fails, document the observed difference and backward trace, add the
smallest production correction, and rerun RED/GREEN. Do not weaken bit-exact
data assertions or numerical tolerances to hide a dtype defect.

- [ ] **Step 7: Commit**

```powershell
git add tests/forward/test_measurement_consistency.py tests/scene/test_numerical_gradients.py tests/data/test_reproducibility.py
git add gsdiff/scene/gaussian2d.py gsdiff/motion/se2.py gsdiff/forward/spi.py gsdiff/data/simulation.py gsdiff/data/patterns.py docs/reproducibility/implementation-log.md
git commit -m "test: lock forward gradient and data contracts"
```

## Task 8: Add `metrics-v1` and Explicit Legacy Metrics

**Files:**

- Create: `gsdiff/evaluation/__init__.py`
- Create: `gsdiff/evaluation/metrics.py`
- Create: `tests/evaluation/test_metrics.py`
- Modify: `gsdiff/baselines/common.py`
- Modify: `gsdiff/utils.py`
- Modify: `train.py`

**Interfaces:**

- Consumes: GT and reconstruction arrays `[T,H,W]`.
- Produces:

```python
def fit_global_affine(
    gt: np.ndarray,
    recon: np.ndarray,
    *,
    nonnegative_slope: bool = True,
) -> tuple[float, float]
def apply_global_affine(recon: np.ndarray, slope: float,
                        intercept: float) -> np.ndarray
def evaluate_video_global_affine(
    gt: np.ndarray, recon: np.ndarray
) -> dict[str, object]
def evaluate_video_legacy_per_frame(
    gt: np.ndarray, recon: np.ndarray
) -> dict[str, object]
```

- [ ] **Step 1: Write the failing known-affine recovery test**

```python
import numpy as np
import pytest

from gsdiff.evaluation.metrics import evaluate_video_global_affine


def test_global_affine_metric_recovers_one_known_video_transform():
    gt = np.linspace(0, 1, 2 * 8 * 9).reshape(2, 8, 9)
    recon = (gt - 0.2) / 1.7
    result = evaluate_video_global_affine(gt, recon)
    assert result["alignment"]["slope"] == pytest.approx(1.7)
    assert result["alignment"]["intercept"] == pytest.approx(0.2)
    assert result["psnr_global_affine"] == pytest.approx(120.0)
    assert result["nrmse_global_affine_l2"] < 1e-12
```

Expected RED: module does not exist.

- [ ] **Step 2: Write the per-frame cheating regression test**

Construct two frames with different erroneous gains. Assert the legacy metric
is near-perfect but `psnr_global_affine` is lower because one global affine fit
cannot remove both gains.

Add a constant-prediction test and a negative-correlation test. The documented
policy is `slope >= 0`; when the unconstrained slope is negative or the
prediction variance is numerically zero, use `slope=0` and
`intercept=mean(gt)`. All returned metrics must remain finite.

- [ ] **Step 3: Implement `metrics-v1`**

Fit `[recon_flat, ones] @ [slope, intercept] ≈ gt_flat` with
the declared nonnegative-slope policy. Apply one pair to every frame, then use
the same `aligned = np.clip(slope * recon + intercept, 0, 1)` array for every
primary image metric:

```python
psnr_global_affine
ssim_global_affine
nrmse_global_affine_l2
psnr_legacy_per_frame_minmax
alignment = {"slope": ..., "intercept": ...}
definition_version = "metrics-v1"
```

Use `skimage.metrics.structural_similarity(..., data_range=1.0)` per frame.
Define nRMSE as `||aligned-gt||₂ / max(||gt||₂, eps)`. Define exact-recovery
PSNR with `data_range=1` and an MSE floor of `1e-12`, giving a documented
120-dB numerical cap distinct from the legacy 60-dB helper. Record
`psnr_mse_floor` and `psnr_cap_db` in the metric-definition metadata. Require
spatial dimensions of at least `7×7`; if later tiny tests need SSIM, pass a
declared largest valid odd `win_size` and test that policy explicitly. Keep the
legacy metric in a separately named function.

- [ ] **Step 4: Integrate without silently changing old keys**

The pure evaluator writes the new fields to `metrics.json`; a local all-in-one
`train.py` invocation may call that evaluator only after optimization has
ended. Its compatibility `results.json` keeps `mean_psnr`, `per_frame_psnr`,
and `dgi_psnr`, but the file root writes
`metric_definition_version="legacy-per-frame-minmax-v1"` and each legacy key
is also exposed under an unambiguous name. Baseline JSON uses the same root
definition field. No consumer may mistake any compatibility key for a primary
metric. Update `gsdiff.baselines.common.evaluate_video` to call the explicitly
named legacy function only in a compatibility evaluator; method-child code
cannot import or call it. Task 9 makes the blind child/evaluator boundary
enforceable before the experiment runner consumes `metrics-v1`.

- [ ] **Step 5: Verify and commit**

Run:

```powershell
D:\conda\envs\spi\python.exe -m pytest tests/evaluation/test_metrics.py -q
D:\conda\envs\spi\python.exe -m pytest -m "not cuda" -q
```

Commit:

```powershell
git add gsdiff/evaluation gsdiff/baselines/common.py gsdiff/utils.py train.py tests/evaluation/test_metrics.py docs/reproducibility/implementation-log.md
git commit -m "feat: add publication metric definitions"
```

## Task 9: Add Immutable Dataset Artifacts

**Files:**

- Create: `gsdiff/data/artifacts.py`
- Create: `tests/data/test_artifacts.py`
- Modify: `gsdiff/data/__init__.py`
- Modify: `train.py`
- Modify: `scripts/run_baselines.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class SPIAcquisitionData:
    # patterns, train/holdout buckets and indices, time grid, H/W/T/K, metadata
    dataset_identity_sha256: str
    ...


@dataclass(frozen=True)
class EvaluationTruth:
    # canonical image, GT video/trajectory, evaluator-only metadata
    dataset_identity_sha256: str
    ...


def save_acquisition_data(data: SPIAcquisitionData, path: Path) -> str
def load_acquisition_data(
    path: Path,
    *,
    expected_spec: Mapping[str, object] | None = None,
) -> SPIAcquisitionData
def save_evaluation_truth(data: EvaluationTruth, path: Path) -> str
def load_evaluation_truth(
    path: Path,
    *,
    expected_dataset_identity_sha256: str,
) -> EvaluationTruth
def artifact_sha256(path: Path) -> str
```

The paired artifact metadata includes `measurements-v1` and
`evaluation-truth-v1`, the complete resolved generation config, generator code
version, target/asset hash, seed, pattern family/order, time-assignment mode,
noise convention and parameters, motion parameters including
acceleration/beta, array schema, and content hashes. The acquisition file
contains no canonical image, GT frame, GT trajectory, evaluator path, or
GT-derived metric.
Both files carry the same canonical `dataset_identity_sha256`, computed from
the complete acquisition/generator spec before serialization. File-byte hashes
remain distinct integrity fields.

- [ ] **Step 1: Write the failing round-trip test**

Generate a tiny dataset, split it into acquisition and truth objects, save/load
both, and compare every dataclass field. Assert each returned SHA-256 equals a
direct file hash and has 64 lowercase hexadecimal characters. Save each object
twice to separate paths and require identical file SHA-256 values.

Inspect every member name and nested metadata value in the acquisition file and
prove no GT/canonical/trajectory/evaluation field is present. Start a real
method-child subprocess in a directory containing only `measurements.npz`; it
must reconstruct without any truth path in arguments, environment, or current
directory. A deliberately truth-seeking child must fail.

Swap truth files between two seeds and between two targets and require the truth
loader/evaluator to reject the pair before computing any metric. A
reconstruction records its expected dataset identity; evaluator validation
requires reconstruction, acquisition, and truth identities to match.

Add mismatch tests: an interpolation artifact loaded under a uniform forward
request, changed pattern family/order, changed noise convention, or changed
motion model must fail validation instead of silently trusting the caller's
YAML.

- [ ] **Step 2: Observe RED**

Run:

```powershell
D:\conda\envs\spi\python.exe -m pytest tests/data/test_artifacts.py -q
```

Expected: import failure.

- [ ] **Step 3: Implement deterministic NPZ serialization**

Store array fields in a compressed NPZ and scalar/string metadata in a
canonical JSON byte array named `__metadata_json__`. Reject object arrays and
missing required fields. Ensure ZIP member order, metadata, and timestamps are
deterministic so repeated serialization has the same bytes. Write to a sibling
temporary file, `fsync`, then `os.replace` into the requested path. Return
SHA-256 of the completed file.

The loaders reconstruct their separate dataclasses and restore optional holdout
arrays as `None` when metadata says they are absent. They validate stored
schema/content hashes and, when given an expected acquisition spec, reject any
physics-affecting mismatch. Neither loader infers the sibling path; possession
of `measurements.npz` does not grant a capability to open
`evaluation-truth.npz`. The truth loader requires an expected dataset identity
and fails closed on mismatch.

- [ ] **Step 4: Add one dataset input path to all method entry points**

`train.py` and `scripts/run_baselines.py` accept the same resolved
`measurements_path` in method-child mode. When present, both load exactly the
same acquisition artifact and do not call `generate_spi_data`. They output only
reconstruction, history, and method information; they cannot compute image or
trajectory metrics.

Add a test that monkeypatches `generate_spi_data` to raise and proves both
entry paths consume the saved acquisition artifact. A compatibility all-in-one
local command may additionally receive an explicit truth path, but it records
`truth_access="child_visible"` and `promotion_eligible=false`. This compatibility
path is structurally forbidden from calling `build_run_identity()`, creating a
complete manifest, or writing under content-addressed run roots. This plan
tests that the identity builder rejects its execution class; the next plan's
manifest/runner tests enforce the same rule at complete-manifest and run-root
boundaries. It therefore cannot share any lockable identity or status with
blind execution.

- [ ] **Step 5: Save raw arrays needed for publication**

Each post-fix training run writes:

```text
reconstruction.npz
iteration-history.jsonl
method-info.json
```

The child reconstruction file includes raw reconstruction, DGI when available,
estimated motion trajectory, and frame/time indices. It contains no GT,
evaluator-only metric, or independently normalized display array. After child
exit, a separate evaluator may combine this file with
`evaluation-truth.npz` to write `metrics.json` and a publication evidence
sidecar containing raw GT, GT trajectory, alignment coefficients, and source
hashes. It first enforces reconstruction/acquisition/truth identity equality.

- [ ] **Step 6: Verify and commit**

Run:

```powershell
D:\conda\envs\spi\python.exe -m pytest tests/data/test_artifacts.py -q
D:\conda\envs\spi\python.exe -m pytest -m "not cuda" -q
```

Commit:

```powershell
git add gsdiff/data/artifacts.py gsdiff/data/__init__.py train.py scripts/run_baselines.py tests/data/test_artifacts.py docs/reproducibility/implementation-log.md
git commit -m "feat: add immutable spi dataset artifacts"
```

## Task 10: Canonical Run Identity Primitives

**Files:**

- Modify: `gsdiff/experiments/identity.py`
- Create: `tests/experiments/test_identity.py`
- Modify: `docs/reproducibility/runtime.md`

**Interfaces:**

```python
@dataclass(frozen=True)
class RunIdentity:
    canonical_payload_json: bytes
    identity_sha256: str
    run_id: str

    def payload(self) -> Mapping[str, object]:
        """Return a newly decoded read-only view for manifest serialization."""
        ...


def sha256_file(path: Path) -> str
def resolved_config_sha256(config: Mapping[str, object]) -> str
def git_state(repo: Path) -> dict[str, object]
def source_tree_sha256(repo: Path, source_roots: Sequence[Path]) -> str
def build_run_identity(
    execution_class: Literal["blind_method_child"],
    scientific_contract_id: str,
    scientific_contract_sha256: str,
    method_id: str,
    target_id: str,
    motion_id: str,
    seed: int,
    config_sha256: str,
    dataset_identity_sha256: str,
    assets_sha256: Mapping[str, str],
    checkpoints_sha256: Mapping[str, str],
    code_commit: str,
    dirty_worktree: bool,
    source_tree_hash: str | None,
    dependencies_sha256: str,
    environment_lock_sha256: str,
    metric_version: str,
) -> RunIdentity
```

- [ ] **Step 1: Write failing canonicalization tests**

```python
def test_config_key_order_does_not_change_identity():
    left = {"solver": {"rho": 0.1, "type": "admm"}, "seed": 7}
    right = {"seed": 7, "solver": {"type": "admm", "rho": 0.1}}
    assert resolved_config_sha256(left) == resolved_config_sha256(right)


def test_checkpoint_hash_change_invalidates_identity():
    first = make_identity(checkpoints_sha256={"diffusion": "a" * 64})
    second = make_identity(checkpoints_sha256={"diffusion": "b" * 64})
    assert first.identity_sha256 != second.identity_sha256


def test_dirty_source_cannot_share_clean_identity():
    clean = make_identity(dirty_worktree=False, source_tree_hash=None)
    dirty = make_identity(dirty_worktree=True, source_tree_hash="c" * 64)
    assert clean.identity_sha256 != dirty.identity_sha256
```

Also require:

- dirty execution without `source_tree_hash` is rejected;
- clean execution with a source-tree hash is rejected;
- two distinct dirty source hashes produce distinct identities;
- ordering of named asset/checkpoint mappings does not change identity;
- each identity-bearing SHA is exactly 64 lowercase hexadecimal characters and
  `code_commit` is exactly 40 lowercase hexadecimal characters;
- changing dataset identity, dependency fingerprint, environment-lock hash,
  metric version, or scientific-contract content SHA changes identity.
- mutating the caller's original nested config/asset/checkpoint mappings after
  construction cannot alter the immutable canonical payload or its hash;
- a compatibility execution with child-visible truth is rejected before
  `RunIdentity` construction.

Test `source_tree_sha256()` itself in temporary Git repositories: clean tree,
staged content, unstaged content, rename, delete, executable-mode change,
binary content, allowed untracked source, excluded artifact/cache, and an
escaping symlink/junction. Every relevant working-source change must alter the
hash; excluded artifacts must not.

Expected RED: helpers do not yet exist.

- [ ] **Step 2: Implement exact identity rules**

Use `canonical_json_bytes` and SHA-256. Deep-copy and normalize every input,
then store only immutable canonical payload bytes inside `RunIdentity`; a
decoded payload view is newly constructed and read-only. Those bytes are the
sole input to `identity_sha256`. The display ID is:

```python
run_id = (
    f"{scientific_contract_id}--{method_id}--{target_id}--"
    f"{motion_id}--s{seed}--{identity_sha256[:8]}"
)
```

Validate IDs with `^[a-z0-9][a-z0-9_-]*$`. Reject NaN/Infinity through
`allow_nan=False`. The later manifest serializes this exact payload instead of
reconstructing a second, drift-prone set of identity fields. Measurements,
truth-file, and dataset-manifest byte hashes are manifest-integrity fields;
`dataset_identity_sha256` is the identity field.

Runtime validation accepts exactly
`execution_class="blind_method_child"` and stores it in the canonical payload.
`compatibility_unblinded`, missing, or unknown execution classes are rejected
before identity construction. The subsequent experiment manifest/runner accept
only a successfully constructed `RunIdentity`, providing the concrete
enforcement handoff; their content-root rejection tests remain in that plan.

`git_state()` returns the full commit, branch, dirty boolean, and baseline
`c03420784bc92b4e9b9eef8330cbd9571ebebc68`.

Publication campaigns reject a dirty worktree before execution. A diagnostic
run may opt in to dirty execution only when a deterministic hash of tracked
source changes and untracked source inputs enters the identity; it can never
be promoted to a locked aggregate. `source_tree_sha256()` hashes, in sorted
UTF-8 repository-relative path order, a framed `source-tree-v1` record
containing HEAD commit and the effective working-source snapshot: union of
tracked paths and allowlisted untracked source paths, with explicit
present/deleted marker, path-byte length/path bytes, regular/executable mode,
content length, and content SHA-256. This captures staged/unstaged edits,
renames as delete+add, deletes, modes, and binary files without Git rename or
text-diff heuristics. It rejects symlinks/junctions escaping the repository and
excludes `.git`, artifacts, caches, environments, `_trash`, and generated paper
outputs by literal policy. Dependency fingerprint and environment-lock hashes
are also identity-bearing.
These are primitives only: reusable-cache decisions, complete manifests,
atomic promotion, stale-path regression tests, failure propagation, and
incremental aggregation are implemented and tested in the subsequent
experiment plan, not claimed complete by this task.

- [ ] **Step 3: Verify the entire foundation**

Run:

```powershell
D:\conda\envs\spi\python.exe -m pytest -m "not cuda" -q
D:\conda\envs\spi\python.exe -m compileall -q gsdiff scripts train.py
D:\conda\envs\spi\python.exe scripts\reproducibility\verify_environment_lock.py --strict
D:\conda\envs\spi\python.exe scripts\reproducibility\verify_implementation_provenance.py --strict
git diff --check
```

Expected: all CPU tests pass, compilation exits zero, and no whitespace errors
are reported.

- [ ] **Step 4: Commit**

```powershell
git add gsdiff/experiments/identity.py tests/experiments/test_identity.py docs/reproducibility/runtime.md docs/reproducibility/implementation-log.md
git commit -m "feat: add canonical experiment identities"
```

- [ ] **Step 5: Close the immutable implementation log**

Append Task 10's exact commit, final accumulated-test evidence, and any approved
deviations. Commit only the log, then do not edit it during the completion
gate:

```powershell
git add docs/reproducibility/implementation-log.md
git commit -m "docs: close correctness implementation log"
```

## Plan Completion Gate

Before moving to the final-experiments plan:

```powershell
$ErrorActionPreference = "Stop"
function Invoke-Checked {
    param([string]$Label, [scriptblock]$Command)
    & $Command
    $code = $LASTEXITCODE
    if ($code -ne 0) { throw "$Label failed with exit code $code." }
}

Invoke-Checked "CPU tests" { & 'D:\conda\envs\spi\python.exe' -m pytest -m "not cuda" -q }
$cudaStarted = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
$cudaReport = Join-Path ([IO.Path]::GetTempPath()) ("gsdiff-cuda-" + [guid]::NewGuid().ToString("N") + ".xml")
Invoke-Checked "CUDA tests" { & 'D:\conda\envs\spi\python.exe' -m pytest -m cuda -q "--junitxml=$cudaReport" }
Invoke-Checked "CUDA report verification" { & 'D:\conda\envs\spi\python.exe' scripts\reproducibility\verify_pytest_junit.py $cudaReport --created-after-utc $cudaStarted --min-passed 1 --max-skipped 0 }
Remove-Item -LiteralPath $cudaReport -Force
Invoke-Checked "Compileall" { & 'D:\conda\envs\spi\python.exe' -m compileall -q gsdiff scripts train.py }
Invoke-Checked "Environment lock" { & 'D:\conda\envs\spi\python.exe' scripts\reproducibility\verify_environment_lock.py --strict }
Invoke-Checked "Plan provenance" { & 'D:\conda\envs\spi\python.exe' scripts\reproducibility\verify_implementation_provenance.py --strict }
Invoke-Checked "Whitespace check" { git diff --check }
$dirty = git status --porcelain=v1 --untracked-files=all
if ($LASTEXITCODE -ne 0 -or $dirty) { throw "Correctness gate requires a clean worktree." }
```

Required evidence:

- zero CPU failures;
- at least one executed CUDA smoke test, zero CUDA failures, and zero CUDA
  skips;
- weighted adjoint tolerance below `1e-10` for all five alpha values;
- exact diffusion endpoint tests pass;
- global-affine metric tests pass;
- dataset artifacts round-trip bit-exactly;
- method entry points require no truth and pass the isolated blind fixture; the
  runner-level allowlist/audit capability boundary is proved in the experiment
  plan;
- environment and immutable-plan provenance verifiers pass;
- working tree is clean;
- task review and whole-plan review have no unresolved critical or important
  findings.
