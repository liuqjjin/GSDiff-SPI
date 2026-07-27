# Implementation ledger

This permanent append-only ledger is intentionally empty of implementation records at initialization. Future task progress, deviations, RED/GREEN evidence, and commits are appended below without modifying approved specifications or plans.

## Task 0 — Isolation consent and starting state (2026-07-27)

- Plan baseline and starting HEAD: `abca49b36439efc6cb607a45c1601e50d84d6656`.
- Legacy evidence baseline: `c03420784bc92b4e9b9eef8330cbd9571ebebc68`; approved design commit: `24c1959599d9d775114d068f6de41ef2e31b5e36`.
- The user explicitly declined a separate worktree and instructed us to implement in place. Detection confirmed a normal checkout: resolved Git directory equals common directory, no superproject, and no submodule.
- Initial `git status --porcelain=v1` output was empty. The immutable design and three plan inputs match `plan_baseline_commit`; their SHA-256 values and provenance validation are recorded in `implementation-provenance.json`.

### No-install repository baseline (2026-07-27T12:06:58.1736552+08:00)

- Interpreter: `D:\conda\envs\spi\python.exe` (`Python 3.12.13`); no dependencies installed.
- `D:\conda\envs\spi\python.exe -m compileall -q gsdiff scripts train.py` → observed exit `0`.
- `D:\conda\envs\spi\python.exe -c "import gsdiff; print('import_gsdiff=ok')"` → observed output `import_gsdiff=ok`, observed exit `0`.
- A tracked-file scan found no test files and no pytest/tox/nox/project test configuration, so no test command was discoverable.
- Independent baseline-record validation succeeded: it checked the recorded schema, required command definitions, and recorded status fields. It does not re-execute the baseline or prove historical subprocess exit codes. Overall observed baseline status: **passed**.

## Task 1 — Reproducible test runtime (2026-07-27)

- Prior task commit and clean starting HEAD:
  `2e53688a07ea619e6409a60cbaba53f5ca6cb385` (`git status --short
  --branch` showed only
  `## debug/admm-vs-sgd...origin/debug/admm-vs-sgd [ahead 4]`).
- Authoritative interpreter: `D:\conda\envs\spi\python.exe`.
- Distribution fingerprints use normalized lowercase names (runs of `-`, `_`,
  and `.` mapped to `-`), version strings, sorting by `(name, version)`, compact
  canonical UTF-8 JSON, and SHA-256.
- Pre-install distribution fingerprint: 61 records,
  `d90c3b9a96bee35607727c9082f85ad235bf7983e559b8754d8ddbf738e0e164`.
- `D:\conda\envs\spi\python.exe -m pip install -r requirements-dev.txt` →
  exit `0`; installed `attrs==26.1.0`, `jsonschema==4.25.1`,
  `jsonschema-specifications==2025.9.1`, `referencing==0.37.0`, and
  `rpds-py==2026.6.3`. Pip emitted one workstation PATH notice because
  `D:\conda\envs\spi\Scripts` is not on `PATH`; no package-resolution warning
  or error occurred.
- Immediate `D:\conda\envs\spi\python.exe -m pip check` → exit `0`,
  `No broken requirements found.`
- Post-install distribution fingerprint: 66 records,
  `eb3af5ece140fc59207647f151a9d2220f08a572e43fb4bb73a20189fa135465`.
- The sorted `requirements-lock.txt` contains all 64 entries reported by
  `D:\conda\envs\spi\python.exe -m pip list --format=freeze`; an independent
  comparison reported `lock_matches_sorted_pip_list=True`.

### RED/GREEN evidence

- Required focused RED, after dependency installation and while
  `gsdiff.experiments.identity` was absent:
  `D:\conda\envs\spi\python.exe -m pytest tests/test_runtime_contract.py
  tests/reproducibility/test_environment_lock.py -q` → exit `1`, collection
  interrupted with two errors; both were
  `ModuleNotFoundError: No module named 'gsdiff.experiments'`. No test ran.
- After the minimal identity implementation, the same focused command → exit
  `0`, `10 passed in 0.50s`.
- Verifier import RED:
  `D:\conda\envs\spi\python.exe -m pytest
  tests/reproducibility/test_environment_lock.py
  tests/reproducibility/test_implementation_provenance.py
  tests/reproducibility/test_pytest_junit_verifier.py -q` → exit `1`, three
  collection errors because `scripts.reproducibility` was absent.
- The first verifier implementation run exercised the negative paths and
  reported `3 failed, 26 passed in 2.73s`. All three failures exposed a real
  aliasing error: mutating the stored dependency, ABI, or numerical-environment
  payload also mutated the supplied current fingerprint. Snapshotting the
  fingerprint with a deep copy was the minimal fix.
- The verifier-focused GREEN rerun → exit `0`, `29 passed in 2.46s`.
- Final focused GREEN:
  `D:\conda\envs\spi\python.exe -m pytest tests/test_runtime_contract.py
  tests/reproducibility/test_environment_lock.py -q` → exit `0`,
  `19 passed in 1.34s`.

### Locked runtime and exact verification

- Workstation runtime preflight output: Python `3.12.13`, PyTorch
  `2.8.0+cu128`, CUDA build `12.8`, device
  `NVIDIA GeForce RTX 5060 Ti`. The canonical full-environment lock records
  driver `596.21` and has fingerprint SHA-256
  `b5d6922a9f3a9638ee8826b9a74f00998cd3ac81aa25c03de016358e0e435a56`.
- `D:\conda\envs\spi\python.exe -m pytest -m "not cuda" -q` → exit `0`,
  `35 passed, 1 deselected in 5.06s`.
- `D:\conda\envs\spi\python.exe -m pytest -m cuda -q` → exit `0`,
  `1 passed, 35 deselected in 0.64s`; the CUDA smoke executed and there were
  zero skips.
- `D:\conda\envs\spi\python.exe
  scripts\reproducibility\verify_environment_lock.py --strict` → exit `0`,
  strict verification passed with fingerprint
  `b5d6922a9f3a9638ee8826b9a74f00998cd3ac81aa25c03de016358e0e435a56`.
- `D:\conda\envs\spi\python.exe
  scripts\reproducibility\verify_implementation_provenance.py --strict` →
  exit `0`; four immutable inputs and current commit
  `2e53688a07ea619e6409a60cbaba53f5ca6cb385` verified.
- Fresh full-suite JUnit boundary:
  `2026-07-27T04:24:44.8695103+00:00`.
  `D:\conda\envs\spi\python.exe -m pytest -q
  --junitxml=.superpowers\sdd\2026-07-27-gsdiff-correctness-reproducibility\task-1-junit.xml`
  → exit `0`, `36 passed in 2.90s`.
- `D:\conda\envs\spi\python.exe
  scripts\reproducibility\verify_pytest_junit.py
  .superpowers\sdd\2026-07-27-gsdiff-correctness-reproducibility\task-1-junit.xml
  --created-after-utc 2026-07-27T04:24:44.8695103+00:00` → exit `0`,
  `tests=36 failures=0 errors=0 skipped=0`.
- Final `D:\conda\envs\spi\python.exe -m pip check` → exit `0`,
  `No broken requirements found.`
- Test output was pristine: no warnings, unexpected skips, or hidden noise.
- After staging exactly the Task 1 scope, both `git diff --check` and
  `git diff --cached --check` → exit `0` with silent output.

## Task 1 Fix Round 1 — Harden reproducibility verifiers (2026-07-27)

- Fix base and clean starting HEAD:
  `d82362257824ae8f6f6f0664a203820f1c2843c9`; Task 0 and the original Task 1
  commit were not amended or rewritten.
- Scope: close coordinated-substitution gaps in strict implementation
  provenance; make JUnit inspection namespace- and nested-suite-safe; use
  integer nanosecond freshness with conservative exact-boundary rejection.
- `implementation-provenance.json`, the four immutable documents, and
  `environment-lock.json` were not edited. The environment fingerprint remains
  `b5d6922a9f3a9638ee8826b9a74f00998cd3ac81aa25c03de016358e0e435a56`.

### Provenance RED/GREEN

- Added coordinated immutable-path/hash substitution; exact legacy, design,
  start, and plan commit substitution; missing/forged `in_place`,
  linked-worktree, and submodule facts; and controlled temporary-Git-history
  cases for both Task-0 direct parents and terminal-to-current ancestry.
- Initial command:
  `D:\conda\envs\spi\python.exe -m pytest
  tests/reproducibility/test_implementation_provenance.py -q` → exit `1`,
  `14 failed, 8 passed in 11.43s`. One coordinated-path fixture initially used
  a file absent from the selected baseline tree and therefore failed for the
  wrong setup reason; it was replaced with four tracked files before
  production hardening. Its isolated rerun then failed correctly because the
  old verifier accepted the coordinated substitution.
- Root cause: every purported anchor came from the untrusted provenance JSON;
  the verifier checked only internal consistency and broad ancestry.
- Minimal hardening anchors the exact four paths/hashes, exact legacy/design/
  start/plan commits, first Task-0 commit and parent, Task-0 terminal commit and
  parent, terminal-to-current ancestry, `in_place` decision, recorded normal
  checkout facts, actual non-linked/non-submodule state, and the recorded
  worktree path/head/branch relationship.
- Focused GREEN, same file command → exit `0`,
  `22 passed in 9.74s`. The real immutable provenance passed; no genuine
  recorded-state mismatch was exposed.

### JUnit RED/GREEN

- Added fully namespaced passing XML; namespace-qualified failure, error, and
  skip fail-open cases; nested child-suite and aggregate declaration cases;
  and exact/±1 nanosecond freshness boundaries.
- RED command:
  `D:\conda\envs\spi\python.exe -m pytest
  tests/reproducibility/test_pytest_junit_verifier.py -q` → exit `1`,
  `9 failed, 11 passed in 0.23s`.
- Root cause: raw qualified tags were compared to unqualified literals,
  declarations were checked only at a flat aggregate level, and floating-point
  `st_mtime` was converted through microsecond `datetime`.
- Minimal hardening compares XML local names, counts outcome-bearing testcases,
  validates every suite recursively plus any aggregate root against its own
  descendant testcase reality, parses 1–9 fractional UTC digits to integer
  epoch nanoseconds, compares against `st_mtime_ns`, and requires
  `mtime_ns > created_after_utc_ns`.
- Focused GREEN, same file command → exit `0`,
  `20 passed in 0.19s`.
- Combined strict-verifier unit suites:
  `D:\conda\envs\spi\python.exe -m pytest
  tests/reproducibility/test_implementation_provenance.py
  tests/reproducibility/test_pytest_junit_verifier.py -q` → exit `0`,
  `42 passed in 9.91s`.

### Fix Round 1 verification

- `D:\conda\envs\spi\python.exe -m pytest -m "not cuda" -q` → exit `0`,
  `61 passed, 1 deselected in 10.78s`.
- `D:\conda\envs\spi\python.exe -m pytest -m cuda -q` → exit `0`,
  `1 passed, 61 deselected in 0.48s`; one CUDA test executed and zero skipped.
- `D:\conda\envs\spi\python.exe
  scripts\reproducibility\verify_environment_lock.py --strict` → exit `0`,
  strict verification passed with unchanged fingerprint
  `b5d6922a9f3a9638ee8826b9a74f00998cd3ac81aa25c03de016358e0e435a56`.
- `D:\conda\envs\spi\python.exe
  scripts\reproducibility\verify_implementation_provenance.py --strict` →
  exit `0`; all four exact immutable inputs and exact Task-0/Git/worktree
  relationships verified at current commit
  `d82362257824ae8f6f6f0664a203820f1c2843c9`.
- Fresh JUnit boundary:
  `2026-07-27T04:49:26.969501800Z`.
  `D:\conda\envs\spi\python.exe -m pytest -q
  --junitxml=.superpowers\sdd\2026-07-27-gsdiff-correctness-reproducibility\task-1-fix-round-1-junit.xml`
  → exit `0`, `62 passed in 11.11s`.
- `D:\conda\envs\spi\python.exe
  scripts\reproducibility\verify_pytest_junit.py
  .superpowers\sdd\2026-07-27-gsdiff-correctness-reproducibility\task-1-fix-round-1-junit.xml
  --created-after-utc 2026-07-27T04:49:26.969501800Z` → exit `0`,
  `tests=62 failures=0 errors=0 skipped=0`.
- `D:\conda\envs\spi\python.exe -m pip check` → exit `0`,
  `No broken requirements found.`
- `git diff --check` → exit `0` with silent output. Test and verifier output
  was pristine: no warnings, unexpected skips, or hidden noise.
- After staging exactly the five Fix Round 1 files, both
  `git diff --cached --check` and `git diff --check` → exit `0` with silent
  output.

## Task 2 — Correct weighted 3D-TV adjoint (2026-07-27)

- Prior approved commit and clean starting HEAD:
  `e14f847cc5b81d3adee11a31dfdeb0552d3b80a6` on
  `debug/admm-vs-sgd`.
- Root cause: the temporal forward gradient was scaled by `alpha`, but both
  duplicated temporal divergence blocks were not. The existing video
  `[T,H,W]` and dual-field `[T,H,W,3]` conventions matched the task brief.

### TDD RED/GREEN

- Added the deterministic float64 weighted-adjoint test with
  boundary-compatible dual fields before the helpers existed.
- Exact RED command:
  `D:\conda\envs\spi\python.exe -m pytest
  tests/prior/test_tv3d.py::test_weighted_gradient_divergence_are_negative_adjoint
  -q` → exit `1` during collection with the expected
  `ImportError: cannot import name '_divergence3d'`; one collection error and
  no collectors.
- Added only `_gradient3d` and `_divergence3d`, with `alpha` applied to the
  temporal component of each operator and one-sided zero-flux boundaries.
- Exact initial GREEN command, identical to RED → exit `0`,
  `5 passed in 0.04s`.
- Added the float64 `alpha=0` framewise proximal equivalence test. Its first
  isolated run passed (`1 passed in 0.05s`) because the old 2D and 3D paths
  used matching float32 dual buffers and promoted only the returned tensors.
- Refactored both 3D divergence sites and the 3D forward-gradient site to call
  the helpers, removed the duplicate temporal divergence, and changed only the
  relevant 2D/3D Chambolle buffers to input-derived `new_zeros`.
- Final focused command:
  `D:\conda\envs\spi\python.exe -m pytest tests/prior/test_tv3d.py -q` →
  exit `0`, `6 passed in 0.07s`.

### Numerical evidence

- Deterministic float64 weighted-adjoint relative errors were:
  `alpha=0`: `2.36821363826887335e-16`;
  `alpha=0.05`: `8.41102715057286788e-16`;
  `alpha=0.3`: `1.29726451974729065e-16`;
  `alpha=1`: `1.33563412971027626e-15`;
  `alpha=2`: `5.65907821739121724e-16`.
  The maximum was `1.33563412971027626e-15`, below `1e-10`.
- For `alpha=0`, the framewise 2D and 3D proximal outputs had maximum absolute
  error `0.0`; both outputs preserved `torch.float64`.

### Verification

- Accumulated CPU suite:
  `D:\conda\envs\spi\python.exe -m pytest -m "not cuda" -q` → exit `0`,
  `67 passed, 1 deselected in 11.10s`.
- Real CUDA suite after extending the existing 4x7x3 Gaussian+SE(2) smoke with
  a tiny corrected `TVPrior3D.proximal` call:
  `D:\conda\envs\spi\python.exe -m pytest -m cuda -q` → exit `0`,
  `1 passed, 67 deselected in 0.50s`; one CUDA test executed and zero skipped.
- Strict environment verifier:
  `D:\conda\envs\spi\python.exe
  scripts\reproducibility\verify_environment_lock.py --strict` → exit `0`,
  fingerprint
  `b5d6922a9f3a9638ee8826b9a74f00998cd3ac81aa25c03de016358e0e435a56`.
- Strict implementation-provenance verifier:
  `D:\conda\envs\spi\python.exe
  scripts\reproducibility\verify_implementation_provenance.py --strict` →
  exit `0`; four immutable inputs verified at current commit
  `e14f847cc5b81d3adee11a31dfdeb0552d3b80a6`.
- One full suite before commit:
  `D:\conda\envs\spi\python.exe -m pytest -q` → exit `0`,
  `68 passed in 11.49s`.
- `D:\conda\envs\spi\python.exe -m pip check` → exit `0`,
  `No broken requirements found.`
- `git diff --check` → exit `0` with silent output. All test and verifier
  output was pristine, with no warnings or unexpected skips.

## Task 3 — Make TV definitions internally consistent (2026-07-27)

- Prior approved commit and clean starting HEAD:
  `953ce67b30b5abb33a35f9d76f89d36cf71cba4a` on
  `debug/admm-vs-sgd`.
- The scientifically intentional objectives remain distinct: the z-step
  proximal and reported prior energy use a pointwise isotropic sum, while the
  differentiable theta-step regularizer retains its historical componentwise
  anisotropic per-axis means. Solver coefficients and reductions were not
  changed.

### Two-phase TDD evidence

- Energy-value RED, before any production edit:
  `D:\conda\envs\spi\python.exe -m pytest
  tests/prior/test_tv3d.py::test_tv3d_energy_uses_pointwise_isotropic_norm
  tests/prior/test_tv3d.py::test_tv2d_energy_uses_pointwise_isotropic_norm -q`
  → exit `1`, `2 failed in 0.05s`. Both prior energy methods returned the
  historical anisotropic value `4.0`, rather than the required isotropic
  `2 + sqrt(2) = 3.414213562373095`.
- Named-helper/shared-regularizer RED, still before production edits:
  `D:\conda\envs\spi\python.exe -m pytest tests/prior/test_tv3d.py
  tests/solver/test_tv_regularizer.py -q` → exit `1`, collection interrupted
  with two expected `ImportError`s: `isotropic_tv2d_sum` and
  `anisotropic_tv_mean` did not exist.
- Minimal implementation introduced the three named tensor helpers, routed
  `TVPrior.energy` and `TVPrior3D.energy` through the isotropic helpers, and
  made both solver functions thin wrappers around the shared anisotropic
  helper. Focused GREEN, the same two-file command → exit `0`,
  `22 passed in 0.21s`.
- Deterministic float64 values were: 2D isotropic
  `3.4142135623730949`; 3D isotropic at `alpha=0.5`
  `3.6502815398728847`; historical anisotropic mean at `alpha=0.5`, `1.25`.
  Both solver wrappers had absolute error `0` against the shared helper.
- At `alpha=0`, the float64 energy absolute difference from the sum of
  independent framewise 2D energies was `2.8421709430404007e-14`; the
  proximal maximum absolute difference was `0` for both weights `0.01` and
  `0.2`.

### Documentation and verification

- `README.md`, `THEORY.md`, and `CLAUDE.md` now distinguish the theta-step
  componentwise anisotropic mean from the z-step pointwise isotropic sum.
  A targeted search found no remaining statements that equate their
  reductions or definitions.
- Accumulated CPU suite:
  `D:\conda\envs\spi\python.exe -m pytest -m "not cuda" -q` → exit `0`,
  `83 passed, 1 deselected in 12.17s`.
- Real CUDA suite:
  `D:\conda\envs\spi\python.exe -m pytest -m cuda -q` → exit `0`,
  `1 passed, 83 deselected in 0.61s`; one CUDA test executed and zero skipped.
- Full suite: `D:\conda\envs\spi\python.exe -m pytest -q` → exit `0`,
  `84 passed in 12.55s`.
- Strict environment verifier → exit `0`, fingerprint
  `b5d6922a9f3a9638ee8826b9a74f00998cd3ac81aa25c03de016358e0e435a56`.
- Strict implementation-provenance verifier → exit `0`; four immutable
  inputs verified at current commit
  `953ce67b30b5abb33a35f9d76f89d36cf71cba4a`.
- `D:\conda\envs\spi\python.exe -m pip check` → exit `0`,
  `No broken requirements found.`
- `git diff --check` → exit `0` with silent output. Test and verifier output
  was pristine, with no warnings or unexpected skips.

## Task 3 Fix Round 1 — Clarify theta-TV objective (2026-07-27)

- Fix base and clean starting HEAD:
  `79cff5fb4949001c2bd0e676d4d5b56eb339095e`; the Task 3 commit was not
  amended.
- Corrected `CLAUDE.md` so `f(theta)` is unambiguously data fidelity and the
  augmented objective and theta-step each show exactly one
  `lambda_soft * TVtheta` term. No solver coefficient or implementation
  changed.
- Added review-strength coverage, not a fabricated RED: a deterministic
  non-square float64 tensor computes the expected historical objective
  independently as `mean(abs(dy)) + mean(abs(dx)) +
  alpha * mean(abs(dt))`, then checks the shared helper and both compatibility
  wrappers. Its isolated run → exit `0`, `1 passed in 0.04s`.
- Focused Task 3 suite → exit `0`, `23 passed in 0.17s`.
- Accumulated CPU suite → exit `0`,
  `84 passed, 1 deselected in 12.13s`.
- Real CUDA suite → exit `0`, `1 passed, 84 deselected in 0.51s`; one CUDA
  test executed and zero skipped.
- Full suite → exit `0`, `85 passed in 12.28s`.
- Strict environment verification → exit `0`, fingerprint
  `b5d6922a9f3a9638ee8826b9a74f00998cd3ac81aa25c03de016358e0e435a56`.
- Strict implementation-provenance verification → exit `0`, four immutable
  inputs verified at
  `79cff5fb4949001c2bd0e676d4d5b56eb339095e`.
- Targeted `CLAUDE.md` consistency inspection printed the data-fidelity
  definition, augmented objective, explanatory definition, and theta-step,
  then reported `theta_objective_contradiction_hits=0`.
- `D:\conda\envs\spi\python.exe -m pip check` → exit `0`,
  `No broken requirements found.`
- `git diff --check` → exit `0` with silent output. All test and verifier
  output was pristine, with no warnings or unexpected skips.

## Task 4 — Make diffusion annealing endpoints exact (2026-07-27)

- Prior approved commit and clean starting HEAD:
  `fbaf6c0d6a796c225988432047e1b7c96a2aed48` on
  `debug/admm-vs-sgd`.
- Root cause: `_current_sigma()` divided the zero-based call count by
  `_n_steps`, so the final planned call used `(n-1)/n` and never reached
  `sigma_end`. After each z-step, `train.py` queried `_current_sigma()` again
  and therefore displayed the next call's sigma rather than the value just
  consumed.

### Strict TDD RED/GREEN

- Pure helper RED, before the helper existed:
  `D:\conda\envs\spi\python.exe -m pytest
  tests/prior/test_diffusion_schedule.py -q` → exit `1` during collection with
  the expected `ImportError: cannot import name 'log_annealed_sigma'`;
  one collection error.
- Minimal helper GREEN, identical command → exit `0`,
  `10 passed in 0.06s`. The helper validates `count`, the zero-based index,
  and positive sigma inputs; a one-call schedule returns the exact end, and
  multi-call schedules return exact Python-float endpoints with log-linear
  interiors.
- Actual prior/proximal RED, before changing prior accounting:
  the focused file → exit `1`, `9 failed, 10 passed in 0.13s`. The historical
  four-call sequence was
  `[0.3, 0.1916829312738817, 0.12247448713915889,
  0.07825422900366437]`; `last_sigma` and the inspectable DDIM ladder were
  absent, and `set_n_steps(0)` did not raise.
- Actual prior/proximal GREEN → exit `0`, `19 passed in 0.08s`. Tests exercise
  real `proximal()` on CPU through an actual `DiffusionPrior` constructor with
  only the external checkpoint/model boundary replaced by a shape-preserving
  zero-output denoiser. The real proximal input/output contract remained
  `[T,1,H,W]`.
- Training-history RED:
  `D:\conda\envs\spi\python.exe -m pytest
  tests/prior/test_diffusion_schedule.py::test_training_history_records_consumed_sigma_without_schedule_lookahead
  -q` → exit `1` during collection with the expected missing
  `_record_sigma_used` import.
- Training-history GREEN, identical command → exit `0`,
  `1 passed in 3.08s`; final focused Task 4 suite → exit `0`,
  `20 passed in 0.55s`.

### Endpoint and accounting evidence

- Four outer calls consume
  `[0.3, 0.1650963624447313, 0.09085602964160698, 0.05]`; a one-call
  schedule consumes exactly `0.05`.
- For requested start `0.3`, `sigma_min=0.002`, and three DDIM steps, the
  float32 linear ladder was
  `[0.30000001192092896, 0.2006666660308838, 0.10133333504199982,
  0.0020000000949949026]`; the log ladder was
  `[0.30000001192092896, 0.056462161242961884, 0.010626583360135555,
  0.0020000000949949026]`. Both include the requested endpoints at tensor
  precision and retain the historical decreasing order.
- `last_sigma` starts and resets to `None`, is read-only, and is set to the
  current outer sigma before `_call_count` increments. `set_n_steps(0)` now
  raises `ValueError`.
- After `solver.step()` completes the current z-step, `train.py` records the
  public `prior.last_sigma` as `info["sigma_used"]` and uses the same value for
  console output. A targeted tracked-Python search found `_current_sigma`
  only in its definition and inside `DiffusionPrior.proximal`; no
  post-proximal private lookahead label remains.

### Verification

- Accumulated CPU suite:
  `D:\conda\envs\spi\python.exe -m pytest -m "not cuda" -q` → exit `0`,
  `104 passed, 1 deselected in 15.31s`.
- Real CUDA suite:
  `D:\conda\envs\spi\python.exe -m pytest -m cuda -q` → exit `0`,
  `1 passed, 104 deselected in 1.07s`; exactly one CUDA test executed and
  zero skipped.
- Full suite:
  `D:\conda\envs\spi\python.exe -m pytest -q` → exit `0`,
  `105 passed in 23.91s`.
- Strict environment verification → exit `0`, fingerprint
  `b5d6922a9f3a9638ee8826b9a74f00998cd3ac81aa25c03de016358e0e435a56`.
- Strict implementation-provenance verification → exit `0`; four immutable
  inputs verified at the pre-Task commit
  `fbaf6c0d6a796c225988432047e1b7c96a2aed48`.
- `D:\conda\envs\spi\python.exe -m pip check` → exit `0`,
  `No broken requirements found.`
- `git diff --check` → exit `0` with silent output. All test and verifier
  output was pristine, with no warnings or unexpected skips.

## Task 4 Fix Round 1 — Persist consumed diffusion sigma (2026-07-27)

- Fix base and clean starting HEAD:
  `03818b50c4a750fd974bd46c41579cdcea9bb0cb`; the Task 4 commit was not
  amended.
- Non-finite validation RED:
  `D:\conda\envs\spi\python.exe -m pytest
  tests/prior/test_diffusion_schedule.py::test_schedule_rejects_non_finite_sigma_values
  -q` → exit `1`, `4 failed, 2 passed in 0.64s`. NaN and positive infinity
  were accepted for either endpoint; negative infinity already reached the
  prior nonpositive guard.
- After adding the minimal `math.isfinite` guard, the identical test command
  → exit `0`, `6 passed in 0.57s`. NaN and both infinities now fail closed
  for either endpoint, alongside the existing zero/negative coverage.
- Persisted-history RED:
  `D:\conda\envs\spi\python.exe -m pytest
  tests/prior/test_diffusion_schedule.py::test_results_json_persists_history_with_consumed_sigma
  -q` → exit `1` during collection with the expected missing
  `_write_results_json` import (`1` collection error in `0.57s`).
- Persisted-history GREEN, identical command → exit `0`,
  `1 passed in 0.64s`. The same private helper used by the real final
  `results.json` path retains all existing summary keys, adds scalar
  iteration `history`, omits the large/non-serializable `video` tensor, and
  round-trips `sigma_used` values `0.3` and `0.05`.
- Contract-coverage first run for a real fifth proximal call, diffusion
  warmup, and a non-diffusion object → exit `1`,
  `2 failed, 1 passed in 0.66s`. The fifth call already failed closed with
  `IndexError` without advancing `_call_count`; warmup added a misleading
  `sigma_used: None`, and a non-diffusion object raised `AttributeError`.
  The minimal conditional public-property read made the identical three-test
  command GREEN: `3 passed in 0.53s`.

### Fix Round 1 verification

- Focused Task 4 suite → exit `0`, `30 passed in 0.67s`.
- Accumulated CPU suite → exit `0`,
  `114 passed, 1 deselected in 15.18s`.
- Real CUDA suite → exit `0`,
  `1 passed, 114 deselected in 1.04s`; exactly one CUDA test executed and
  zero skipped.
- Full suite → exit `0`, `115 passed in 22.25s`.
- Strict environment verification → exit `0`, fingerprint
  `b5d6922a9f3a9638ee8826b9a74f00998cd3ac81aa25c03de016358e0e435a56`.
- Strict implementation-provenance verification → exit `0`; four immutable
  inputs verified at the Fix Round 1 base
  `03818b50c4a750fd974bd46c41579cdcea9bb0cb`.
- Fresh targeted JSON persistence test → exit `0`,
  `1 passed in 0.65s`.
- Targeted tracked-Python search found `_current_sigma` only in its
  definition, `DiffusionPrior.proximal`, and tests; `train.py` contains no
  private schedule call and no misleading `sigma` history field.
- `D:\conda\envs\spi\python.exe -m pip check` → exit `0`,
  `No broken requirements found.`
- `git diff --check` → exit `0` with silent output. All test and verifier
  output was pristine, with no warnings or unexpected skips.

## Task 5 — Isolate solver parameter groups (2026-07-27)

- Prior approved commit and clean starting HEAD:
  `08e0e629d8ce6075f246a5ec405ad87d06078f1b` on
  `debug/admm-vs-sgd`.
- Constructor/caller inspection found no solver optimizer/scheduler
  `state_dict` persistence or documented parameter-group index contract.
  Training checkpoints save scene and motion module state separately. Active
  optimizer groups therefore retain stable scene-then-motion order while
  frozen or empty logical groups are omitted.
- The differentiable CPU fixture uses one float64 scene scalar and one float64
  motion scalar, exposes `.scene`, `.motion`, inherited `.parameters()`,
  `.H`, `.W`, `render_video`, and the real solver forward signature. Its
  prediction and video depend on both scalars, and changing the dormant motion
  derivative scale from `1` to `1e6` leaves the initial forward value
  unchanged.

### Three-phase TDD evidence

- Phase A frozen-parameter RED:
  `D:\conda\envs\spi\python.exe -m pytest
  tests\solver\test_sgd_parameters.py -q` → exit `1`,
  `2 failed in 1.34s`. Frozen motion still had `requires_grad=True`, and the
  scale-`1` versus scale-`1e6` scene updates had different bytes
  (`185fecab749368bf` versus `cbcdc2f3637a68bf`) because the inactive motion
  gradient entered global clipping.
- The minimal Phase A implementation added `freeze_parameters` and
  `active_parameters`, stored explicit scene/motion parameter lists, froze and
  cleared motion before optimizer construction, and built SGD groups only
  from active parameters. Identical focused GREEN → exit `0`,
  `2 passed in 1.27s`. The final one-step scene value was
  `-0.002999999880000003` with identical bytes `185fecab749368bf` for both
  dormant motion derivative scales; motion had `requires_grad=False` and
  `grad=None`.
- Phase B group-clipping RED:
  `D:\conda\envs\spi\python.exe -m pytest
  tests\solver\test_sgd_parameters.py
  tests\solver\test_gradient_groups.py -q` → exit `1`,
  `4 failed, 5 passed in 1.20s`. `clip_grad_groups` was absent; both real SGD
  and ADMM update paths reduced the invariant scene gradient from
  `0.2499999964644661` to `2.4999999999946876e-06` when only the motion
  derivative changed from `1` to `1e6`; ADMM also retained an empty motion
  optimizer group.
- The minimal Phase B implementation clips each nonempty active logical group
  independently and routes both solvers through stored
  `[scene, motion]` lists. ADMM now also constructs only nonempty active
  optimizer groups. Focused GREEN → exit `0`, `9 passed in 1.09s`.
  The direct helper returned pre-clip norms `[120.0, 5.0]` in original
  nonempty-group order, clipped motion `[0,120]` to `[0,10]`, left scene
  `[3,4]` unchanged, and ignored empty/frozen groups. Real SGD and ADMM both
  retained scene gradient `0.2499999964644661`; the scale-`1e6` motion
  gradient clipped independently to `-4.99999999999`. The SGD motion-warmup
  regression also retained an exactly zero scene gradient and byte-identical
  scene parameter.
- Phase C LR-floor RED, before scheduler production changes: the same focused
  two-file command → exit `1`, `11 failed, 9 passed in 1.26s`.
  `cosine_multiplier` was absent. Both solvers used pre-step LR states
  `[[0.009,0.15], [0.007813782463805517,0.1281648105374571],
  [0.0049499999999999995,0.07544999999999998],
  [0.0020862175361944825,0.022735189462542882]]` and ended after four
  declared steps at the shared absolute floor `[0.0009,0.0009]`.
- The minimal Phase C implementation added a capped, validated multiplicative
  cosine helper and one shared `LambdaLR` function per optimizer. Focused
  GREEN → exit `0`, `20 passed in 1.14s`. Both SGD (`4` steps) and ADMM
  (`2 outer * 2 inner`) use multiplier states
  `[1.0, 0.8681980515339464, 0.55, 0.23180194846605365, 0.1]`
  and LR states
  `[[0.009,0.15], [0.007813782463805517,0.13022970773009196],
  [0.00495,0.0825], [0.0020862175361944825,0.03477029226990805],
  [0.0009,0.015]]`. Optimizer step one uses the base LR; optimizer step four
  uses the preceding scheduled LR, and its immediately following scheduler
  call sets the exact per-group 10% floors with `last_epoch == 4`. No extra
  optimizer or scheduler step is required.

### Verification

- Focused Task 5 suite → exit `0`, `20 passed in 1.14s`.
- Accumulated CPU suite:
  `D:\conda\envs\spi\python.exe -m pytest -m "not cuda" -q` → exit `0`,
  `134 passed, 1 deselected in 18.31s`.
- Real CUDA suite:
  `D:\conda\envs\spi\python.exe -m pytest -m cuda -q` → exit `0`,
  `1 passed, 134 deselected in 0.93s`; exactly one CUDA test executed and
  zero skipped.
- Full suite:
  `D:\conda\envs\spi\python.exe -m pytest -q` → exit `0`,
  `135 passed in 14.26s`.
- Strict environment verification → exit `0`, fingerprint
  `b5d6922a9f3a9638ee8826b9a74f00998cd3ac81aa25c03de016358e0e435a56`.
- Strict implementation-provenance verification → exit `0`; four immutable
  inputs verified at the pre-Task commit
  `08e0e629d8ce6075f246a5ec405ad87d06078f1b`.
- `D:\conda\envs\spi\python.exe -m pip check` → exit `0`,
  `No broken requirements found.`
- A targeted solver search found no remaining flattened/global clipping call
  outside the single `clip_grad_groups` implementation and no
  `CosineAnnealingLR` or `eta_min` absolute-floor path. `git diff --check` →
  exit `0` with silent output. All test and verifier output was pristine,
  with no warnings or unexpected skips.

## Task 5 Fix Round 1 — Ignore parameters without gradients (2026-07-27)

- Fix base and clean starting HEAD:
  `dbce67af00aea1023ef4dc3cd24ecd7fddcd9415`; the original Task 5 commit was
  not amended.
- Root cause: `clip_grad_groups` filtered only on `requires_grad`. An active
  logical group whose parameters all had `grad=None` therefore called
  `clip_grad_norm_` unnecessarily and inserted a spurious zero tensor into
  the returned norm sequence.
- Direct-helper RED:
  `D:\conda\envs\spi\python.exe -m pytest
  tests\solver\test_gradient_groups.py::test_clip_grad_groups_ignores_no_grad_frozen_and_empty_groups_in_order
  -q` → exit `1`, `1 failed in 0.06s`. For logical groups ordered as
  active-gradient motion, active no-gradient, empty, frozen, and
  active-gradient scene, the helper returned `[120.0, 0.0, 5.0]` instead of
  `[120.0, 5.0]`.
- The minimal fix builds each clipping list from parameters satisfying both
  `requires_grad` and `grad is not None`. The identical isolated command →
  exit `0`, `1 passed in 0.03s`. The active no-gradient parameter remained
  unchanged with `grad=None`; only the two real-gradient norms were returned
  in original logical-group order.
- Focused Task 5 suite → exit `0`, `20 passed in 1.11s`.
- Accumulated CPU suite → exit `0`,
  `134 passed, 1 deselected in 13.69s`.
- Real CUDA suite → exit `0`,
  `1 passed, 134 deselected in 0.90s`; exactly one CUDA test executed and
  zero skipped.
- Full suite → exit `0`, `135 passed in 14.74s`.
- Strict environment verification → exit `0`, fingerprint
  `b5d6922a9f3a9638ee8826b9a74f00998cd3ac81aa25c03de016358e0e435a56`.
- Strict implementation-provenance verification → exit `0`; four immutable
  inputs verified at Fix Round 1 base
  `dbce67af00aea1023ef4dc3cd24ecd7fddcd9415`.
- `D:\conda\envs\spi\python.exe -m pip check` → exit `0`,
  `No broken requirements found.`
- Targeted clipping search found the sole `clip_grad_norm_` call inside
  `clip_grad_groups` and the two solver call sites. `git diff` for
  `sgd.py` and `admm.py` was empty, confirming no solver-flow or scheduler
  edit. `git diff --check` → exit `0` with silent output. All output was
  pristine, with no warnings or unexpected skips.

## Task 6 — Align rectangular INR geometry and field-of-view boundaries (2026-07-27)

- Prior approved commit and clean starting HEAD:
  `38d49d639ee2961e7efe4b272c4f2a43afd1c85a` on
  `debug/admm-vs-sgd`. Inspection confirmed that `recinr_se2` uses the
  repository's vendored `gsdiff.baselines.recinr_model`; no external
  `D:\SPI\ReCINR`, checkpoint, network, or process dependency was used.
- Coordinate contract: coordinates and centers are ordered `(y,x)`, shapes are
  `(H,W)`, and the `32x64` image center is `(15.5,31.5)`. The per-axis
  denominators are therefore `(15.5,31.5)`, so `(0,0)` maps exactly to
  `(-1,-1)` and `(31,63)` maps exactly to `(1,1)`. The helper constructs the
  denominator through `coordinates.new_tensor`, preserving coordinate dtype,
  device, and autograd.
- Phase 1 corner-helper RED:
  `D:\conda\envs\spi\python.exe -m pytest
  tests\scene\test_rectangular_geometry.py -q` → exit `1` during collection
  with `ImportError: cannot import name 'normalize_pixel_coordinates'`.
  The normalization-only implementation added the per-axis helper and routed
  both `INRForwardModel.render_video` and `norm_grid` through it. The identical
  focused command → exit `0`, `1 passed in 0.27s`.
- Phase 2 physical-boundary RED, before masking: the focused command → exit
  `1`, `4 failed, 1 passed in 0.35s`. A constant-one scene translated by
  `(0,128)` returned `2048/2048` ones instead of zeros. Translation `(0,63)`
  returned ones everywhere rather than only the 32 exact-boundary samples in
  the last raster column. The real SIREN extrapolated to nonzero values as
  large as `0.7172287702560425`, and the grid border path returned `0.5` at
  every sample.
- The first minimal mask run exposed the intended ordering check: multiplying
  queried `[T,H,W]` values by the still-flat `[T,H*W]` predicate failed with a
  `64` versus `2048` dimension mismatch. Reshaping the predicate in the same
  row-major order as the flattened `(y,x)` pixel grid and scene query fixed
  the mismatch. The shared renderer now queries the canonical scene and then
  multiplies by `(abs(x_norm) <= 1).all(-1)`; direct canonical-query
  `padding_mode="border"` defaults were not changed. Phase 2 GREEN → exit `0`,
  `5 passed in 0.33s`. The full-outside case has `0/2048` inside samples; the
  `(0,63)` case has exactly `32/2048`, proving normalized `x=-1` is included.
- Phase 3 rectangular-construction RED, before grid/low-rank/ReCINR production
  changes: the focused command → exit `1`, `6 failed, 17 passed in 0.57s`.
  `GridCanonical` lacked `Hc/Wc`, `build_scene("grid")` remained `64x64`, and
  prefit inferred `sqrt(2048)=45` before an invalid reshape. ReCINR lacked
  `gw`, scalar `grid_size=16` remained `16x16`, explicit `(12,20)` failed
  construction, zero silently fell back to `H`, and negative sizes reached
  tensor allocation instead of validation.
- The minimal construction implementation stores grid dimensions, validates
  prefit target numel exactly before reshaping to `(Hc,Wc)`, falls back from
  absent `Hc/Wc` to requested `H/W`, and passes requested `H/W` into low-rank.
  ReCINR stores `(gh,gw)`, validates positive scalar or explicit tuple sizes,
  and uses both dimensions in feature allocation and interpolation checks. A
  scalar is a short-side resolution: for `H=32,W=64,grid_size=16`, scale is
  `16/min(32,64)=0.5`, hence `(gh,gw)=(16,32)`; `grid_size=8` similarly gives
  `(8,16)`. Phase 3 focused GREEN → exit `0`, `23 passed in 0.46s`.
- Real `build_scene` plus `INRForwardModel` rectangular tests exercised SIREN
  (`hidden=8`), grid (`grid=[1,1,32,64]`, `Hc/Wc=32/64`), low-rank
  (`U=[32,4]`, `V=[64,4]`), and ReCINR SE(2)
  (`features=[1,4,8,16]`). Each produced finite `[3,1,32,64]` videos and
  present finite gradients for every trainable scene and motion parameter.
  Under the all-outside translation, all four produced exact-zero
  `[1,1,32,64]` outputs with present finite masked-output gradients. The
  Gaussian forward renderer also produced finite exact-zero
  `[1,1,32,64]` output under the controlled all-outside fixture and finite
  gradients for every trainable parameter.

### Verification

- Focused rectangular suite → exit `0`, `23 passed in 0.46s`.
- Accumulated CPU suite:
  `D:\conda\envs\spi\python.exe -m pytest -m "not cuda" -q` → exit `0`,
  `157 passed, 1 deselected in 16.85s`.
- Real CUDA suite:
  `D:\conda\envs\spi\python.exe -m pytest -m cuda -q` → exit `0`,
  `1 passed, 157 deselected in 1.21s`; exactly one CUDA test executed and
  zero skipped.
- Full suite:
  `D:\conda\envs\spi\python.exe -m pytest -q` → exit `0`,
  `158 passed in 16.29s`.
- Strict environment verification → exit `0`, fingerprint
  `b5d6922a9f3a9638ee8826b9a74f00998cd3ac81aa25c03de016358e0e435a56`.
- Strict implementation-provenance verification → exit `0`; four immutable
  inputs verified at prior commit
  `38d49d639ee2961e7efe4b272c4f2a43afd1c85a`.
- `D:\conda\envs\spi\python.exe -m pip check` → exit `0`,
  `No broken requirements found.`
- Targeted square-assumption and FOV searches found no remaining
  max-dimension/scalar normalization, square `H=W`, `gh,gh`, or single-`gh`
  interpolation path in the INR/ReCINR canonical implementation. The only
  remaining `padding_mode="border"` paths are direct canonical queries; the
  physical zero boundary is applied once in the shared forward renderer.
  `git diff --check` → exit `0` with silent output. All verifier and test
  output was pristine, with no warnings or unexpected skips.

## Task 6 Fix Round 1 — Validate rectangular prefit shapes (2026-07-27)

- Fix base and clean starting HEAD:
  `345571bc8a269749f23fd74cbdde0df1152285e5`; the original Task 6 commit was
  not amended.
- Call-site inspection found one production prefit path:
  `train.py` passes `normalize_01(dgi_img)`, where `dgi_reconstruct` documents
  and returns exact `[H,W]`. The explicit compatibility contract accepts only
  `(Hc,Wc)`, flat `(Hc*Wc,)`, `(1,Hc,Wc)`, and `(1,1,Hc,Wc)`. It rejects
  transposed or arbitrary same-numel shapes and all non-singleton leading
  batch/channel dimensions.
- Shape RED, before production validation changes:
  `D:\conda\envs\spi\python.exe -m pytest
  tests\scene\test_rectangular_geometry.py -q` → exit `1`,
  `2 failed, 32 passed in 0.46s`. Same-numel targets shaped `(64,32)` and
  `(2048,1)` were silently accepted and reinterpreted as `(32,64)`. The four
  supported compatibility shapes already passed; `(2,32,64)` and
  `(1,2,32,64)` were already rejected by size alone.
- The minimal fix replaces numel-only validation with exact tuple-shape
  membership before the existing reshape. Errors enumerate the supported
  shapes and report the received shape. Focused GREEN → exit `0`,
  `34 passed in 0.39s`. The accepted and rejected matrices above all exercise
  the real `GridCanonical.prefit`.
- ReCINR scalar aspect regressions cover portrait `64x32 -> (32,16)`,
  non-integral landscape `30x47 -> (16,25)`, and non-integral portrait
  `47x30 -> (25,16)` for `grid_size=16`. Each keeps the short side exactly
  16. The class documentation now states the implementation's deterministic
  Python round-to-nearest, ties-to-even rule for the scaled other dimension.
  Existing explicit tuple behavior remains covered and unchanged.

### Verification

- Focused rectangular suite → exit `0`, `34 passed in 0.39s`.
- Accumulated CPU suite → exit `0`,
  `168 passed, 1 deselected in 14.64s`.
- Real CUDA suite → exit `0`,
  `1 passed, 168 deselected in 1.13s`; exactly one CUDA test executed and
  zero skipped.
- Full suite → exit `0`, `169 passed in 15.13s`.
- Strict environment verification → exit `0`, fingerprint
  `b5d6922a9f3a9638ee8826b9a74f00998cd3ac81aa25c03de016358e0e435a56`.
- Strict implementation-provenance verification → exit `0`; four immutable
  inputs verified at Fix Round 1 base
  `345571bc8a269749f23fd74cbdde0df1152285e5`.
- `D:\conda\envs\spi\python.exe -m pip check` → exit `0`,
  `No broken requirements found.`
- Targeted shape/aspect audit reported
  `numel_only_or_sqrt_prefit_validation=0`, found the exact shape check before
  the single `(Hc,Wc)` reshape, and found the documented short-side scale plus
  deterministic `round` path. `git diff --check` → exit `0` with silent
  output. All test and verifier output was pristine, with no warnings or
  unexpected skips.

## Task 7 — Lock forward, gradient, and data reproducibility contracts (2026-07-27)

- Prior approved commit and clean starting HEAD:
  `f9216bfbda54615c909a300239e237c45d2780ac` on
  `debug/admm-vs-sgd`.
- API/reference inspection confirmed identical NumPy/PyTorch interpolation:
  `u=k/max(K-1,1)*(T-1)`, `m=floor(u)`, `alpha=u-m`, with the last boundary
  represented by `m=T-2, alpha=1`. No time-semantics choice was required.
- Initial focused RED, after all three test files existed and before production
  edits: `4 failed, 6 passed in 0.43s`. All data tests and the independent
  center/translation/angular gradient tests already passed and were locked
  without simulation, pattern, scene, or motion edits.
- Direct-measurement root/backward trace:
  `forward -> measure -> y[mask] = P @ fr`. Float64 operands produced a
  float64 matmul result, but `torch.empty(K)` created a float32 destination.
  The minimal fix is `video.new_empty(K)`. The direct and end-to-end rerun
  passed: `2 passed in 0.10s`.
- Interpolation-shape root/backward trace: two per-pattern dot vectors had
  shape `[K]`, while `alpha.unsqueeze(1)` had shape `[K,1]`; broadcasting
  produced `[K,K]` (`[1,1]` for `K=1`) instead of `[K]`. Retaining `alpha`
  as `[K]` restores the declared elementwise equation.
- After that correction, interpolation dtype RED was
  `1 failed, 1 passed in 0.07s`: four of seven fractional-time measurements
  differed from float64 NumPy, with maximum absolute error
  `1.48018202e-07` and relative error `5.00679017e-06`. Backward trace:
  `float32 arange -> u -> alpha_raw -> y`. The minimal fix inherits
  `video.dtype` for time indices and `u.dtype` for integer-to-time
  subtraction.
- The dtype-literal audit then identified concrete Step 5 violations despite
  output promotion. Four direct creation-instrumentation tests RED:
  `4 failed, 4 deselected in 0.15s`. Gaussian render requested float32 grids
  and a dtype-less precision identity; forward render requested a dtype-less
  identity and float32 grids on each frame; SE2 no-rotation failed an actual
  Float-versus-Double einsum; affine transport used a dtype-less identity.
  Minimal fixes make Gaussian grids/eye inherit
  `self.centers.dtype`/`Sigma.dtype`, forward grids/eye inherit
  `centers_t.dtype`/`Sigma_t.dtype`, and both SE2 identities inherit
  `t.dtype`, with devices unchanged. Dtype-focused GREEN:
  `4 passed, 4 deselected in 0.12s`.
- Final focused GREEN:
  `D:\conda\envs\spi\python.exe -m pytest
  tests\forward\test_measurement_consistency.py
  tests\scene\test_numerical_gradients.py
  tests\data\test_reproducibility.py -q` → exit `0`,
  `14 passed in 0.33s`.
- Direct float64 measurements were
  `[-0.425, 2.175, -2.1, -1.025, -2.275]`. Interpolated `K=7` measurements
  matched NumPy to `2.220446049250313e-16`; `K=1` returned the first-frame
  inner product and the final boundary returned the final-frame inner
  product. The direct fixture assigns no measurement to frame one.
- Central differences used `h=1e-6`. Autograd versus finite difference
  `(relative error)` was: center `0.3900542265806254` versus
  `0.39005422669546874` (`2.944292080086366e-10`); translation velocity
  `1.048` versus `1.0479999996704237` (`3.144812654562266e-10`); angular
  velocity `-0.7193208014691927` versus `-0.7193208011457841`
  (`4.496027816053279e-10`). End-to-end measurement-loss values were center
  `-1.314375592723244` versus `-1.3143755928091139`
  (`6.533138492894905e-11`), velocity `-0.8315084566939432` versus
  `-0.8315084567556141` (`7.416748454623187e-11`), and omega
  `1.959294777810261` versus `1.9592947770874503`
  (`3.6891370730341706e-10`). All were finite and meaningfully nonzero; the
  documented near-zero absolute fallback was not used.
- Seed-7 repeated generation was byte-equal for all nine explicit public
  arrays. Seed 7 with holdout zero versus 16 was byte-equal for all six
  training arrays. Seed 11 preserved deterministic target/frames/indices/time
  and shapes/dtypes while patterns, noisy measurements, holdout
  patterns/measurements, and inferred noise differed. Inferred-noise SHA-256:
  seed 7
  `63e7e31cbeb3b4309444b5a83cb057b625e0f371109dd44c1bf071356241d470`;
  seed 11
  `2111c58d54fe78ad18d3f9221e62194d4d6547db4bc5847110001678a3c61f7e`.
  Simulation and pattern RNG code was not modified.

### Verification

- Accumulated CPU suite → exit `0`,
  `182 passed, 1 deselected in 21.52s`.
- Real CUDA suite → exit `0`,
  `1 passed, 182 deselected in 1.19s`; exactly one executed and zero skipped.
- Full suite → exit `0`, `183 passed in 16.79s`.
- Strict environment verification → exit `0`, fingerprint
  `b5d6922a9f3a9638ee8826b9a74f00998cd3ac81aa25c03de016358e0e435a56`.
- Strict implementation-provenance verification → exit `0`; four immutable
  inputs verified at prior commit
  `f9216bfbda54615c909a300239e237c45d2780ac`.
- `D:\conda\envs\spi\python.exe -m pip check` → exit `0`,
  `No broken requirements found.`
- Targeted forward dtype audit reported
  `hardcoded_forward_creation_violations=0`; every grid, identity,
  interpolation index, and measurement buffer now inherits participating
  dtype/device. `SE2Motion.center` also follows `.double()`.
- `git diff --check` → exit `0` with silent output. All test and verifier
  output was pristine, with no warnings or unexpected skips.

## Task 7 Fix Round 1 — Strengthen forward contract coverage (2026-07-27)

- Clean fix base:
  `885bb3213fb7df32e7887a1808a88ca636c520de`; the original Task 7 commit
  was not amended.
- This round closed coverage gaps only. All requested assertions were added
  before rerunning tests, and current fixed production passed immediately:
  focused Task 7 → exit `0`, `17 passed in 1.56s`. This is recorded as honest
  coverage-only GREEN; no production file was modified and no RED was
  fabricated.
- Measurement shape is now asserted before every numerical comparison:
  direct `K=5` with frame one unassigned has exact shape `(5,)`;
  interpolated `K=7`, including the exact final-frame boundary, has `(7,)`;
  and `K=1` has `(1,)`. A broadcastable `[1,1]` regression can therefore no
  longer satisfy the NumPy comparison or `.item()` assertion.
- The registered `SE2Motion.center` buffer is asserted float64 after
  `.double()` and on the same device as both `velocity` and `omega`.
  Portable non-CPU tests move the real module and participating inputs to the
  PyTorch `meta` device. No-rotation `transform_centers` returns meta
  `[3,1,2]` float64, and affine `transform_covariances` returns meta
  `[3,1,2,2]` float64. These real operations lock device inheritance for both
  identity constructors without changing the CUDA marker count.
- Reference inspection confirmed `eval_frame_idx[j] =
  clip((j*T)//Ke, 0, T-1)`, independent of seed. The seed-sensitivity test now
  explicitly asserts seed-7 and seed-11 `eval_frame_idx` equality in addition
  to existing shape/dtype checks.

### Verification

- Focused Task 7 suite → exit `0`, `17 passed in 1.56s`.
- Accumulated CPU suite → exit `0`,
  `185 passed, 1 deselected in 14.28s`.
- Real CUDA suite → exit `0`,
  `1 passed, 185 deselected in 1.17s`; exactly one executed and zero skipped.
- Full suite → exit `0`, `186 passed in 15.53s`.
- Strict environment verification → exit `0`, fingerprint
  `b5d6922a9f3a9638ee8826b9a74f00998cd3ac81aa25c03de016358e0e435a56`.
- Strict implementation-provenance verification → exit `0`; four immutable
  inputs verified at the clean fix base
  `885bb3213fb7df32e7887a1808a88ca636c520de`.
- `D:\conda\envs\spi\python.exe -m pip check` → exit `0`,
  `No broken requirements found.`
- `git diff --check` → exit `0` with silent output. The diff contained only
  the three intended Task 7 test files plus this append-only ledger entry.
  Output was pristine, with no warnings or unexpected skips.

## Task 8 — Add `metrics-v1` and explicit legacy metrics (2026-07-27)

- Clean approved base:
  `2c8b34329b6f3759b91d8b452a3e9c49c6e4a6b7` on
  `debug/admm-vs-sgd`; tracked and untracked status was empty before work.
- Legacy inventory before implementation found one deliberately preserved
  formula. `train.evaluate` and `gsdiff.baselines.common.evaluate_video` both
  clipped each reconstruction frame below at zero, independently min-max
  normalized reconstruction and ground truth with a constant-range threshold
  of `1e-8`, and passed the pair to `gsdiff.utils.psnr_fn`, whose
  `MSE < 1e-12` branch returned the historical 60-dB sentinel. DGI instead
  independently min-max normalized one DGI image and the canonical image and
  used the same 60-dB helper. These formulas were preserved under explicit
  legacy names rather than selected as primary metrics.
- The plan's opening file list and staging example omitted the sole real
  `baselines.json` writer, `scripts/run_baselines.py`, while the binding task
  requirement mandated a root legacy definition label for that file. The
  scoped resolution was to modify only that writer's serialization: no
  baseline algorithm, execution, method-child behavior, or unrelated field
  changed.

### RED and GREEN evidence

- Known-affine import RED was captured while `gsdiff.evaluation` did not
  exist:
  `D:\conda\envs\spi\python.exe -m pytest
  tests\evaluation\test_metrics.py -q` → exit `1`,
  `ModuleNotFoundError: No module named 'gsdiff.evaluation'`, one collection
  error in `0.10s`.
- API-only `NotImplementedError` placeholders then allowed all behavior and
  input contracts to be written before implementation. Edge-contract RED →
  exit `1`, `20 failed in 0.19s`; every failure reached the intentional
  placeholder. Coverage included known-affine recovery, differing frame-gain
  cheating, constant predictions, negative correlation, explicit negative
  slope opt-out, clipped float64 application, metadata, non-array/non-numeric/
  mismatched/nonfinite/malformed inputs, and the SSIM spatial boundary.
- Minimal pure-metric GREEN → exit `0`, `20 passed in 0.29s`.
- Serialization integration tests were then added before integration.
  Integration RED → exit `1`, `22 passed, 2 failed in 0.91s`; the failures
  were the intentionally absent `_write_metrics_json` and
  `_write_baselines_json`.
- Final focused GREEN:
  `D:\conda\envs\spi\python.exe -m pytest
  tests\evaluation\test_metrics.py -q` → exit `0`,
  `24 passed in 0.89s`. A later fresh focused run after annotations was
  `24 passed in 1.39s`. The existing results/history writer regression suite
  also remained GREEN: `30 passed in 0.70s`.
- Pre-staging self-review found that zero-length spatial axes passed the
  general 3-D validation and reached inconsistent downstream behavior:
  `fit_global_affine` returned `(0,0)` with NumPy empty-reduction warnings,
  `apply_global_affine` returned an empty array, and the two evaluators raised
  different downstream errors. The focused contract RED was
  `2 failed, 24 deselected, 10 warnings in 0.34s`. The root cause was the
  validator checking non-empty `T` but not non-empty `H,W`; the minimal shared
  validation added an explicit positive-spatial-dimension error. Final
  focused GREEN → exit `0`, `26 passed in 1.12s`, with no warnings.

### Definitions and integration

- `fit_global_affine` solves one float64 least-squares system
  `[recon_flat, ones] @ [slope, intercept] ~= gt_flat` over the complete
  video. The default nonnegative policy selects
  `slope=0, intercept=mean(gt)` when the unconstrained slope is negative or
  when
  `variance(recon) <= eps64 * max(1, mean(recon**2))`.
- `apply_global_affine` returns the one float64
  `clip(slope * recon + intercept, 0, 1)` array consumed unchanged by all
  primary image metrics. Primary PSNR is
  `-10 log10(max(MSE, 1e-12))`, capped numerically at 120 dB; SSIM is the
  mean of per-frame `structural_similarity` values with `data_range=1` and
  fixed `win_size=7`, requiring `H,W >= 7`; nRMSE is
  `||aligned-gt||_2 / max(||gt||_2, eps64)`.
- Known-affine recovery measured slope `1.6999999999999988`, intercept
  `0.20000000000000026`, PSNR `120.0` dB, and nRMSE
  `3.407460242541517e-16`.
- The two-frame differing-gain fixture measured global slope
  `0.33189361218458746`, intercept `0.292566492384633`, primary PSNR
  `12.998621540547393` dB, and legacy PSNR `60.0` dB: a
  `47.00137845945261`-dB cheating gap. Constant prediction and negative
  correlation both selected the boundary `(0.0, 0.5)` and returned finite
  metrics; opting out of the nonnegative constraint recovered slope `-1` and
  intercept `1`.
- `metrics.json` is written once after final reconstruction from the final
  `[T,H,W]` reconstruction and ground truth. Its root contains
  `definition_version="metrics-v1"`,
  `psnr_global_affine`, `ssim_global_affine`,
  `nrmse_global_affine_l2`, labelled
  `psnr_legacy_per_frame_minmax`, `alignment`, and JSON-native
  `metric_definition` metadata including the variance, clipping, PSNR, SSIM,
  and nRMSE policies.
- Compatibility `results.json` preserves `mean_psnr`, `per_frame_psnr`,
  `dgi_psnr`, history, and all existing fields, while declaring root
  `metric_definition_version="legacy-per-frame-minmax-v1"`. Its aliases are
  `mean_psnr_legacy_per_frame_minmax`,
  `per_frame_psnr_legacy_per_frame_minmax`, and the deliberately distinct
  `dgi_psnr_legacy_canonical_minmax_60db`. It contains no primary global
  affine key.
- `baselines.json` declares the same root legacy definition. Every method row
  preserves `mean_psnr` and `per_frame_psnr` and adds
  `mean_psnr_legacy_per_frame_minmax` and
  `per_frame_psnr_legacy_per_frame_minmax`. Only the common compatibility
  adapter calls `evaluate_video_legacy_per_frame`; no baseline method child
  directly imports or calls it.

### Verification

- Fresh accumulated CPU suite after the self-review fix → exit `0`,
  `211 passed, 1 deselected in 13.33s`.
- Real CUDA suite → exit `0`,
  `1 passed, 211 deselected in 1.19s`; exactly one executed and zero skipped.
- Full suite → exit `0`, `212 passed in 13.72s`.
- Strict environment verification → exit `0`, fingerprint
  `b5d6922a9f3a9638ee8826b9a74f00998cd3ac81aa25c03de016358e0e435a56`.
- Strict implementation-provenance verification → exit `0`; four immutable
  inputs verified at the approved base
  `2c8b34329b6f3759b91d8b452a3e9c49c6e4a6b7`.
- `D:\conda\envs\spi\python.exe -m pip check` → exit `0`,
  `No broken requirements found.`
- Independent formula audit matched the direct least-squares, clipped-array,
  PSNR, SSIM, and nRMSE calculations exactly. The audit fixture measured
  slope `0.52233394585058834`, intercept `0.15599013272581511`, PSNR
  `22.636881044371059`, SSIM `0.71673241888650085`, and nRMSE
  `0.13080952673187204`.
- Targeted key/boundary audit reported one and only one train metrics-writer
  call, zero method-child references to the explicit legacy evaluator, strict
  separation of `metrics.json` from `results.json`/`baselines.json`, and all
  required root labels and aliases.
- `git diff --check` → exit `0` with silent output. Test and verifier output
  was pristine, with no warnings or unexpected skips.
