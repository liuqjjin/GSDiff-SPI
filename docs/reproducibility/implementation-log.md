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
