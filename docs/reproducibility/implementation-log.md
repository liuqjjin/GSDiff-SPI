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
