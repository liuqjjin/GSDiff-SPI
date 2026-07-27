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
