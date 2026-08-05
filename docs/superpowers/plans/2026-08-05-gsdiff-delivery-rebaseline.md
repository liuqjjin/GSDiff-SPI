# GSDiff-SPI Research Delivery Rebaseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans` or `superpowers:subagent-driven-development`
> task by task. Use TDD only for required behavior changes; verification-only
> closure tasks do not invent new RED cases.

**Goal:** Freeze the working experiment runner, return the project to the
shortest reproducible path toward locked numerical results, and deliver a
venue-neutral paper package without further security-engineering scope growth.

**Architecture:** Keep the existing scientific code and the already-tested
Task 1 control plane. Add only the minimum phase-aware aggregation layer needed
to separate selection from confirmation, reject incomplete evidence, compute
paired statistics, and lock publication inputs. Treat results-lock data as the
sole numerical source for the paper.

**Tech Stack:** Python 3.12.13, PyTorch 2.8.0+cu128, CUDA 12.8, pytest,
NumPy/SciPy, JSON Schema, Git, PowerShell, Python-only publication figures,
and a project-local TeX/PDF toolchain.

## Global Constraints

- Work only in `D:\Research\gsdiff_spi` on `debug/admm-vs-sgd`.
- Use only `D:\conda\envs\gsdiff-spi\python.exe` for project Python commands.
- Never access, enumerate, read, execute, hash, or modify `D:\SPI\ReCINR`.
- Never modify or remove `D:\conda\envs\spi`.
- Do not change reconstruction algorithms, scientific protocols, or paper
  claims while closing Task 1 or implementing Task 2.
- Do not claim mathematical/global optimality; use only
  `grid-optimal under protocol v1` where supported.
- Push reviewed commits non-force. Never force-add `.superpowers/sdd`.
- Raw datasets, checkpoints, per-run logs, and large transient arrays remain
  outside Git; compact evidence, schemas, code, and paper sources are tracked.

## Operational Boundary

The publication system must resist ordinary research failures:

- interrupted or failed child processes;
- two legitimate requests for the same identity;
- stale claims and partial/corrupt cache directories;
- pre-existing links/reparse points, malformed manifests, and wrong hashes;
- dirty Git state, wrong environment/dataset/checkpoint/method identity;
- truth leakage into blind method children;
- partial, duplicate, mixed, nonfinite, or unblinded aggregation inputs.

The following are non-blocking hardening backlog, not protocol-v1 completion
criteria:

- a malicious process running as the same Windows account and racing every
  filesystem syscall;
- native NT handle-relative tree writers or exhaustive path-swap fault
  injection;
- an actor rewriting all local files, Git history, remote refs, and GitHub
  credentials;
- a second same-identity artifact root beyond the required 207-run replay;
- cryptographic receipt chains or remote evidence refs before results lock.

No implementation or review may broaden this boundary without a new explicit
user decision.

---

### Task 1: Freeze and Close the Existing Atomic Runner

**Files:**

- Preserve the current Task 1 production/test scope under
  `gsdiff/experiments/`, `scripts/experiments/`, the three compatibility
  wrappers, and `tests/experiments/`.
- Update only the local Task 1 report and progress ledger after fresh evidence.
- Do not add a native Windows writer, another claim protocol, or another
  source-provenance subsystem.

**Produces:** The explicit public API `RunExecutionPlan`, `RunRequest`,
`RunOutcome`, `run_request`, and `reusable_run`, plus the versioned campaign
CLI and compatibility wrappers. The original four-field `RunRequest` contract
is narrowly corrected here: execution evidence is a required immutable request
field because device, source/checkpoint locators, resolved configuration, and
disk policy cannot be recovered uniquely from identity hashes. The two-argument
`run_request(request, artifact_root)` signature remains unchanged and no ambient
execution context is permitted.

- [x] **Step 1: Verify the most recently touched provenance files**

```powershell
D:\conda\envs\gsdiff-spi\python.exe -m pytest `
  tests\experiments\test_identity.py `
  tests\experiments\test_source_snapshot.py -q
```

Expected: all runnable tests pass; only platform-capability skips are allowed.

- [x] **Step 2: Run one uninterrupted runner suite**

```powershell
D:\conda\envs\gsdiff-spi\python.exe -m pytest `
  tests\experiments\test_runner.py -q
```

Expected: all runner scenarios pass in one process invocation.

- [x] **Step 3: Run the remaining focused Task 1 integration suite**

```powershell
D:\conda\envs\gsdiff-spi\python.exe -m pytest `
  tests\experiments\test_campaign_cli.py `
  tests\experiments\test_manifest.py `
  tests\experiments\test_source_snapshot.py `
  tests\experiments\test_child_outputs.py `
  tests\experiments\test_baseline_adapters.py `
  tests\experiments\test_gsdiff_adapter.py `
  tests\experiments\test_method_execution.py -q
```

- [x] **Step 4: Run global CPU, CUDA, and static gates**

```powershell
D:\conda\envs\gsdiff-spi\python.exe -m pytest -m "not cuda" -q
D:\conda\envs\gsdiff-spi\python.exe -m pytest -m cuda -q
D:\conda\envs\gsdiff-spi\python.exe -m compileall -q gsdiff scripts tests
D:\conda\envs\gsdiff-spi\python.exe -I -B -X utf8 `
  scripts\experiments\run_campaign.py --help
git diff --check
git diff --name-only -- configs\protocols
```

Expected: CPU suite passes; at least one real CUDA test runs; schemas and CLI
remain usable; no protocol file changes; no whitespace errors.

- [x] **Step 5: Review only against the original Task 5 contract**

Review cache identity, failure isolation, legitimate concurrency, stale
recovery, complete-manifest-last, no-clobber promotion, blind child boundary,
and scientific identity binding. Findings outside the operational boundary are
recorded as backlog and do not restart implementation.

- [ ] **Step 6: Commit and push the runner checkpoint**

```powershell
git diff --cached --check
git commit -m "feat: run experiments through atomic identities"
git fetch origin
git push origin debug/admm-vs-sgd
```

Stage only the reviewed Task 1 files and this rebaseline plan. Never force push.

### Task 2: Implement the Minimum Phase-Aware Aggregation Core

**Files:**

- Create `gsdiff/experiments/phases.py`.
- Create `gsdiff/experiments/aggregation.py`.
- Create `gsdiff/experiments/statistics.py`.
- Create `scripts/experiments/aggregate_campaign.py`.
- Create `scripts/experiments/verify_campaign.py`.
- Create `scripts/experiments/lock_results.py`.
- Create `tests/experiments/test_phases.py`.
- Create `tests/experiments/test_aggregation.py`.
- Create `tests/experiments/test_statistics.py`.
- Create `tests/experiments/test_results_lock.py`.
- Modify `scripts/experiments/run_campaign.py` only to require an explicit
  versioned `--phase` and explicit `--artifact-root` before campaign execution.

**Interfaces:**

```python
@dataclass(frozen=True)
class LogicalRunKey:
    method_id: str
    target_id: str
    motion_id: str
    acquisition_config_id: str
    seed: int
    phase_id: str


def load_complete_records(
    run_root: Path,
    *,
    phase_id: str,
    expected_identities: Mapping[LogicalRunKey, str],
) -> tuple[dict[str, object], ...]: ...


def aggregate_seed_metrics(
    records: Sequence[Mapping[str, object]],
    *,
    required_seeds: Sequence[int],
    n_bootstrap: int = 10_000,
    bootstrap_seed: int = 20260727,
) -> dict[str, object]: ...


def merge_aggregate(
    existing: Mapping[str, object],
    incoming: Mapping[str, object],
) -> dict[str, object]: ...
```

- [ ] **Step 1: RED phase separation**

Test that decision, replay, stress, primary-selection, and confirmatory phase
views materialize only their declared seeds/methods/cells. A decision/replay
identity must not satisfy a different phase, and confirmatory cells must not be
constructed before the 630-run prerequisite record verifies.

- [ ] **Step 2: GREEN phase materialization**

Implement fixed protocol-v1 phase views and exact expected identity maps.
Avoid free-form filters, environment overrides, remote refs, and receipt chains.

- [ ] **Step 3: RED/GREEN exact complete-record loading**

Reject missing, failed, dirty, unblinded, mixed-version, nonfinite, duplicate,
ambiguous, or wrong-phase inputs. Ignore unrelated immutable runs. Write a
separate atomic `partial-report.json` and leave the last complete aggregate
byte-identical when coverage is incomplete.

- [ ] **Step 4: RED/GREEN deterministic paired statistics**

Produce per-seed rows, mean, sample SD (`ddof=1`), paired effects, and the fixed
seed-cluster bootstrap. For `n < 2`, emit JSON `null` for SD/CI; never emit NaN
or infinity.

- [ ] **Step 5: RED/GREEN atomic aggregate and lock refusal**

Validate schema, write a sibling temporary file, flush, replace, reload, and
compare canonical hash. `lock_results.py` refuses partial coverage, wrong
phase, dirty/unblinded/mixed/nonfinite evidence, and paper data that cannot be
regenerated.

- [ ] **Step 6: Verify, review, commit, and push**

```powershell
D:\conda\envs\gsdiff-spi\python.exe -m pytest `
  tests\experiments\test_phases.py `
  tests\experiments\test_aggregation.py `
  tests\experiments\test_statistics.py `
  tests\experiments\test_results_lock.py -q
D:\conda\envs\gsdiff-spi\python.exe -m pytest -m "not cuda" -q
D:\conda\envs\gsdiff-spi\python.exe -m pytest -m cuda -q
git diff --check
git commit -m "feat: aggregate complete paired experiment records"
git push origin debug/admm-vs-sgd
```

### Task 3: Freeze Readiness Inputs

**Files:** Versioned protocol/method configs, checkpoint provenance records,
readiness evidence, and the implementation ledger.

- [ ] Audit every native budget and method eligibility without changing the
  scientific grid.
- [ ] Resolve the diffusion checkpoint locator and training provenance. If the
  existing checkpoint cannot be proven, retrain a versioned secondary prior;
  never invent missing history.
- [ ] Prove corrected datasets, checkpoint hashes, GPU/disk preflight, and
  phase counts.
- [ ] Change `execution_ready` only after the corresponding evidence verifies.
- [ ] Run an 11-method non-publication pilot and use measured times to report a
  realistic campaign ETA.

### Task 4: Execute the Locked Numerical Sequence

- [ ] Run 207 decision cells and freeze the selection record.
- [ ] Freeze one clean `publication_experiment_commit`.
- [ ] From that commit run replay 207, stress 126, and primary-selection 297;
  verify all 630 before constructing confirmatory requests.
- [ ] Run untouched confirmatory seeds 73 and 101: exactly 198 runs.
- [ ] Run supplement-only 231, OOD 198, and failure-budget 180.
- [ ] After every phase, verify exact membership, finite evidence, shared
  datasets across methods, and immutable resume behavior.

### Task 5: Lock Results and Prepare Publication Evidence

- [ ] Generate `results-lock-v1` from complete aggregates and raw arrays.
- [ ] Independently recompute a declared sample of metrics from arrays.
- [ ] Export compact tidy tables and non-tabular figure evidence with hashes,
  shapes, dtypes, units, and regeneration commands.
- [ ] Inventory repository debris; quarantine only reviewed regenerable items.
  Do not permanently delete valuable or uncertain files.

### Task 6: Build and Review the Paper

- [ ] Install or provision the project-local fixed TeX/PDF/font toolchain.
- [ ] Build the evidence map, LaTeX macros/tables, and shared Python figure
  system from results-lock data only.
- [ ] Produce six main figures and their supplementary evidence.
- [ ] Verify references claim by claim and cross-check bibliographic fields.
- [ ] Draft Methods/Results first; then Introduction, Discussion, Conclusion,
  Abstract, and Title.
- [ ] Reduce AI-like prose through argument structure, terminology consistency,
  paragraph function, and sentence rhythm—not mechanical synonym replacement.
- [ ] Complete two clean builds, render every PDF page, inspect full resolution,
  and resolve all Critical/Major independent review findings.
- [ ] Deliver a venue-neutral anonymous review package. Do not invent missing
  authorship, affiliation, funding, conflict, license, DOI, or venue metadata.

## Stop Rules

- A scientific or algorithmic defect stops the campaign and requires an impact
  decision before reruns.
- An infrastructure defect receives the smallest direct fix; it does not reopen
  the operational boundary or trigger a new security subsystem.
- No confirmatory seed may be read or materialized before the 630-run gate.
- No paper number or result claim is frozen before results lock.
- Do not promise a calendar ETA before the pilot measures per-method runtime.
