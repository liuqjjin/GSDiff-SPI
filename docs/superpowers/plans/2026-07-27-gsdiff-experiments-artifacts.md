# GSDiff-SPI Experiment and Artifact Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development to execute this plan one task at a
> time. Use superpowers:systematic-debugging for every failed cell and
> superpowers:verification-before-completion before locking any campaign.

**Goal:** Replace path-based experiment reuse with manifest-backed immutable
runs, finish Claude Code's interrupted target-by-motion study, execute the
approved primary/supplementary/OOD/ablation campaigns, and lock only complete,
reproducible evidence for the paper.

**Architecture:** A versioned protocol resolves into one dataset identity and
one method identity per target/motion/seed cell. Every method consumes the same
serialized measurements for that cell. A content-addressed runner writes into
a temporary directory, validates required artifacts, then promotes the run
atomically. Aggregation reads only complete manifests, preserves unrelated
cells, and produces paired seed-level statistics plus compact publication
datasets.

**Tech Stack:** Python 3.12, PyTorch 2.8, NumPy, SciPy, scikit-image, PyYAML,
jsonschema, pytest 9, CUDA 12.8, Git, PowerShell.

## Preconditions and Non-Negotiable Rules

- Complete
  `docs/superpowers/plans/2026-07-27-gsdiff-correctness-reproducibility.md`
  first, including all CPU tests.
- Use `D:\conda\envs\spi\python.exe` for every Python command.
- Never reuse a result based only on its path or the presence of
  `results.json`.
- Never generate a separate synthetic dataset inside a method runner.
- Ground-truth images, trajectories, PSNR, SSIM, nRMSE, and trajectory error
  are forbidden for tuning and early stopping.
- `gsdiff_tv` is the primary method identity. `gsdiff_diffusion` is a
  secondary distribution-dependent method and must remain a separate row.
- A locked aggregate cannot contain missing seeds, failed runs, dirty-code
  identities, or mixed protocol versions.
- The finite search space and selection rule are declared before inspecting
  confirmatory seeds. "Optimal" means best within this declared grid and
  protocol; it is not a claim of universal mathematical optimality.
- No destructive cleanup occurs in this plan. Candidates are inventoried and
  moved to a recoverable quarantine only after exact path review.
- Execute every native command through the fail-closed `Invoke-Checked` pattern
  shown in the campaign gate; never allow a later successful command to mask an
  earlier nonzero exit.

## Canonical File Structure

Create:

- `configs/protocols/pilot-v1.yaml`
- `configs/protocols/primary-v1.yaml`
- `configs/protocols/supplement-grid-v1.yaml`
- `configs/protocols/ood-v1.yaml`
- `configs/protocols/ablations-v1.yaml`
- `configs/protocols/methods-v1.yaml`
- `configs/protocols/scientific-contracts-v1.yaml`
- `configs/protocols/noise-calibration-v1.yaml`
- `schemas/experiment-manifest-v1.schema.json`
- `schemas/experiment-aggregate-v1.schema.json`
- `schemas/publication-artifacts-v1.schema.json`
- `gsdiff/experiments/protocol.py`
- `gsdiff/experiments/identity.py` from the correctness plan is extended, not
  replaced.
- `gsdiff/experiments/methods.py`
- `gsdiff/experiments/manifest.py`
- `gsdiff/experiments/runner.py`
- `gsdiff/experiments/aggregation.py`
- `gsdiff/experiments/statistics.py`
- `scripts/experiments/build_datasets.py`
- `scripts/experiments/run_campaign.py`
- `scripts/experiments/aggregate_campaign.py`
- `scripts/experiments/verify_campaign.py`
- `scripts/experiments/lock_results.py`
- `scripts/maintenance/inventory_artifacts.py`
- `scripts/maintenance/quarantine_artifacts.py`
- `tests/experiments/test_protocol.py`
- `tests/experiments/test_identity.py`
- `tests/experiments/test_methods.py`
- `tests/experiments/test_manifest.py`
- `tests/experiments/test_runner.py`
- `tests/experiments/test_aggregation.py`
- `tests/experiments/test_statistics.py`
- `tests/experiments/test_campaign_cli.py`
- `tests/maintenance/test_artifact_inventory.py`
- `docs/experiments/protocol-v1.md`
- `docs/experiments/method-registry-v1.md`
- `docs/experiments/selection-policy-v1.md`
- `docs/experiments/legacy-to-corrected.md`
- `docs/reproducibility/results-lock.md`
- `docs/reproducibility/cleanup-report.md`
- `paper/figure_data/` for compact locked CSV/JSON created near the end.

Modify:

- `scripts/run_eval_matrix.py`
- `scripts/run_multiseed.py`
- `scripts/autoresearch.py`
- `scripts/run_baselines.py`
- `train.py`
- `gsdiff/baselines/common.py`
- `gsdiff/baselines/cs.py`
- `gsdiff/baselines/gidc.py`
- `gsdiff/baselines/inr.py`
- `gsdiff/baselines/monin.py`
- `gsdiff/baselines/recinr.py`
- `gsdiff/baselines/tv3d.py`
- `.gitignore`
- `README.md`
- `CLAUDE.md`

Generated but untracked:

- `artifacts/datasets/<dataset_identity_sha256>/`
- `artifacts/runs/<full_identity_sha256>/`
- `artifacts/aggregates/<campaign_id>/`
- `_trash/<YYYY-MM-DD>-artifact-quarantine/`

## Locked Protocol Matrix

The implementation must encode the following values literally in versioned
YAML rather than reconstructing them from prose.

Three identity documents are deliberately separate:

- `campaign_id` names a requested matrix such as `primary-v1` or
  `supplement-grid-v1`;
- a `scientific_contract_id` resolves through
  `scientific-contracts-v1.yaml` to immutable acquisition, evaluation, and
  method-definition content; its canonical content SHA enters run identity;
- `campaign_sha256` hashes target/motion/seed/method membership for aggregation
  and the results lock, but not individual run identity.

`primary-v1`, `supplement-grid-v1`, and `ood-v1` use
`scientific_contract_id: gsdiff-sim-v1` and the same contract SHA. Therefore an
overlapping logical cell in two campaigns resolves to one identical run
identity. Campaign membership is an aggregation concern. `pilot-v1` and
`ablations-v1` use their own contract IDs. Reusing a contract ID with changed
contract content is rejected; different campaign files may legitimately share
that contract while having different campaign hashes.

### Shared acquisition

```yaml
image_size: [64, 64]
num_frames: 20
train_measurements: 2560
holdout_measurements: 250
pattern_family: bernoulli
pattern_values: [0, 1]
snr_db: 25
metric_version: metrics-v1
primary_seeds: [7, 11, 42]
confirmatory_seeds: [73, 101]
```

Locked protocol enums use the implementation's canonical `bernoulli` ID and
record its values explicitly. Human aliases such as `bernoulli_01` may be
accepted only by a migration parser and must resolve to `bernoulli` before
config hashing, so aliases cannot create distinct identities.

For each acquisition cell, calibrate one absolute detector-noise sigma from the
noiseless Bernoulli `0/1` training measurements at the requested 25 dB. Use
that same sigma for its training and neutral held-out measurements. For a
cross-pattern study, all pattern families reuse the corresponding Bernoulli
reference sigma. `noise-calibration-v1.yaml` declares the formula, reference
cell grain, and seed mapping; the generated calibration record stores sigma,
reference dataset/hash, realized SNR, and code/config hashes. The holdout
policy is fixed uniform `[0,1]` patterns from the declared independent seed
stream. Neither train nor holdout sigma is recalibrated per alternate pattern
family.

### Main-text grid

- targets: `tank`, `digit5`, `usaf`
- motions: `trans`, `rot`, `transrot`
- seeds: `7`, `11`, `42`, `73`, `101`
- methods: all 11 registered methods

These three targets represent a natural image, a sparse glyph, and a
resolution-chart regime. `letterR` remains in the full supplement to avoid
duplicating the sparse-glyph regime in the main table.

### Claude completion grid retained in the supplement

- targets: `tank`, `digit5`, `letterR`, `usaf`
- motions: `trans`, `rot`, `transrot`, `accel`
- seeds: `7`, `11`, `42`
- methods: all 11 registered methods

Overlapping main-text cells reuse the exact same complete run identity. They
are not rerun under a second directory name.

The exact accounting is:

- main logical runs: `3 * 3 * 5 * 11 = 495`;
- supplement logical runs: `4 * 4 * 3 * 11 = 528`;
- overlap: `3 * 3 * 3 * 11 = 297`;
- supplement-only additions: `528 - 297 = 231`;
- deduplicated union: `495 + 528 - 297 = 726` runs;
- acquisition datasets: main `45`, supplement `48`, overlap `27`, union `66`.

### OOD and failure analysis

- OOD targets: `cx_camera`, `cx_clutter`
- OOD motions: `trans`, `rot`, `transrot`
- OOD seeds: `7`, `11`, `42`
- OOD methods: all 11 registered methods
- OOD total: `2 * 3 * 3 * 11 = 198` runs and `18` datasets
- failure targets: `cx_coins`, `cx_text`
- failure motion: `transrot`
- failure seeds: `7`, `11`, `42`
- failure train-measurement counts: `320`, `640`, `1280`, `2560`, `3840`,
  `5120`
- failure methods: `dgi`, `tv3d`, `recinr_se2`, `gsdiff_tv`,
  `gsdiff_diffusion`
- failure total: `2 * 1 * 3 * 6 * 5 = 180` runs and `36` datasets
- the failure study is a measurement-budget/identifiability study, not a
  representation-win claim

### Method IDs

```text
dgi
static_cs
perframe_cs
tv3d
monin
gidc3dtv
recinr
siren
recinr_se2
gsdiff_tv
gsdiff_diffusion
```

## Task 1: Encode and Validate Versioned Protocols

**Files:**

- Create the eight versioned files under `configs/protocols/`.
- Create `gsdiff/experiments/protocol.py`.
- Create `tests/experiments/test_protocol.py`.
- Create `docs/experiments/protocol-v1.md`.

**Interface:**

```python
@dataclass(frozen=True)
class ExperimentCell:
    scientific_contract_id: str
    scientific_contract_sha256: str
    campaign_id: str
    target: str
    motion: str
    seed: int
    method: str


def load_protocol(path: Path) -> dict[str, object]
def expand_cells(protocol: Mapping[str, object]) -> tuple[ExperimentCell, ...]
def validate_protocol(protocol: Mapping[str, object]) -> None
```

`scientific_contract_id` is the human-readable immutable contract name and
`scientific_contract_sha256` is the canonical content hash. Both enter
`RunIdentity`. `campaign_id` is scheduling/aggregation metadata and must not
enter individual run identity.

- [ ] **Step 1: Write RED tests for exact matrix expansion**

Assert the main grid contains `3 * 3 * 5 * 11 = 495` unique cells. Assert the
Claude supplement contains `4 * 4 * 3 * 11 = 528` logical cells, the overlap is
exactly `297`, the supplement-only set is `231`, and the union is `726`.
At acquisition grain, assert main `45`, supplement `48`, overlap `27`, and
union `66`. Assert the OOD expansion is exactly `198` runs/`18` datasets and
the failure expansion is exactly `180` runs/`36` datasets. Overlaps share the
same `(scientific_contract_id, scientific_contract_sha256, target, motion,
seed, method)` key.

- [ ] **Step 2: Write RED tests for invalid protocol rejection**

Reject duplicate method IDs, unknown targets/motions, empty seed lists,
nonpositive measurements, a holdout count of zero, missing metric version,
reuse of a scientific contract ID with changed contract content, and a campaign
whose declared campaign SHA does not match its matrix. Do not reject two
different campaigns merely because they share a valid scientific contract.

- [ ] **Step 3: Implement strict YAML loading and expansion**

The loader must use a restricted safe loader, reject unknown top-level keys,
reject YAML anchors and aliases before construction, and reject scalar types
outside JSON's `null`/boolean/string/finite-number domain (including implicit
timestamps and binary values). It preserves declared order for human-readable
reports. Protocol hashes use canonical JSON, not raw YAML bytes.

- [ ] **Step 4: Document the exact cell counts and overlap rule**

Include main, supplement, OOD, failure, and ablation matrices. State that the
corrected protocol does not inherit legacy random-pattern outputs.

- [ ] **Step 5: Verify and commit**

```powershell
D:\conda\envs\spi\python.exe -m pytest tests/experiments/test_protocol.py -q
git diff --check
git add configs/protocols gsdiff/experiments/protocol.py tests/experiments/test_protocol.py docs/experiments/protocol-v1.md
git commit -m "feat: encode versioned experiment protocols"
```

## Task 2: Define Immutable Manifest Schemas

**Files:**

- Create `schemas/experiment-manifest-v1.schema.json`.
- Create `schemas/experiment-aggregate-v1.schema.json`.
- Create `gsdiff/experiments/manifest.py`.
- Create `tests/experiments/test_manifest.py`.
- Modify `gsdiff/experiments/identity.py`.
- Modify `tests/experiments/test_identity.py`.
- Consume and verify `requirements-lock.txt` and
  `docs/reproducibility/environment-lock.json` created by the correctness plan.

**Manifest contract:**

```json
{
  "schema_version": "experiment-manifest-v1",
  "status": "complete",
  "run_id": "gsdiff-sim-v1--gsdiff_tv--tank--trans--s7--<hash8>",
  "identity_sha256": "<sha256>",
  "protocol": {
    "scientific_contract_id": "gsdiff-sim-v1",
    "scientific_contract_sha256": "<sha256>",
    "target": "tank",
    "motion": "trans",
    "seed": 7,
    "method": "gsdiff_tv"
  },
  "code": {
    "git_commit": "<40 hex>",
    "dirty_worktree": false,
    "source_tree_sha256": null
  },
  "config": {
    "resolved": {},
    "sha256": "<sha256>"
  },
  "inputs": {
    "dataset_identity_sha256": "<sha256>",
    "measurements_file_sha256": "<sha256>",
    "evaluation_truth_file_sha256": "<sha256>",
    "dataset_manifest_sha256": "<sha256>",
    "assets": {},
    "checkpoints": {}
  },
  "runtime": {
    "python": "",
    "pytorch": "",
    "cuda": "",
    "gpu": "",
    "os": "",
    "dependencies_sha256": "<sha256>",
    "environment_lock_sha256": "<sha256>"
  },
  "execution": {
    "command": [],
    "started_at_utc": "",
    "ended_at_utc": "",
    "return_code": 0,
    "runtime_seconds": 0.0,
    "peak_vram_bytes": 0
  },
  "measurement": {
    "train_count": 2560,
    "holdout_count": 250,
    "pattern_family": "bernoulli",
    "requested_snr_db": 25,
    "noise_calibration_id": "bernoulli-25db-v1",
    "noise_calibration_sha256": "<sha256>",
    "noise_sigma_absolute": 0.0123,
    "realized_train_snr_db": 24.97,
    "realized_holdout_snr_db": 25.04
  },
  "metrics": {
    "version": "metrics-v1",
    "path": "metrics.json",
    "sha256": "<sha256>"
  },
  "artifacts": [
    {
      "role": "reconstruction",
      "path": "reconstruction.npz",
      "sha256": "<sha256>",
      "size_bytes": 123,
      "schema_version": "reconstruction-v1",
      "required": true
    }
  ]
}
```

`run_id` is display-only. The on-disk directory name and cache key are the full
64-character `identity_sha256`; an eight-character suffix is never sufficient
for storage identity.
Campaign membership is not stored in the immutable run manifest. Separate
campaign index/aggregate records map `campaign_id` to expected full identities.
Thus primary and supplement references to the same identity reuse byte-identical
run manifests; a different requesting campaign cannot mutate the run.

- [ ] **Step 1: Write RED schema tests**

Test one valid complete manifest and invalid variants: omitted config,
dirty-worktree locked run, missing dataset hash, return code nonzero with
`complete`, missing metrics version, invalid SHA length, negative runtime, and
an undeclared extra property. Also reject a scientific-contract ID without its
content SHA, a calibration ID without its record SHA, and a runtime whose
installed-distribution fingerprint does not equal the pinned environment lock.
Resolve one identity from both primary and supplement campaigns and assert the
serialized reusable run manifest is byte-identical while the two campaign
indices independently reference it.

- [ ] **Step 2: Implement constructors and validation**

```python
def build_manifest(*, status: str, identity: RunIdentity, ...) -> dict:
    ...

def validate_manifest(value: Mapping[str, object]) -> None:
    ...

def load_complete_manifest(
    path: Path,
    *,
    expected_identity_sha256: str | None = None,
) -> dict[str, object] | None:
    ...
```

Use `jsonschema` for shape validation and explicit semantic checks for
`status/return_code/dirty_worktree`.
Every artifact item has a role, safe relative path, SHA-256, byte size, schema
version, and required flag. A complete run enumerates every required output;
unlisted or hash-mismatched files cannot satisfy completion.

- [ ] **Step 3: Extend and test the exact run-identity contract**

Use the correctness plan's `build_run_identity()` interface unchanged in
meaning. Verify that identity changes independently for:

- `scientific_contract_id` and `scientific_contract_sha256`;
- resolved method/config, dataset identity, assets, and checkpoints;
- clean code commit, or diagnostic dirty flag plus deterministic source-tree
  hash;
- exact installed-distribution/environment-lock fingerprint; and
- metric implementation version.

Require `execution_class="blind_method_child"`; reject compatibility-unblinded
or unknown classes before manifest construction. The manifest and runner accept
only this constructed identity, and tests prove an unblinded compatibility
command cannot create a complete manifest or content-addressed run directory.

Publication campaign entry points reject a dirty tree before creating a run.
Diagnostic dirty runs require a deterministic tracked-diff plus untracked-source
hash and can never enter an aggregate or results lock. At startup, canonicalize
the exact installed distributions, hash them, compare them with
`environment-lock.json`, and fail closed on mismatch rather than merely
recording the mismatch.

- [ ] **Step 4: Make incomplete and failed states explicit**

Allowed status values are `running`, `failed`, and `complete`. Only complete
manifests are reusable or aggregatable. Failed manifests retain stderr tail and
diagnostic paths but never masquerade as final runs.

- [ ] **Step 5: Verify and commit**

```powershell
D:\conda\envs\spi\python.exe -m pytest tests/experiments/test_manifest.py tests/experiments/test_identity.py -q
git add schemas gsdiff/experiments/manifest.py gsdiff/experiments/identity.py tests/experiments/test_manifest.py tests/experiments/test_identity.py requirements-dev.txt
git commit -m "feat: validate immutable experiment manifests"
```

## Task 3: Build One Persistent Dataset per Acquisition Cell

**Files:**

- Modify `gsdiff/data/artifacts.py`.
- Create `scripts/experiments/build_datasets.py`.
- Create `tests/experiments/test_campaign_cli.py`.
- Modify `.gitignore`.

**Dataset identity excludes method** and includes scientific-contract ID and
content SHA, target, motion, seed, acquisition parameters, noise-calibration
ID/SHA, asset hashes, generator implementation commit, environment
fingerprint, and generator configuration hash.

Expected directory:

```text
artifacts/datasets/<dataset_identity_sha256>/
  measurements.npz
  evaluation-truth.npz
  dataset-manifest.json
  preview.png
```

`measurements.npz` contains only acquisition arrays and non-GT metadata needed
by a method: patterns, bucket measurements, frame indices, time grid, H/W/T/K,
holdout measurements/patterns/indices, and declared acquisition parameters.
It contains no canonical image, GT frames, GT trajectory, or evaluation
metric. `evaluation-truth.npz` contains those evaluator-only arrays. Child
method commands receive only the measurements path; the parent runner loads
truth after the child exits and computes `metrics-v1`.

- [ ] **Step 1: Write RED tests for method-independent reuse**

Resolve `gsdiff_tv` and `dgi` for the same acquisition cell and assert they
receive the same dataset identity and byte-identical `measurements.npz`.
Inspect the blind file's member names and prove no GT/canonical/trajectory
field is present. Run a dummy method in a subprocess and prove it cannot
resolve the evaluator-only truth path from its arguments or environment.

- [ ] **Step 2: Write RED tests for hash sensitivity**

Changing target asset bytes, seed, motion, SNR, pattern family, train K, or
holdout K must change the dataset identity. Changing only method or optimizer
must not.

- [ ] **Step 3: Implement atomic dataset creation**

Write both files into a sibling temporary directory. Load them back, compare
every array bit-exactly, validate the dataset manifest, compute final hashes,
then rename. If the target identity already exists, validate it instead of
overwriting it.

Keep three hashes distinct:

- `dataset_identity_sha256`: canonical acquisition/generator identity and
  directory name;
- `measurements_file_sha256` and `evaluation_truth_file_sha256`: exact file
  bytes;
- `dataset_manifest_sha256`: exact manifest bytes.

A generator-code change can produce a new identity even when arrays happen to
match; do not overload one `dataset_sha256` field with all three meanings.

- [ ] **Step 4: Add dry-run and verify-only CLI modes**

```powershell
D:\conda\envs\spi\python.exe scripts\experiments\build_datasets.py --protocol configs\protocols\pilot-v1.yaml --dry-run
D:\conda\envs\spi\python.exe scripts\experiments\build_datasets.py --protocol configs\protocols\pilot-v1.yaml
D:\conda\envs\spi\python.exe scripts\experiments\build_datasets.py --protocol configs\protocols\pilot-v1.yaml --verify-only
```

- [ ] **Step 5: Verify and commit**

```powershell
D:\conda\envs\spi\python.exe -m pytest tests/data/test_artifacts.py tests/experiments/test_campaign_cli.py -q
git add gsdiff/data/artifacts.py scripts/experiments/build_datasets.py tests/experiments/test_campaign_cli.py .gitignore
git commit -m "feat: persist shared immutable acquisition cells"
```

## Task 4: Create a Strict Method Registry

**Files:**

- Create `gsdiff/experiments/methods.py`.
- Create `tests/experiments/test_methods.py`.
- Create `docs/experiments/method-registry-v1.md`.
- Modify `scripts/run_baselines.py`.
- Modify `train.py`.
- Modify `gsdiff/baselines/common.py`.
- Modify the adapters in `gsdiff/baselines/cs.py`,
  `gsdiff/baselines/gidc.py`, `gsdiff/baselines/inr.py`,
  `gsdiff/baselines/monin.py`, `gsdiff/baselines/recinr.py`, and
  `gsdiff/baselines/tv3d.py`.

**Interface:**

```python
@dataclass(frozen=True)
class ResolvedMethod:
    method_id: str
    command_template: tuple[str, ...]
    semantic_config: Mapping[str, object]
    required_outputs: tuple[str, ...]
    checkpoint_paths: tuple[Path, ...]


def resolve_method_semantics(
    method_id: str,
    *,
    base_config: Mapping[str, object],
    measurements_path: Path,
) -> ResolvedMethod:
    ...


def materialize_method_execution(
    method: ResolvedMethod,
    *,
    staging_output_dir: Path,
) -> tuple[tuple[str, ...], Mapping[str, object]]:
    ...
```

Semantic config and command templates use stable tokens such as
`${MEASUREMENTS_PATH}` and `${OUTPUT_DIR}`. Their hashes exclude absolute
workspace paths and random staging UUIDs. Only after the runner creates its
staging directory does materialization substitute exact paths. The
materialized command is logged but never fed back into semantic identity.

Before child launch, the parent copies or verified-hardlinks
`measurements.npz` into a fresh child-input directory containing no dataset
manifest or truth file. The child receives only that staged path, an output
path, code, and allowlisted checkpoints; its current directory and sanitized
environment expose no upstream dataset/artifact root. A Python audit hook logs
file-open attempts and the runner rejects access to any non-allowlisted data
path. This is a reproducible procedural blind boundary for trusted method code,
not a claim of an adversarial operating-system sandbox.

- [ ] **Step 1: Write RED identity tests**

Assert all 11 method IDs resolve once and only once. Explicitly assert:

```python
resolve_method_semantics("gsdiff_tv", ...).semantic_config["solver"]["prior_type"] == "tv"
resolve_method_semantics("gsdiff_diffusion", ...).semantic_config["solver"]["prior_type"] == "diffusion"
```

Assert the two identities have different config hashes even when all other
cell fields match. Assert `gaussian_count` is required only for Gaussian scenes
and rejected for every non-Gaussian scene; changing an inactive field can never
manufacture a new identity. Assert native `recinr` cannot be passed to the
generic GSDiff scene/solver resolver.

- [ ] **Step 2: Move ad hoc method mutations into declarative YAML**

`methods-v1.yaml` contains the fixed algorithm choices and any predeclared
GT-free selection grid. Remove hidden mutations such as `scene_type` branches
from campaign orchestration.

- [ ] **Step 3: Enforce the blind child boundary through real entry points**

Refactor `train.py`, `scripts/run_baselines.py`, the shared baseline helper, and
every affected baseline adapter so the method process accepts only
`measurements.npz` plus semantic config. Remove child-side ground-truth
generation/loading, image/trajectory metric computation, and GT-based
selection. Add real subprocess tests for both GSDiff and at least one baseline:
give the child a fresh directory containing only a staged
`measurements.npz`, remove the evaluator truth and upstream artifact roots from
arguments/environment/current directory, and assert the run still produces
reconstruction output. Verify the audit log contains no GT/data-root access.
Add negative tests in which a deliberately truth-seeking child guesses a
sibling path or scans the upstream dataset root; the allowlist/audit wrapper
must fail the run. The parent runner alone receives `evaluation-truth.npz`
after child exit.

- [ ] **Step 4: Normalize completed-run outputs and ownership**

Every successfully completed staging run contains:

```text
reconstruction.npz
metrics.json
method-info.json
stdout.log
stderr.log
```

The child owns `reconstruction.npz` and `method-info.json`; the parent runner
captures `stdout.log`/`stderr.log`, loads evaluator truth after child exit, and
writes `metrics.json`. The method info records parameter count, convergence
status, selected hyperparameters, selection objective, and checkpoint hashes
where applicable. Method children do not compute image/trajectory metrics and
cannot import the evaluator truth file. Refactor any baseline selection helper
that currently writes PSNR into a tuning table; only held-out measurement
residual and predeclared compute/convergence constraints are visible inside the
child.

`scripts/run_baselines.py` gains a strict single-method interface:

```text
--method <one canonical method ID>
--dataset <absolute measurements.npz>
--output-dir <absolute staging directory>
```

One process produces one method identity. The parent runner computes metrics
afterward. Legacy `--name`, multi-baseline batches, arbitrary `--override`, and
autoresearch calls either translate into a fully content-hashed ad-hoc
campaign or fail with an actionable migration message; they cannot bypass the
registry.

- [ ] **Step 5: Record ReCINR provenance**

Document upstream repository URL, upstream commit
`9149d1d228db2e4eb3ae852a004f1d9e95ee0229`, vendored-file hashes, local
modifications, authorship, and license state. A missing license blocks a
submission archive that redistributes the vendored code; it does not justify
inventing a license.

- [ ] **Step 6: Verify and commit**

```powershell
D:\conda\envs\spi\python.exe -m pytest tests/experiments/test_methods.py tests/experiments/test_campaign_cli.py -q
git add configs/protocols/methods-v1.yaml gsdiff/experiments/methods.py scripts/run_baselines.py train.py gsdiff/baselines tests/experiments/test_methods.py tests/experiments/test_campaign_cli.py docs/experiments/method-registry-v1.md
git commit -m "feat: separate all experiment method identities"
```

## Task 5: Add an Atomic Content-Addressed Runner

**Files:**

- Create `gsdiff/experiments/runner.py`.
- Create `scripts/experiments/run_campaign.py`.
- Create `tests/experiments/test_runner.py`.
- Modify `scripts/run_eval_matrix.py`.
- Modify `scripts/run_multiseed.py`.
- Modify `scripts/autoresearch.py`.

**Interface:**

```python
@dataclass(frozen=True)
class RunRequest:
    cell: ExperimentCell
    dataset_dir: Path
    method: ResolvedMethod
    identity: RunIdentity


@dataclass(frozen=True)
class RunOutcome:
    status: Literal["complete", "cached", "failed"]
    run_dir: Path | None
    diagnostic_dir: Path | None
    return_code: int


def run_request(request: RunRequest, artifact_root: Path) -> RunOutcome:
    ...

def reusable_run(
    artifact_root: Path,
    expected_identity: RunIdentity,
) -> Path | None:
    ...
```

- [ ] **Step 1: Write RED cache tests**

Reject caches when the resolved config, code commit, dirty flag, dataset,
asset, checkpoint, dependency lock, method identity, or metric version
changes. Reject directories with only `results.json`, truncated artifacts,
nonzero return code, or `status != complete`.

- [ ] **Step 2: Write RED atomicity tests**

Run a dummy child that exits nonzero after writing a partial metric. Assert no
complete final directory exists and the previous valid result remains
byte-identical. Run a successful child and assert final promotion occurs only
after required-file hashes validate.

Add two concurrent processes requesting the same identity. Use an atomic
claim/lock file so only one executes; the other waits, then validates the
winner. Test `FileExistsError`/rename races, stale claim recovery, and process
interruption. Fsync artifact files, the complete manifest, the staging
directory, and the parent directory before/after the final same-filesystem
rename.

- [ ] **Step 3: Implement lifecycle**

1. Resolve and hash the request.
2. Validate an exact cache match.
3. Atomically claim the full identity and create a sibling
   `<identity_sha256>.tmp-<uuid>` directory.
4. Write `running` manifest and resolved config.
5. run child with captured logs and peak VRAM sampling;
6. validate outputs and compute metrics/artifact hashes;
7. write `complete` manifest last;
8. atomically rename into `artifacts/runs/<identity_sha256>`.

On failure, move diagnostics into
`artifacts/failed/<display_run_id>-<timestamp>` and return a failed
`RunOutcome`. The CLI maps any requested failed outcome to a nonzero exit
status.

- [ ] **Step 4: Migrate old entry points**

Keep old script names as compatibility wrappers that translate arguments into
protocol/campaign requests. They must print a deprecation warning and may not
use file-existence caching.

- [ ] **Step 5: Verify and commit**

```powershell
D:\conda\envs\spi\python.exe -m pytest tests/experiments/test_runner.py tests/experiments/test_campaign_cli.py -q
git add gsdiff/experiments/runner.py scripts/experiments/run_campaign.py scripts/run_eval_matrix.py scripts/run_multiseed.py scripts/autoresearch.py tests/experiments
git commit -m "feat: run experiments through atomic identities"
```

## Task 6: Aggregate Complete Runs Incrementally

**Files:**

- Create `gsdiff/experiments/aggregation.py`.
- Create `gsdiff/experiments/statistics.py`.
- Create `scripts/experiments/aggregate_campaign.py`.
- Create `scripts/experiments/verify_campaign.py`.
- Create `scripts/experiments/lock_results.py`.
- Create `tests/experiments/test_aggregation.py`.
- Create `tests/experiments/test_statistics.py`.
- Create `tests/experiments/test_results_lock.py`.

**Interfaces:**

```python
def load_complete_records(
    run_root: Path,
    *,
    scientific_contract_id: str,
    scientific_contract_sha256: str,
    expected_identities: Mapping[tuple[str, str, str, int], str],
) -> tuple[dict[str, object], ...]:
    ...


def aggregate_seed_metrics(
    records: Sequence[Mapping[str, object]],
    *,
    required_seeds: Sequence[int],
    n_bootstrap: int = 10_000,
    bootstrap_seed: int = 20260727,
) -> dict[str, object]:
    ...


def merge_aggregate(
    existing: Mapping[str, object],
    incoming: Mapping[str, object],
) -> dict[str, object]:
    ...
```

- [ ] **Step 1: Write RED subset/failure tests**

Start with cells A and B. Re-aggregate only A and assert B remains unchanged.
Inject a failed A and assert the existing aggregate remains byte-identical.
Retain an older complete A from a previous code/config identity and assert it
is reported as superseded while the exact currently expected identity is used.
Inject two artifact directories that both claim the same expected identity but
have different manifest/artifact hashes and require an integrity error.

- [ ] **Step 2: Write deterministic paired-statistics tests**

Report per-seed rows, arithmetic mean, sample standard deviation (`ddof=1`),
paired method differences, and a 10,000-resample percentile 95% bootstrap CI
with a fixed seed. Match methods first at the exact
`(dataset_identity_sha256, target, motion, seed)` grain. For a headline
multi-target/motion effect, average paired cell effects within each seed, then
cluster-bootstrap seed IDs so repeated target/motion rows do not masquerade as
independent samples. Unit tests use a small injected bootstrap count.

When `n < 2` (including the one-seed pilot), return `sd=null` and
`bootstrap_ci=null` with the observed `n`; never serialize NaN/Infinity.

- [ ] **Step 3: Encode partial status**

Missing required seeds leaves the last authoritative complete aggregate
byte-identical. The command writes a separate atomic `partial-report.json`
containing `status: partial`, missing logical keys, failed requests, and
superseded historical identities, then exits nonzero. A partial report cannot
be locked or exported to the paper. Historical complete runs with a different
expected identity remain immutable and inspectable but are not an aggregation
conflict.

- [ ] **Step 4: Write aggregates atomically**

Validate against `experiment-aggregate-v1.schema.json`, write a temporary JSON,
fsync, replace, reload, and compare the canonical SHA. The CLI prints cell
counts for requested/complete/cached/failed/missing.

- [ ] **Step 5: Implement verification and lock refusal before any campaign**

Using dummy complete/partial/superseded manifests, test that
`verify_campaign.py` checks exact campaign membership and that
`lock_results.py` refuses partial coverage, dirty runs, unblinded tuning
outputs, mixed metric versions, ambiguous identities, and nonfinite values.
The production lock export runs later in Task 11, but its code and refusal
tests are part of the code freeze before pilot execution.

- [ ] **Step 6: Verify and commit**

```powershell
D:\conda\envs\spi\python.exe -m pytest tests/experiments/test_aggregation.py tests/experiments/test_statistics.py tests/experiments/test_results_lock.py -q
git add gsdiff/experiments/aggregation.py gsdiff/experiments/statistics.py scripts/experiments/aggregate_campaign.py scripts/experiments/verify_campaign.py scripts/experiments/lock_results.py schemas/experiment-aggregate-v1.schema.json tests/experiments
git commit -m "feat: aggregate complete paired experiment records"
```

## Task 7: Run the Corrected Pilot and Freeze the Search

**Files:**

- Use `configs/protocols/pilot-v1.yaml`.
- Create `docs/experiments/legacy-to-corrected.md`.
- Create `docs/experiments/selection-policy-v1.md`.
- Modify `configs/protocols/ablations-v1.yaml` only before pilot inspection.

**Pilot matrix:**

- image size: `32 × 32`
- frames: `4`
- measurements: `128` training plus `16` held-out
- pattern/noise: Bernoulli `0/1`, `25 dB`
- target: `tank`
- motion: `transrot`
- seeds: `7`
- methods: all 11 registered method IDs
- sharply reduced optimizer steps declared in YAML

The pilot validates execution, artifact, and metric plumbing only. It cannot
enter a publication table or select the final method configuration.

- [ ] **Step 1: Verify a clean code identity**

```powershell
$dirty = git status --porcelain=v1 --untracked-files=all
if ($LASTEXITCODE -ne 0 -or $dirty) { throw "Pilot requires a clean worktree." }
D:\conda\envs\spi\python.exe -m pytest -m "not cuda" -q
D:\conda\envs\spi\python.exe -m pytest -m cuda -q
nvidia-smi
```

Do not start if the worktree is dirty, CPU tests fail, the CUDA smoke test
fails, or disk/GPU memory is insufficient.

- [ ] **Step 2: Dry-run and inspect every resolved cell**

```powershell
D:\conda\envs\spi\python.exe scripts\experiments\run_campaign.py --protocol configs\protocols\pilot-v1.yaml --dry-run --explain
```

The report must show dataset-identity SHA, run-identity SHA, exact command,
checkpoint SHA,
estimated cell count, and whether an exact cache exists.

- [ ] **Step 3: Execute and verify**

```powershell
D:\conda\envs\spi\python.exe scripts\experiments\build_datasets.py --protocol configs\protocols\pilot-v1.yaml
D:\conda\envs\spi\python.exe scripts\experiments\run_campaign.py --protocol configs\protocols\pilot-v1.yaml --workers 1
D:\conda\envs\spi\python.exe scripts\experiments\verify_campaign.py --protocol configs\protocols\pilot-v1.yaml
```

- [ ] **Step 4: Compare legacy and corrected behavior**

Use legacy artifacts only as explicitly labelled historical evidence. Document
the weighted-TV numerical change, primary metric change, cache invalidation,
runtime, failures, and any method-order reversal. Never copy a legacy number
into a corrected table.

- [ ] **Step 5: Freeze the finite search**

Before running seeds `73` and `101`, commit the exact selection grid and rule.
Selection minimizes held-out measurement residual subject to convergence and a
literal 1,800-second/run and 15,032,385,536-byte VRAM cap. Use the exact
21-sample convergence/stability predicate in Task 8; candidates within 0.5%
relative residual are tied and use median runtime, peak VRAM, then config ID.

- [ ] **Step 6: Commit only compact pilot reports**

```powershell
git add docs/experiments/legacy-to-corrected.md docs/experiments/selection-policy-v1.md configs/protocols
git commit -m "docs: freeze corrected experiment selection policy"
```

## Task 8: Run Controlled Selection and Ablations Before Confirmation

**Locked axes:**

- representation: `gaussian`, `siren`, `grid`, `recinr_se2`
- solver: `sgd`, `hqs`, `admm`
- prior: `tv2d`, `tv3d_corrected`, `diffusion`
- motion warmup fraction: `0`, `0.1`, `0.2`, `0.4`
- temporal TV weight: `0`, `0.05`, `0.1`, `0.3`
- train measurements: `640`, `1280`, `1920`, `2560`, `3840`, `5120`
- SNR dB: `15`, `20`, `25`, `30`
- pattern: `bernoulli`, `gaussian`, `hadamard_natural`, `fourier`
- motion fit: `matched`, `translation_only`, `rotation_only`
- Gaussian count: `250`, `500`, `1000`, `1500`

Use staged factor isolation rather than a blind Cartesian explosion. The YAML
must specify the anchor cell for each one-factor study and the joint shortlist
that follows.

The semantic anchor configuration is literal:

```yaml
representation: gaussian
solver: admm
prior: tv3d_corrected
motion_warmup_fraction: 0.2
temporal_tv_weight: 0.1
gaussian_count: 1000
```

The selection policy is also literal:

```yaml
selection_rule:
  objective: heldout_normalized_l2
  objective_formula: "l2(pred-y)/max(l2(y),1e-12)"
  aggregation: unweighted_mean_over_3_anchors_x_3_seeds
  required_complete_cells: 9
  residual_tie_relative_tolerance: 0.005
  convergence:
    required_history_samples: 21
    final_window_samples: 3
    minimum_relative_improvement_from_initial: 0.01
    maximum_final_window_to_best_ratio: 1.05
    require_all_finite: true
  compute_cap:
    wall_time_seconds_per_run: 1800
    peak_vram_bytes: 15032385536
    on_exceed: ineligible-retain-artifacts
  tie_break_order: [median_runtime_seconds, peak_vram_bytes, config_id]
confirmation_rule:
  rule_id: primary-confirmation-v1
  scientific_contract_id: gsdiff-sim-v1
  method: gsdiff_tv
  comparator: recinr_se2
  metric: psnr_global_affine
  targets: [tank, digit5, usaf]
  motions: [trans, rot, transrot]
  seeds: [73, 101]
  minimum_effect_db: 0.0
  require_each_seed_mean_delta_strictly_greater_than_minimum: true
  require_complete_finite_pairs_per_seed: 9
  require_method_failure_count_not_greater_than_comparator: true
  inferential_significance_claim: false
```

`heldout_normalized_l2` is computed on the evaluator-blind held-out
measurements. The best eligible objective wins. Candidates within `0.5%`
relative objective of the best are tied and use the exact tie-break order.
Over-budget/nonconvergent runs remain visible but are ineligible. Protocol
tests compare every field/value above, reject missing caps/tolerances, and
verify inactive fields cannot create distinct semantic identities.

One-factor selection comprises the anchor plus every single-axis alternative:
`1 + 3 + 2 + 2 + 3 + 3 + 3 = 17` unique configurations. The joint shortlist
is not chosen after observing results; `ablations-v1.yaml` contains exactly
these six named configurations:

```yaml
joint_shortlist:
  - {id: j1, representation: recinr_se2, solver: hqs,  prior: diffusion,       motion_warmup_fraction: 0.2, temporal_tv_weight: 0.1,  gaussian_count: null}
  - {id: j2, representation: grid,       solver: admm, prior: diffusion,       motion_warmup_fraction: 0.2, temporal_tv_weight: 0.1,  gaussian_count: null}
  - {id: j3, representation: siren,      solver: sgd,  prior: tv3d_corrected,  motion_warmup_fraction: 0.1, temporal_tv_weight: 0.05, gaussian_count: null}
  - {id: j4, representation: gaussian,   solver: hqs,  prior: tv2d,             motion_warmup_fraction: 0.1, temporal_tv_weight: 0.05, gaussian_count: 1500}
  - {id: j5, representation: recinr_se2, solver: sgd,  prior: tv2d,             motion_warmup_fraction: 0.4, temporal_tv_weight: 0.3,  gaussian_count: null}
  - {id: j6, representation: grid,       solver: hqs,  prior: tv3d_corrected,  motion_warmup_fraction: 0.4, temporal_tv_weight: 0.05, gaussian_count: null}
```

`gaussian_count` is active only for `representation: gaussian`; it is required
there and must be `null` or absent for `siren`, `grid`, and `recinr_se2`. The
loader removes null inactive keys before semantic hashing, so null and absent
canonicalize identically; any non-null inactive value is rejected. The
Gaussian-count one-factor sweep therefore uses the Gaussian anchor above.
Protocol validation cannot hash two semantically identical inactive-field
variants. `grid` and `recinr_se2` use the existing generic scene/solver adapter;
native `recinr` remains the separate registered baseline method and is not
composed with GSDiff solvers.

Thus selection schedules `(17 + 6) * 3 anchors * 3 seeds = 207` logical
runs. The post-freeze stress stage uses the selected method and the anchor plus
single-axis alternatives over K, SNR, pattern, and three motion-fit models:
`1 + 5 + 3 + 3 + 2 = 14` configurations, or
`14 * 3 anchors * 3 seeds = 126` runs. One clean publication commit therefore
contributes `207 + 126 = 333` lockable logical runs. Including the earlier 207
decision runs, the complete selection workflow executes `540` logical runs
before retries; decision runs are provenance rather than publication numerical
inputs. The protocol test must fail unless the six shortlist IDs and all counts
match exactly; semantic duplicate identities are a configuration error, not a
reason to silently reduce the count.

Selection anchors are declared before execution:

- `tank/trans`
- `digit5/rot`
- `usaf/transrot`
- seeds `7`, `11`, `42`

Representation, solver, prior, warmup, temporal weight, and Gaussian count may
enter method selection. Measurement budget, SNR, pattern family, and
misspecification are stress studies run after the method configs freeze; they
cannot retroactively tune the main method. Confirmatory seeds `73` and `101`
do not run in this task.

The resolver converts warmup fraction into each solver's native unit with one
declared rounding rule: SGD uses optimizer steps; ADMM/HQS use outer
iterations, while inner-step histories retain the derived boundary. Store both
fraction and resolved integer in `method-info.json`.

- [ ] **Step 1: Validate that no confirmatory seed enters selection**

Add a guard test that raises if `73` or `101` appears in any tuning campaign.
Also assert the exact `207` decision, `207` post-freeze replay, `126` stress,
`333` lockable publication-commit total, and `540` workflow execution total,
and compare the six joint-shortlist mappings field-for-field with the locked
YAML. Compare the objective formula, all convergence thresholds, GPU-time/VRAM
caps, over-budget policy, tie tolerance/order, and confirmation rule
field-for-field as well.

- [ ] **Step 2: Run one-factor sweeps**

Rank by held-out residual under convergence and compute caps. Record all
candidates, not only the winner. Apply the literal normalized-residual,
eligibility, cap, and tie policy above; an implementer may not choose new
tolerances after seeing results.

- [ ] **Step 3: Run the declared joint shortlist**

Run the six literal configurations above. Rank them by mean held-out residual
over the nine selection cells with the literal policy above. Ties choose lower
median runtime, then lower peak VRAM, then lexicographically smaller
configuration ID.

- [ ] **Step 4: Freeze and document**

Call the selected setting "grid-optimal under protocol v1". If multiple
settings are indistinguishable within the declared tolerance, select the
winner with the exact runtime/VRAM/config-ID tie-break and report the tie.
Commit the resolved `methods-v1.yaml`, selection decision records, and their
hashes before creating any dataset for seeds `73` or `101`:

```powershell
git add configs/protocols/methods-v1.yaml configs/protocols/ablations-v1.yaml docs/experiments/selection-policy-v1.md
git commit -m "data: freeze grid-optimal method selection"
```

Record the resulting clean HEAD as `publication_experiment_commit`. The
untouched primary campaign in Task 9 supplies confirmation.

- [ ] **Step 5: Replay selection under the frozen publication commit**

Rerun all 207 selection logical cells from
`publication_experiment_commit`. Recompute ranking with no config change and
require the winner/tie set to match the original decision record exactly.
These replay runs are the lockable numerical selection evidence. The original
pre-freeze decision runs remain immutable provenance proving when the choice
was made, but are not numerical inputs to publication figures/tables. Any
ranking mismatch aborts the protocol and requires diagnosis/new approval.

- [ ] **Step 6: Run the locked robustness studies**

Using the frozen selected method, execute the K, SNR, pattern-family, and
motion-misspecification axes on selection seeds only. For cross-pattern
comparisons, use the fixed absolute noise calibrated from the Bernoulli
reference. All 126 stress runs use the same
`publication_experiment_commit`.

- [ ] **Step 7: Generate compute and convergence data**

Export iteration-residual histories, runtime, VRAM, parameter count, and
failure rates needed for the publication figures.

## Task 9: Execute and Verify the Main Campaign

**Files generated:** untracked immutable artifacts plus tracked compact
aggregate data only after lock.

- [ ] **Step 1: Preflight**

Confirm the frozen method-registry hash, exact
`publication_experiment_commit`, clean status, free disk,
GPU identity, checkpoint hashes, all dataset hashes, and 495 expected logical
cells. Save preflight JSON.

- [ ] **Step 2: Execute primary selection seeds first**

Run seeds `7`, `11`, and `42` for all methods and cells. Resume only through
exact manifest identity. Do not start confirmatory seeds while any primary cell
is failed or missing.

- [ ] **Step 3: Diagnose failures without broad changes**

For each failure, use systematic debugging. If a code fix is required, commit
it and abort the current campaign. Because full repository commit is
identity-bearing, create a new scientific-contract patch version. If the fix
can affect semantics, eligibility, residuals, runtime ranking, or if impact is
uncertain, rerun the 207 decision cells and freeze/commit the new winner. For a
proven infrastructure-only fix, retain the frozen decision record but record
the impact justification. In both cases establish one final clean
`publication_experiment_commit` and unconditionally rerun from it:

- all 207 post-freeze selection replay cells, requiring the frozen winner/tie
  set;
- all 126 stress cells; and
- all 297 main selection-seed cells.

Only after these `207 + 126 + 297 = 630` lockable logical runs pass may Step 4
touch the 198 confirmatory cells. Prior replay/stress/main artifacts remain
immutable but are superseded. Never hand-edit a manifest.

- [ ] **Step 4: Execute untouched confirmatory seeds**

Run seeds `73` and `101` using the frozen configs. No hyperparameter, method
selection, comparator, or headline effect definition may change after
inspecting these seeds. If a necessary change invalidates confirmation, v1
loses its confirmation claim; a new protocol and a newly approved,
predeclared seed set are required before another confirmatory run. Do not imply
that uninspected reserve seeds already exist.

This step contains exactly `3 targets * 3 motions * 2 seeds * 11 methods = 198`
logical runs and executes only after the 297 selection-seed main cells and all
replayed selection/stress evidence pass.

- [ ] **Step 5: Aggregate and verify**

```powershell
D:\conda\envs\spi\python.exe scripts\experiments\aggregate_campaign.py --protocol configs\protocols\primary-v1.yaml
D:\conda\envs\spi\python.exe scripts\experiments\verify_campaign.py --protocol configs\protocols\primary-v1.yaml --require-complete
```

Report selection seeds and confirmatory seeds separately. A pooled five-seed
summary is descriptive only and cannot be called an independent confirmation.
Apply `primary-confirmation-v1` exactly: GSDiff-TV versus ReCINR-SE2,
global-affine PSNR, all nine matched cells per seed, both raw seed-level mean
deltas strictly greater than 0 dB, complete finite coverage, and no greater
GSDiff-TV failure count. Show both raw effects and avoid significance language.

## Task 10: Finish Claude's Supplement Grid, OOD, and Failure Studies

- [ ] **Step 1: Execute non-overlapping supplement cells**

Reuse exact overlapping primary runs, then run the remaining `letterR`,
`accel`, and primary-seed combinations from `supplement-grid-v1`. Verification
must report supplement `528`, overlap `297`, supplement-only `231`, main plus
supplement union `726`, and acquisition-dataset union `66`.

- [ ] **Step 2: Execute OOD cells**

Run `cx_camera` and `cx_clutter` on seeds `7`, `11`, and `42`. Report
`gsdiff_diffusion` jointly with in-/out-of-distribution status. Do not discard
cases where it underperforms TV. Require all 11 methods: exactly `198` logical
runs and `18` datasets.

- [ ] **Step 3: Execute failure targets across measurement budget**

Run `cx_coins` and `cx_text` over the locked K grid. Treat these as
measurement-budget/identifiability cases. Preserve reconstructions and
residuals that demonstrate failure, not just successful examples. Assign a
failure-cause label only through the predeclared diagnostic rubric; otherwise
use `undetermined` or `consistent-with`, not a causal conclusion.
Require the five declared methods, `transrot`, three declared seeds, and six K
values: exactly `180` logical runs and `36` datasets.

- [ ] **Step 4: Verify all three aggregates**

Every requested logical cell must resolve to exactly one expected identity,
one dataset hash shared across methods, all required seeds, and finite metrics
or an explicit predeclared failure record. Emit a machine-readable missing and
superseded-cell report and fail on any unresolved inconsistency.

If Task 10 exposes a code defect, abort rather than patching one OOD/failure
subset. Any new code commit invalidates every locked numerical run from the old
`publication_experiment_commit` and returns to the Task 8/9 recovery sequence;
all completed publication sets are rerun from the new final commit. Because
confirmatory outcomes have already been inspected, v1 then loses its untouched
confirmation claim and requires a newly approved protocol/seed set before
claiming confirmation again.

## Task 11: Lock Results and Export Publication Data

**Files:**

- Use the already-tested `scripts/experiments/verify_campaign.py`.
- Use the already-tested `scripts/experiments/lock_results.py`.
- Create `schemas/publication-artifacts-v1.schema.json`.
- Modify `tests/experiments/test_results_lock.py`.
- Create `docs/reproducibility/results-lock.md`.
- Create compact CSV/JSON under `paper/figure_data/`.

**Lock contract:**

```json
{
  "lock_version": "results-lock-v1",
  "selection_decision_code_commit": "<sha>",
  "selection_decision_record_sha256": "<sha256>",
  "experiment_code_commit": "<sha>",
  "aggregation_tool_commit": "<sha>",
  "lock_tool_source_commit": "<sha>",
  "protocol_sha256": {},
  "scientific_contract_sha256": {},
  "campaign_sha256": {},
  "aggregate_sha256": {},
  "dataset_manifest_sha256": {},
  "checkpoint_sha256": {},
  "blind_tuning_audit_sha256": "<sha256>",
  "publication_files": {},
  "publication_artifacts_manifest_sha256": ""
}
```

- [ ] **Step 1: Write RED lock refusal tests**

Refuse partial aggregates, dirty code, duplicate logical identities, changed
source aggregate bytes, missing artifact hashes, unknown metric versions, and
paper data that cannot be regenerated byte-for-byte.
For protocol v1, require one clean `experiment_code_commit` across every locked
numerical run: the 207 post-freeze selection replay, 126 stress runs, primary,
supplement, OOD, and failure campaigns. The earlier selection-decision runs are
not publication numerical inputs; lock their decision record/hash and code
commit separately, and verify the post-freeze replay reproduced the winner/tie
set. Record the aggregation-tool commit and lock-tool source commit separately
so later documentation commits are not misrepresented as the code that
produced the measurements.

The blind-tuning audit is a required generated artifact. It hashes every
method's declared input capability, child command/environment allowlist,
held-out selection record, `method-info.json`, and the runner event proving
that evaluator truth was opened only after child termination. Refuse any run
whose child received a truth path, GT field, image metric, or evaluator import.
The lock stores the audit SHA and the verifier reconstructs it from manifests
rather than trusting a prose declaration.

- [ ] **Step 2: Export compact tidy data**

Create one row per observation at the appropriate grain, including protocol,
method, target, motion, seed, ID/OOD label, primary metrics, residuals, motion
errors, runtime, VRAM, parameter count, convergence state, and source run ID.

Also create a `publication-artifacts-v1` manifest for non-tabular evidence
required by Figures 2, 5, and 6:

- predeclared representative GT, raw reconstruction, DGI, and error arrays;
- estimated and ground-truth trajectories with time coordinates;
- raw iteration histories, including the sigma/rho actually used;
- measurement-budget sweep observations and reconstruction examples;
- convergence flags, failure diagnostics, and any human-reviewed failure label.

Each entry declares schema version, source run ID, relative path, SHA-256,
shape, dtype, units/range, protocol/metric version, and exact regeneration
command. Human labels include rubric version, evidence fields, reviewer, and
`undetermined` as the safe default. The results lock hashes this manifest and
every referenced file; a tidy metric row alone is not sufficient publication
evidence.

- [ ] **Step 3: Recompute independently**

Recompute a random declared sample directly from reconstruction arrays and
datasets. Compare every metric to the manifest with strict tolerances before
locking. Reload every non-tabular publication artifact, validate
shape/dtype/range against its schema, and regenerate it byte-for-byte or by a
declared semantic-array hash.

- [ ] **Step 4: Freeze the lock tool, then generate and commit evidence**

Only compact aggregates, figure data, schemas, documentation, and lock hashes
are tracked. Raw datasets, checkpoints, per-frame dumps, and transient logs
remain untracked.

```powershell
D:\conda\envs\spi\python.exe -m pytest tests\experiments\test_results_lock.py -q
git diff --check
git add scripts/experiments/verify_campaign.py scripts/experiments/lock_results.py schemas/publication-artifacts-v1.schema.json tests/experiments/test_results_lock.py
git commit -m "feat: freeze publication evidence lock tooling"

$dirty = git status --porcelain=v1 --untracked-files=all
if ($LASTEXITCODE -ne 0 -or $dirty) { throw "Results must be generated from a clean source commit." }
D:\conda\envs\spi\python.exe scripts\experiments\lock_results.py --protocol-root configs\protocols --output docs\reproducibility\results-lock.md
D:\conda\envs\spi\python.exe -m pytest tests\experiments\test_results_lock.py -q
git diff --check
git add paper/figure_data docs/reproducibility/results-lock.md
git commit -m "data: lock corrected publication evidence"
```

`lock_tool_source_commit` is the clean first commit above, not the later commit
that contains the generated lock. The lock never attempts to contain the hash
of the commit that contains itself.

## Task 12: Inventory and Quarantine Repository Debris

**Files:**

- Create `scripts/maintenance/inventory_artifacts.py`.
- Create `scripts/maintenance/quarantine_artifacts.py`.
- Create `tests/maintenance/test_artifact_inventory.py`.
- Create `docs/reproducibility/cleanup-report.md`.
- Modify `.gitignore`.

- [ ] **Step 1: Inventory without changing files**

Record absolute-relative path, type, byte size, timestamps, Git status,
manifest status, content hash, duplicate group, downstream references, and a
reasoned classification:

```text
authoritative
regenerable-cache
duplicate
scratch
incomplete
corrupt
unknown-review-required
```

- [ ] **Step 2: Prove classification**

Anything referenced by a complete manifest, results lock, paper figure data,
README, config, or checkpoint/data card is authoritative until proven
otherwise. `unknown-review-required` never moves automatically.

- [ ] **Step 3: Generate a dry-run move plan**

Resolve and validate every path remains inside the repository. Reject symlinks
or junctions whose resolved target escapes it. Never target the repository
root, `$HOME`, or an unresolved glob.

- [ ] **Step 4: Move approved candidates to dated quarantine**

Only after exact target review, move candidates to
`_trash/<date>-artifact-quarantine/` with a reversible mapping JSON. Do not
delete them.

- [ ] **Step 5: Regenerate all locked outputs**

Regenerate aggregate and paper data files from retained authoritative inputs.
Compare canonical hashes with the results lock. Record tracked/untracked file
counts and sizes before and after quarantine.

- [ ] **Step 6: Commit the governance record**

```powershell
git add scripts/maintenance tests/maintenance .gitignore docs/reproducibility/cleanup-report.md
git commit -m "chore: inventory and quarantine reproducible debris"
```

Permanent deletion is a separate user-authorized action after the quarantine
has survived full regeneration.

## Campaign Completion Gate

All conditions must hold:

```powershell
$ErrorActionPreference = "Stop"
function Invoke-Checked {
    param([string]$Label, [scriptblock]$Command)
    & $Command
    $code = $LASTEXITCODE
    if ($code -ne 0) { throw "$Label failed with exit code $code." }
}

Invoke-Checked "CPU tests" { & 'D:\conda\envs\spi\python.exe' -m pytest -m "not cuda" -q }
Invoke-Checked "CUDA tests" { & 'D:\conda\envs\spi\python.exe' -m pytest -m cuda -q }
Invoke-Checked "Campaign verification" { & 'D:\conda\envs\spi\python.exe' scripts\experiments\verify_campaign.py --all-locked --require-complete }
Invoke-Checked "Results lock" { & 'D:\conda\envs\spi\python.exe' scripts\experiments\lock_results.py --verify-only }
Invoke-Checked "Whitespace check" { git diff --check }
$dirty = git status --porcelain=v1 --untracked-files=all
if ($LASTEXITCODE -ne 0 -or $dirty) { throw "Campaign completion requires a clean worktree." }
```

Manual review checklist:

- [ ] The main grid has 495 complete logical cells and no duplicate identity.
- [ ] The Claude grid has all 528 logical cells; overlap/new/union counts are
  exactly 297/231/726 and acquisition union is 66.
- [ ] OOD and failure studies contain exactly 198 and 180 logical runs,
  respectively.
- [ ] Each acquisition cell has one `dataset_identity_sha256` shared by every
  method.
- [ ] Confirmatory seeds were not used for tuning.
- [ ] The reconstructed blind-tuning audit proves no child accessed evaluator
  truth or GT-derived metrics.
- [ ] OOD regressions and failure cases remain visible.
- [ ] Every reported mean is recoverable from per-seed rows.
- [ ] Every paper datum resolves to a complete run manifest.
- [ ] Quarantined files are recoverable and no permanent deletion occurred.
- [ ] The results-lock verification is byte-for-byte reproducible.
