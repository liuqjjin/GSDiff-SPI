# Task 4 Strict Method Registry and Blind Child Boundary

**Status:** Approved for implementation on 2026-07-28  
**Applies to:** Task 4 of
`docs/superpowers/plans/2026-07-27-gsdiff-experiments-artifacts.md`  
**Precedence:** This document resolves Task 4 implementation details left open
by the immutable plan and the ignored controller addendum. It does not change
the locked campaign matrices, acquisition contract, confirmation rule, or
publication claims.

## Objective

Task 4 must replace the repository's ad hoc method dispatch with one strict,
content-addressed registry for exactly eleven canonical method identities. A
method child must receive only a staged measurement artifact, a canonical
method configuration, allowlisted code and checkpoints, and a writable output
directory. It must not receive evaluator truth or compute image/trajectory
metrics.

The implementation must make the following statements mechanically true:

1. every publication method identity resolves once and only once;
2. aliases cannot create additional identities or bypass the registry;
3. scientific semantics are hashed before machine-specific paths are
   materialized;
4. algorithm randomness is separated from acquisition randomness;
5. held-out selection uses one locked raw physical residual;
6. a child owns only reconstruction and method metadata;
7. evaluator truth and publication metrics remain parent-only concerns;
8. unresolved checkpoint or licensing evidence blocks promotion rather than
   being guessed.

## Scope

Task 4 includes:

- the canonical method registry and semantic resolver;
- declarative method configurations and per-profile budgets;
- path-free semantic hashes and domain-separated algorithm seeds;
- a structured execution materializer;
- a procedural blind-boundary bootstrap with file-access auditing;
- strict one-method child entry points for GSDiff and baselines;
- removal of child-side evaluator-truth access and GT-derived selection;
- a versioned `method-info-v2` contract;
- real subprocess smoke and negative path-access tests;
- ReCINR source/provenance documentation.

Task 4 does not include:

- the atomic campaign scheduler, retry/resume state machine, parent metric
  evaluator, or final five-file promotion transaction; those remain Task 5;
- publication GPU execution;
- regeneration of figures or manuscript text;
- modification of the separate `D:\SPI\ReCINR` working tree;
- claiming an adversarial operating-system sandbox;
- inventing checkpoint training provenance or redistribution rights.

## Canonical identities

The registry contains exactly these canonical IDs:

| ID | Execution family | Canonical scientific binding |
| --- | --- | --- |
| `dgi` | baseline | Direct ghost-imaging reconstruction; no optimizer trajectory |
| `static_cs` | baseline | Static TV-CS, ADMM 150, `rho=0.5`, locked lambda grid |
| `perframe_cs` | baseline | Per-frame TV-CS, ADMM 120, `rho=0.5`, locked lambda grid |
| `tv3d` | baseline | 3D TV primal-dual reconstruction, 500 iterations |
| `monin` | baseline | Measurement-derived global translation plus motion-compensated TV-CS, ADMM 150, five motion blocks |
| `gidc3dtv` | baseline | GIDC-3DTV, 2500 Adam steps |
| `recinr` | native baseline | Native ReCINR with `round(1.7 * T)` temporal nodes |
| `siren` | GSDiff core | SIREN representation, random initialization, SGD 4000 |
| `recinr_se2` | GSDiff core | ReCINR-style representation with SE(2), grid 20, SGD 3000 |
| `gsdiff_tv` | GSDiff core | Gaussian-1000 scene, ADMM, corrected 3D-TV prior |
| `gsdiff_diffusion` | GSDiff core | Gaussian-1000 scene, ADMM, diffusion prior |

`gsdiff_tv` remains the primary identity. A diffusion prior always resolves to
`gsdiff_diffusion`; it cannot replace or alias the primary row. Native
`recinr` never enters the generic GSDiff scene/solver resolver.

Input aliases may be accepted only by an explicit migration layer. Resolution
must return a canonical ID before hashing, logging, or execution. Unknown IDs,
duplicate declarations, and identity-changing inactive fields are errors.

Campaign execution profiles normalize through one locked mapping:

```text
primary-full-v1       -> publication-v1
supplement-full-v1    -> publication-v1
ood-full-v1           -> publication-v1
failure-budget-v1     -> publication-v1
pilot-smoke-v1        -> controller-cpu-smoke-v1
ablation-selection-v1 -> ablation-selection-v1
```

The normalized method profile, not the campaign alias, enters
`method_config_sha256`, so scientifically identical cells remain reusable
across primary, supplement, OOD, and failure campaigns. The parent manifest
still records the requested campaign profile.

The locked structural `pilot-v1` matrix declares `method_config_id=default`.
Task 4 resolves that input through an exact profile-scoped alias
`pilot-smoke-v1/default -> smoke-default-v1`. `ResolvedMethod` records both
the requested and normalized IDs, and only the normalized smoke ID enters the
method hash. This preserves the locked pilot membership while preventing smoke
semantics from overwriting publication `default`.

The six literal `ablation-j1-v1` through `ablation-j6-v1` configurations are
validated and content-hashed in Task 4, but their profile is fail-closed with
`execution_ready=false` and blocker
`missing-versioned-ablation-native-budgets`. Task 4 does not invent the
previously unspecified common SGD/ADMM/HQS selection budgets. A later approved
version must remove that blocker before materialization.

## Scientific decisions that close legacy conflicts

The approved implementation binds the following previously inconsistent
choices:

1. `gidc3dtv` uses 2500 publication optimization steps. The legacy dispatcher
   value of 2000 is noncanonical.
2. Native `recinr` uses `round(1.7 * T)` temporal nodes. Documentation that
   says `T` nodes is corrected.
3. `static_cs` and `perframe_cs` select lambda on the locked held-out
   measurement objective, then refit once using all measurements. The
   pre-refit held-out score and selected lambda are retained in method info.
4. `motion_warmup_fraction` is distinct from splitting warmup. For ADMM/HQS,
   its resolved count is
   `ceil(motion_warmup_fraction * outer_iterations)`. During those outer
   iterations scene parameters are frozen while motion and the normal
   splitting state transitions continue. The warmup is part of, not added to,
   the declared outer-iteration budget.
5. The current diffusion checkpoint is identified by logical ID
   `gsdiff-diffusion-prior-v1` and exact SHA-256
   `667948800911acb9f9a7271e20af5692b0f007007d0fc32a15ac169eba32c5dd`.
   Local nonpromotable smoke may use a verified copy. Publication execution is
   blocked until a reproducible locator and training provenance are recorded.
6. ReCINR local research execution may continue with recorded source and
   hashes. A public or submission archive that redistributes vendored ReCINR
   code remains blocked until license and copyright evidence is complete.

These decisions are scientific configuration and must be versioned if changed.

## Locked publication and smoke parameters

All publication profiles carry a 1800-second wall-time cap and a
15,032,385,536-byte resource cap. The native algorithm budgets below remain
part of semantic identity.

| Method | Publication parameters | `controller-cpu-smoke-v1` |
| --- | --- | --- |
| `dgi` | One direct reconstruction pass | One pass |
| `static_cs` | `rho=0.5`, ADMM 150, Chambolle 100, lambda grid `[0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]`, final all-measurement refit | ADMM 1, lambda `[0.001]` |
| `perframe_cs` | `rho=0.5`, ADMM 120, Chambolle 100, the same lambda grid, final all-measurement refit | ADMM 1, lambda `[0.001]` |
| `tv3d` | Primal-dual 500, operator-norm power iterations 30, `lambda_xy=[0.003,0.03,0.3]`, `lambda_t=[0.001,0.01,0.1,1.0]`, final all-measurement refit | One primal-dual iteration, first grid pair, operator-norm iterations 30 |
| `monin` | `rho=1.0`, ADMM 150, Chambolle 100, lambda grid `[0.0001,0.0003,0.001,0.003,0.01,0.03]`, bilinear interpolation, five blocks, polynomial degree one | ADMM 1, first lambda; four blocks for the four-frame pilot |
| `gidc3dtv` | Adam 2500, `lr=0.05`, `betas=[0.5,0.9]`, evaluate every 25, `xi_xy=[0.003,0.03,0.3]`, `xi_t=[0.01,0.1]` | Adam 1, evaluate every 1, first grid pair |
| `recinr` | hidden 32, render layers 3, low-rank order 0, harmonics 2, flow scale 0.5, positional encodings 2 and 5, anneal 0.6, anchor time 0.5, warm 300, flow-only 400, joint 1200, learning rate 0.003 to 0.001, snapshot every 50, nodes `round(1.7*T)` | warm 1, flow-only 1, joint 1; all representation constants unchanged |
| `siren` | `w0=8`, random initialization, SGD 4000, scene learning rate 0.003 | SGD 1, zero motion warmup |
| `recinr_se2` | grid 20, random initialization, SGD 3000, scene learning rate 0.003, motion-only warmup 500 | SGD 1, zero motion warmup |
| `gsdiff_tv` | Gaussian count 1000, DGI-adaptive initialization, ADMM outer 80 by inner 50, splitting warmup 20, `rho=0.1`, growth 1.1, corrected 3D-TV, TV 0.005, soft-TV 0.006, temporal TV 0.1, scene LR 0.009, motion LR 0.15, motion warmup fraction 0.2 | outer 1 by inner 1, both warmups zero |
| `gsdiff_diffusion` | Gaussian count 1000, DGI-adaptive initialization, ADMM outer 80 by inner 50, splitting warmup 20, motion warmup fraction/count 0, `rho=0.1`, growth 1.1, proximal weight 0.005, soft-TV 0.006 with temporal weight 0.05, scene LR 0.009, motion LR 0.15, logical checkpoint `gsdiff-diffusion-prior-v1`, denoise steps 1, clamp `[0,1]`, sigma 0.3 to 0.05, `renoise=false`, DDIM spacing `linear` | outer 1 by inner 1, both warmups zero, verified local checkpoint only |

Every smoke row is nonpromotable and is a different configuration identity.
Taking the first member of a candidate grid validates plumbing only and does
not constitute a scientific hyperparameter selection result.

## Registry data model

The production API is immutable:

```python
@dataclass(frozen=True)
class ResolvedMethod:
    method_id: str
    requested_method_config_id: str
    method_config_id: str
    execution_family: str
    command_template: tuple[str, ...]
    semantic_config: Mapping[str, object]
    method_config_sha256: str
    required_child_outputs: tuple[str, ...]
    checkpoint_requirements: tuple["CheckpointRequirement", ...]
    execution_profile: str
    publication_eligible: bool
    selection_eligible: bool
    promotion_eligible: bool
    convergence_status: str
    execution_ready: bool
    execution_blockers: tuple[str, ...]


@dataclass(frozen=True)
class AlgorithmSeed:
    derivation_sha256: str
    seed_u32: int


@dataclass(frozen=True)
class MaterializedMethodExecution:
    argv: tuple[str, ...]
    cwd: Path
    env: Mapping[str, str]
    measurements_path: Path
    method_config_path: Path
    child_output_dir: Path
    expected_acquisition_spec: Mapping[str, object]
    read_allowlist: tuple[Path, ...]
    read_root_allowlist: tuple[Path, ...]
    write_root_allowlist: tuple[Path, ...]
    requested_runtime_device: str
    child_runtime_device: str
    audit_policy_path: Path
    audit_policy_sha256: str
    audit_log_path: Path
    stdout_path: Path
    stderr_path: Path
    materialization_record: Mapping[str, object]


@dataclass(frozen=True)
class MaterializedMethodRequest:
    method: ResolvedMethod
    algorithm_seed: AlgorithmSeed
    dataset_identity_sha256: str
    measurements_file_sha256: str
    expected_acquisition_spec: Mapping[str, object]
    measurements_path: Path
    child_output_dir: Path
    checkpoint_paths: Mapping[str, Path]
    requested_runtime_device: str
    child_runtime_device: str
```

`semantic_config` contains only JSON-compatible scientific values. It cannot
contain absolute paths, staging UUIDs, host names, environment-specific Python
locations, output paths, or unverified arbitrary overrides.

The canonical resolver accepts:

```python
resolve_method_semantics(
    method_id,
    *,
    method_config_id,
    base_config,
    measurements_metadata,
    execution_profile,
) -> ResolvedMethod
```

It validates active/inactive fields. In particular, `gaussian_count` is
required for Gaussian scenes and rejected for non-Gaussian scenes. An
inactive field cannot manufacture a new semantic hash.

## Canonical hashing and algorithm seeds

`method_config_sha256` is the SHA-256 of canonical JSON containing the
canonical method ID, method-config ID, execution family/profile, path-free
semantic configuration, stable command template/checkpoint requirements,
exact child outputs, and parent-owned profile policy. Distinct ablation
method-config IDs therefore
cannot collapse even when they share an execution family. Mappings are
key-sorted; strings remain strings; floats must be finite; paths and platform
separators are rejected.

Algorithm seeds are derived after the path-free method hash:

```text
sha256(canonical_json({
  "domain": "algorithm-seed-v1",
  "cell_seed": <campaign cell seed>,
  "dataset_identity_sha256": <dataset identity>,
  "method_id": <canonical ID>,
  "method_config_sha256": <path-free method hash>
}))
```

The first four digest bytes interpreted as an unsigned big-endian integer form
`algorithm_seed_u32`. The full derivation digest and integer are recorded.
Python, NumPy, and Torch receive this algorithm seed. Acquisition RNG streams
0--3 are never reused as algorithm RNG streams.

## Locked blind selection objective

Every method that selects a hyperparameter from held-out measurements uses
exactly:

```text
pred[k] = sum(P[k] * reconstruction[frame_indices[k]])
heldout_normalized_l2 =
    ||pred - y||_2 / max(||y||_2, 1e-12)
```

The calculation uses float64 physical measurement values. Global or per-frame
z-scoring, MSE, PSNR, SSIM, trajectory error, or evaluator truth are forbidden
selection signals. Method info records:

- formula ID `heldout-normalized-l2-v1`;
- numerator;
- denominator;
- objective value;
- selected candidate;
- the complete predeclared candidate grid.

GIDC flattens snapshot state into each candidate as
`{xi_xy, xi_t, snapshot_step}`. The 1-based step is the completed Adam update;
publication order is registry `xi_xy`, then registry `xi_t`, then steps
`25, 50, ..., 2500`, for exactly 600 rows. ReCINR candidates contain only
`{snapshot_step}` and use the 1-based global optimization step including
warm-up; publication records
`[701 + 50*j for j in range(24)] + [1900]`, for exactly 25 rows. GIDC's
`selected_hyperparameters` remains the two-xi projection, while ReCINR's
remains null. Both methods mark the selected run's unique producing history
step with `reconstruction_source: true`, and bounded history sampling retains
that row.

Solver-side conditioning is allowed only as an internal numerical device. A
candidate and final reconstruction must be transformed back to physical
forward-model scale before this objective or artifact writing; a normalized
iterate may never be compared directly with raw held-out measurements.

Methods without a selection grid record no synthetic selection result.
Methods without a native motion estimate return no trajectory; zero-filled
fake trajectories are forbidden.

## Child result contract

In-process adapters share one GT-free return type:

```python
@dataclass(frozen=True, kw_only=True)
class MethodChildResult:
    method_id: str
    reconstruction: np.ndarray
    estimated_motion_trajectory: np.ndarray | None
    dgi: np.ndarray | None
    info: Mapping[str, object]
    history: tuple[Mapping[str, object], ...]


def run_canonical_method(
    method: ResolvedMethod,
    acquisition: SPIAcquisitionData,
    *,
    algorithm_seed: AlgorithmSeed,
    checkpoint_paths: Mapping[str, Path],
    device: str,
) -> MethodChildResult:
    raise ValueError
```

The adapter receives no ground-truth image, ground-truth trajectory, dataset
root, campaign manifest, evaluator-truth path, arbitrary YAML path, or
arbitrary keyword override. Candidate grids and budgets come only from the
resolved canonical method.

Iteration history is bounded before serialization. `method-info-v2` embeds the
bounded history and its sampling policy; the child does not create a sixth
final artifact. Full unbounded optimizer traces, if temporarily needed for
debugging, are nonpromotable and remain outside the completed-run contract.

## Materialization and blind boundary

Stable command tokens are:

```text
${PYTHON}
${MEASUREMENTS_PATH}
${OUTPUT_DIR}
${METHOD_CONFIG_PATH}
${DATASET_IDENTITY_SHA256}
${ALGORITHM_SEED}
${DEVICE}
${CHECKPOINT:<logical-id>}
${AUDIT_LOG_PATH}
```

Only `materialize_method_execution` may substitute these tokens. Materialized
paths never feed back into semantic identity.

The materializer distinguishes the requested physical device from the
child-visible logical device. `cpu` maps to `cpu` with no
`CUDA_VISIBLE_DEVICES`. A request `cuda:N` maps to
`CUDA_VISIBLE_DEVICES=N` and `${DEVICE}=cuda:0`, because the selected physical
device is renumbered to logical device zero inside the child. Both values are
recorded in the materialized config and request; method code receives only the
child-visible value. In particular, `cuda:1` must never be passed through as
the child device after setting `CUDA_VISIBLE_DEVICES=1`.

The staging layout is:

```text
execution-workspace/
  input/measurements.npz
  config/method-config.json
  checkpoints/<logical-id>
  code/
  work/
  child-output/
  parent/audit/policy.json
  parent/audit/file-opens.jsonl
  parent/logs/stdout.log
  parent/logs/stderr.log
```

The parent copies `measurements.npz` into `input/` and verifies source and copy
hashes before launch. A hardlink is forbidden because a child write could
mutate the source inode. Neither a dataset manifest nor evaluator truth is
present. Code and checkpoint inputs are read-only; only `child-output/`,
`work/`, the audit file, and process logs are writable by their owners.

The child starts from the clean staged code root with a sanitized environment.
A Python bootstrap audit hook:

- canonicalizes file paths before policy checks;
- distinguishes allowed reads from allowed writes;
- logs attempted opens with operation, resolved path, decision, and timestamp;
- rejects sibling traversal, upstream dataset-root scans, truth-like files,
  unlisted checkpoints, symlink/reparse-point escapes, directory enumeration
  outside allowlisted roots, child-created subprocesses, working-directory
  escapes, and writes outside the output/work directories;
- allows the interpreter/runtime modules required by the declared clean code
  environment.

The strict source snapshot excludes `gsdiff/evaluation/**`,
`gsdiff/baselines/_evaluation.py`, and
`gsdiff/data/_artifact_truth.py`. It is launched with
`-I -S -B -X utf8`; the bootstrap sets `sys.dont_write_bytecode=True`,
installs the hook before `site.main()` and staged imports, and executes the
original entry point with `runpy` in the same process. Its environment is
constructed from the selected Python environment, `SystemRoot`, staged
temporary/cache/home directories, and an explicit CUDA device only; real
home values are replaced with staged paths, and it contains no Python-path,
dataset-root, or upstream-workspace variable.

Because compatibility modules currently obtain `evaluate_video` through
`gsdiff.baselines.common.__getattr__`, `cs.py`, `gidc.py`, `monin.py`, and
`tv3d.py` must move that symbol out of module-scope imports and into legacy
evaluation wrappers. A strict-snapshot import test removes the excluded
evaluator/truth files and must still import `gsdiff.experiments.adapters` and
all four baseline modules successfully. This makes the staged dependency
closure executable rather than merely absent by inventory.

For a governed path, policy evaluation checks each lexical existing ancestor
for a symlink/junction/reparse point before resolution. Nonexistent writes
check the nearest existing parent. Only then does it resolve, case-normalize,
and compare against exact/root allowlists.

The audit log begins with a policy-hash-bound `hook-installed` event, uses
contiguous sequence numbers, and ends with `bootstrap-finished`. The parent
rejects malformed/truncated logs, policy mismatch, missing terminal state, and
any denied event even if child code caught `PermissionError`.
`MaterializedMethodExecution` exposes the exact `audit_policy_path` and
`audit_policy_sha256` alongside the log path, so the parent never infers the
expected hash from an untyped record.

This is a reproducible procedural boundary for trusted research code. The
documentation must not call it an adversarial OS sandbox.

## Entry points and compatibility

The strict baseline CLI is:

```text
python scripts/run_baselines.py \
  --method dgi \
  --dataset D:\stage\input\measurements.npz \
  --dataset-identity-sha256 aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \
  --method-config D:\stage\config\method-config.json \
  --algorithm-seed 123456789 \
  --device cpu \
  --output-dir D:\stage\child-output
```

The strict GSDiff child interface in `train.py` follows the same arguments and
single-method/single-dataset semantics. Diffusion additionally receives one
exact `--checkpoint
gsdiff-diffusion-prior-v1=<verified-staged-path>` pair. Each invocation
produces exactly one canonical method result.

Legacy batch flags such as `--name`, free-form `--override`, and arbitrary YAML
selection cannot enter a publication profile. A compatibility path may emit a
clear migration error or create an explicitly ad-hoc, fully content-hashed,
nonpromotable profile. It may not silently translate scientific choices.

## Output ownership, `reconstruction-v2`, and `method-info-v2`

During Task 4 the child may write only:

```text
reconstruction.npz
method-info.json
```

Task 5 will make the completed run's declared `outputs/` bundle contain
exactly:

```text
reconstruction.npz
metrics.json
method-info.json
stdout.log
stderr.log
```

The parent owns metrics and logs. Run manifests, resolved configuration, audit
evidence, and lifecycle state live outside that exact five-file `outputs/`
bundle. The parent loads evaluator truth only after a successful child exit.
Child metadata cannot set or self-attest promotion eligibility.

`reconstruction-v2` is a capability-separated artifact rather than a silent
mutation of legacy `reconstruction-v1`. It contains:

- required `reconstruction`, `frame_indices`, and `time_grid` arrays;
- optional `dgi` and `estimated_motion_trajectory` arrays with exact presence
  flags;
- dataset identity, canonical method ID, array descriptors, and schema ID.

It does not contain execution class, truth access, promotion eligibility,
image metrics, trajectory metrics, or a fabricated trajectory. The legacy
v1 reader remains available only for compatibility artifacts.

`method-info-v2` contains at least:

- schema/version identifier;
- canonical method ID and execution family;
- dataset identity and exact staged measurement hash;
- method configuration hash and canonical semantic configuration;
- algorithm-seed derivation digest and integer;
- exact iteration/step budgets and native units;
- separate splitting-warmup and motion-warmup fields;
- parameter count;
- selected hyperparameters and locked blind objective, when applicable;
- convergence status and bounded convergence history;
- checkpoint logical IDs and verified hashes;
- native trajectory-present flag;
- child start/end timestamps; the parent alone records process exit status;
- reconstruction array metadata and file hash computed before exit.

`parameter_count` means the number of native scalar degrees of freedom fitted
or optimized from measurements, excluding selected hyperparameters and fixed
priors. The exact classical counts are: DGI `0`; static CS `H*W`; per-frame CS
and TV3D `T*H*W`; and Monin
`H*W + 2*polynomial_degree` (one canonical image plus two anchored translation
polynomials, with no fitted intercept or rotation variable). Neural/GSDiff methods report the
exact sum of unique trainable tensor elements actually passed to their
optimizer; a frozen diffusion prior is excluded. Tests bind these formulas so
the field cannot silently switch between “network weights” and “reconstruction
pixels.”

The parent must verify every echoed parent-known field against the resolved
request before promotion.

## Profiles and readiness

Publication semantics and CPU smoke semantics are different method
configurations with different IDs and hashes.

`controller-cpu-smoke-v1`:

- uses one iteration/step or the smallest viable locked candidate grid;
- is always `promotion_eligible=false`,
  `selection_eligible=false`, and `publication_eligible=false`;
- records `smoke-only/not-convergence-assessed`;
- cannot be reused as publication evidence.

Publication runs are additionally bounded by 1800 seconds and
15,032,385,536 bytes. Crossing a cap preserves diagnostics but makes the run
ineligible.

`pilot-v1.execution_ready` remains false until all of the following hold:

1. all eleven IDs resolve with exact budgets;
2. both strict child entry points work;
3. real blind subprocess and negative access tests pass;
4. the raw objective is the only child selection metric;
5. `method-info-v2` and child ownership checks pass;
6. required checkpoint hashes resolve;
7. the repository is clean and verification gates pass;
8. required GPU and disk preflights pass.

The missing diffusion locator/training provenance blocks publication
promotion. The missing ReCINR redistribution evidence blocks an archive that
contains that vendored code. Neither blocker is converted into a false claim.

## ReCINR provenance record

The Task 4 documentation records:

- repository URL `https://github.com/liuqjjin/ReCINR`;
- upstream commit `9149d1d228db2e4eb3ae852a004f1d9e95ee0229`;
- remote and local hashes for each adapted file;
- a concise description of local modifications;
- the repository's declarations and the absence of a repository license file
  at the audited revision;
- archive status `blocked-license-copyright-review`.

The separate ReCINR working tree remains read-only and out of scope.

## Error handling

Registry and materializer failures are fail-closed. Errors must name the
canonical field or path policy that failed without printing sensitive
environment data.

At minimum, fail on:

- unknown or duplicate method IDs;
- alias cycles or aliases used as stored identities;
- missing or inactive scientific fields;
- nonfinite configuration values;
- absolute paths in semantic configuration;
- unknown command tokens;
- missing or hash-mismatched checkpoints;
- dataset identity or measurement hash mismatch;
- any denied file access;
- child output outside the allowlist;
- missing, extra, malformed, or inconsistent child artifacts;
- child-reported identity/config/seed values that differ from the request.

Failed stages remain diagnostic-only and are never considered complete.

## Verification design

Implementation follows test-driven development in four reviewable slices.

### Slice A: registry and identity

- RED tests for exactly eleven unique canonical IDs;
- path-free stable hashes and alias rejection;
- `gsdiff_tv`/`gsdiff_diffusion` semantic separation;
- Gaussian-only `gaussian_count`;
- native ReCINR isolation;
- domain-separated deterministic algorithm seeds;
- publication and smoke budget separation.

### Slice B: GT-free adapters and objective

- raw objective numerical tests, including zero denominator;
- removal of z-score/MSE/PSNR selection paths;
- optional, nonfabricated motion trajectories;
- CS post-selection all-measurement refit;
- fixed GIDC and ReCINR bindings;
- deterministic seeded stochastic methods.

### Slice C: materializer, entry points, and artifacts

- exact token substitution and rejection of unknown tokens;
- stable semantic hashes across different stage roots;
- exact device mapping, including
  `cuda:1 -> CUDA_VISIBLE_DEVICES=1 + child cuda:0`;
- checkpoint hash verification;
- explicit audit-policy path/hash return and policy-bound log validation;
- strict single-method CLI tests;
- `method-info-v2` round trips and ownership validation;
- exact two child outputs.

### Slice D: real blind subprocess

- one baseline and one GSDiff subprocess succeed with only staged
  measurements;
- no truth or upstream dataset access appears in the audit log;
- deliberate sibling-path, absolute upstream-root, directory-scan,
  symlink/reparse escape, and nested-subprocess probes fail;
- child identity/config/seed tampering fails parent validation;
- CPU smoke stays nonpromotable.

After the four slices, run targeted tests, the complete non-CUDA suite,
`compileall`, schema validation, `git diff --check`, and a same-reviewer formal
spec/code/security review. P0, P1, and P2 must all be zero before Task 4 is
committed as complete and pushed.
