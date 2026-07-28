# Task 4 Strict Method Registry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a strict, content-addressed registry for all eleven experiment methods and a tested procedural blind-child boundary that prevents method code from seeing evaluator truth.

**Architecture:** Declarative YAML holds canonical scientific semantics while `gsdiff.experiments.methods` validates, freezes, hashes, and resolves them. GT-free adapters return one typed child result, versioned v2 writers emit exactly two child-owned artifacts, and a structured materializer stages measurements, code, configuration, and verified checkpoints behind a Python audit hook. Legacy v1 paths remain available only as explicitly nonpromotable compatibility paths.

**Tech Stack:** Python 3.12.13 from `D:\conda\envs\spi`, NumPy, PyTorch, PyYAML, jsonschema Draft 2020-12, pytest, PowerShell, Git.

## Global Constraints

- Work only in `D:\Research\gsdiff_spi`; do not modify or run code from `D:\SPI\ReCINR`.
- Use `D:\conda\envs\spi\python.exe` for every Python and pytest command.
- Follow RED -> verify RED -> GREEN -> verify GREEN -> refactor for every production behavior change.
- Preserve the immutable campaign matrices, acquisition contract, method lanes, and confirmation rule `gsdiff_tv` versus `recinr_se2`.
- The registry contains exactly eleven canonical IDs: `dgi`, `static_cs`, `perframe_cs`, `tv3d`, `monin`, `gidc3dtv`, `recinr`, `siren`, `recinr_se2`, `gsdiff_tv`, and `gsdiff_diffusion`.
- `gsdiff_diff` is input compatibility only and always persists as `gsdiff_diffusion`.
- Semantic hashes contain no absolute path, staging UUID, host name, device name, output path, or checkpoint locator.
- Algorithm seeds use domain `algorithm-seed-v1` and never reuse acquisition RNG streams 0--3.
- Blind selection is only float64 raw `||pred-y||_2 / max(||y||_2, 1e-12)` with formula ID `heldout-normalized-l2-v1`.
- Method children receive no evaluator truth and compute no PSNR, SSIM, image NRMSE, or trajectory error.
- A child owns exactly `reconstruction.npz` and `method-info.json`; Task 5's parent will own `metrics.json`, `stdout.log`, and `stderr.log`.
- Missing native motion estimates are absent, never zero-filled.
- `gidc3dtv` publication budget is 2500 steps; native `recinr` uses `round(1.7*T)` nodes; static/per-frame CS refit on all measurements after held-out selection.
- ADMM/HQS motion warmup is `ceil(fraction * outer_iterations)`, freezes scene gradients, preserves splitting transitions, and does not increase the total outer budget.
- Diffusion logical checkpoint ID is `gsdiff-diffusion-prior-v1`, SHA-256 `667948800911acb9f9a7271e20af5692b0f007007d0fc32a15ac169eba32c5dd`.
- Diffusion publication promotion stays blocked until checkpoint locator and training provenance are reproducible.
- A ReCINR-containing public/submission archive stays `blocked-license-copyright-review`; do not invent a license.
- `controller-cpu-smoke-v1` is always nonpromotable, nonselectable, nonpublication, and `smoke-only/not-convergence-assessed`.
- The audit hook is described as a procedural boundary for trusted Python research code, never as an adversarial OS sandbox.
- Preserve legacy v1 readers/writers and corrected Task 1--3 tests; compatibility artifacts cannot satisfy corrected campaigns.
- Keep ignored SDD ledgers and reports append-only.
- Do not run publication GPU experiments in Task 4.

---

## File responsibility map

| File | Responsibility |
| --- | --- |
| `configs/protocols/methods-v1.yaml` | Canonical IDs, lanes, command tokens, fixed profiles, budgets, and checkpoint declarations |
| `gsdiff/experiments/methods.py` | Registry parsing, exact validation, alias normalization, semantic freezing, config hashing, algorithm seed derivation |
| `gsdiff/experiments/objectives.py` | The only publication held-out measurement objective and candidate selection helper |
| `gsdiff/experiments/child_outputs.py` | `MethodChildResult`, reconstruction-v2/method-info-v2 writing, loading, and cross-validation |
| `gsdiff/experiments/adapters.py` | Canonical in-process method dispatch without evaluator truth |
| `gsdiff/experiments/execution.py` | Staging layout, code/data/checkpoint copying, token materialization, sanitized environment |
| `gsdiff/experiments/audit.py` | File/process audit policy and fail-closed audit hook |
| `scripts/experiments/method_child_bootstrap.py` | Installs the audit hook before importing a strict method entry point |
| `scripts/run_baselines.py` | Strict one-baseline child CLI plus explicit nonpromotable legacy compatibility |
| `train.py` | Strict one-GSDiff child CLI plus explicit nonpromotable legacy compatibility |
| `schemas/method-info-v2.schema.json` | Exact Draft 2020-12 method-info schema |
| `docs/experiments/method-registry-v1.md` | Reader-facing scientific bindings, provenance, limits, and migration |

### Task 1: Canonical Registry, Hashes, and Algorithm Seeds

**Files:**

- Create: `gsdiff/experiments/methods.py`
- Create: `tests/experiments/test_methods.py`
- Modify: `configs/protocols/methods-v1.yaml`
- Modify: `gsdiff/experiments/protocol.py:691-739`
- Modify: `gsdiff/experiments/__init__.py`
- Modify: `tests/experiments/test_protocol.py:532-543`

**Interfaces:**

- Consumes: `gsdiff.experiments.identity.canonical_json_bytes`,
  `gsdiff.experiments.identity.sha256_bytes`, and
  `gsdiff.experiments.protocol.load_protocol`.
- Produces:

```python
@dataclass(frozen=True)
class CheckpointRequirement:
    logical_id: str
    sha256: str
    provenance_status: str


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
    checkpoint_requirements: tuple[CheckpointRequirement, ...]
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


def canonical_method_id(method_id: str) -> str:
    raise ValueError


def resolve_method_semantics(
    method_id: str,
    *,
    method_config_id: str,
    base_config: Mapping[str, object],
    measurements_metadata: Mapping[str, object],
    execution_profile: str,
    registry_path: Path = Path("configs/protocols/methods-v1.yaml"),
) -> ResolvedMethod:
    raise ValueError


def derive_algorithm_seed(
    *,
    cell_seed: int,
    dataset_identity_sha256: str,
    method_id: str,
    method_config_sha256: str,
) -> AlgorithmSeed:
    raise ValueError
```

The `raise ValueError` bodies above document fail-closed APIs; the GREEN
implementation replaces them with validated behavior.

- [ ] **Step 1: Write RED registry identity tests**

Add helpers and tests to `tests/experiments/test_methods.py`:

```python
from pathlib import Path

import pytest

from gsdiff.experiments.methods import (
    CANONICAL_METHOD_IDS,
    canonical_method_id,
    resolve_method_semantics,
)

REGISTRY = Path("configs/protocols/methods-v1.yaml")
METHODS = (
    "dgi",
    "static_cs",
    "perframe_cs",
    "tv3d",
    "monin",
    "gidc3dtv",
    "recinr",
    "siren",
    "recinr_se2",
    "gsdiff_tv",
    "gsdiff_diffusion",
)


def _measurements_metadata() -> dict[str, object]:
    return {"H": 32, "W": 32, "T": 4, "K": 128, "holdout_K": 16}


def test_registry_contains_exactly_eleven_canonical_ids() -> None:
    assert CANONICAL_METHOD_IDS == METHODS
    assert len(CANONICAL_METHOD_IDS) == len(set(CANONICAL_METHOD_IDS))


def test_alias_is_input_only() -> None:
    assert canonical_method_id("gsdiff_diff") == "gsdiff_diffusion"
    assert canonical_method_id("gsdiff_diffusion") == "gsdiff_diffusion"
    with pytest.raises(ValueError, match="unknown method"):
        canonical_method_id("gsdiff-admm")


def test_campaign_profile_aliases_reuse_one_method_identity() -> None:
    primary = resolve_method_semantics(
        "dgi",
        method_config_id="default",
        base_config={},
        measurements_metadata=_measurements_metadata(),
        execution_profile="primary-full-v1",
        registry_path=REGISTRY,
    )
    supplement = resolve_method_semantics(
        "dgi",
        method_config_id="default",
        base_config={},
        measurements_metadata=_measurements_metadata(),
        execution_profile="supplement-full-v1",
        registry_path=REGISTRY,
    )
    assert primary.execution_profile == "publication-v1"
    assert supplement.execution_profile == "publication-v1"
    assert primary.method_config_sha256 == supplement.method_config_sha256


def test_locked_pilot_default_normalizes_to_distinct_smoke_config() -> None:
    pilot = resolve_method_semantics(
        "dgi",
        method_config_id="default",
        base_config={},
        measurements_metadata=_measurements_metadata(),
        execution_profile="pilot-smoke-v1",
        registry_path=REGISTRY,
    )
    publication = resolve_method_semantics(
        "dgi",
        method_config_id="default",
        base_config={},
        measurements_metadata=_measurements_metadata(),
        execution_profile="primary-full-v1",
        registry_path=REGISTRY,
    )
    assert pilot.requested_method_config_id == "default"
    assert pilot.method_config_id == "smoke-default-v1"
    assert pilot.execution_profile == "controller-cpu-smoke-v1"
    assert pilot.method_config_sha256 != publication.method_config_sha256


def test_declared_ablation_is_hashed_but_execution_blocked() -> None:
    method = resolve_method_semantics(
        "gsdiff_diffusion",
        method_config_id="ablation-j1-v1",
        base_config={
            "representation": "recinr_se2",
            "solver": "hqs",
            "prior": "diffusion",
            "motion_warmup_fraction": 0.2,
            "temporal_tv_weight": 0.1,
            "gaussian_count": None,
        },
        measurements_metadata=_measurements_metadata(),
        execution_profile="ablation-selection-v1",
        registry_path=REGISTRY,
    )
    assert method.execution_ready is False
    assert method.execution_blockers == (
        "missing-versioned-ablation-native-budgets",
    )
    assert method.selection_eligible is False


def test_tv_and_diffusion_resolve_to_distinct_semantics() -> None:
    common = {"gaussian_count": 1000}
    tv = resolve_method_semantics(
        "gsdiff_tv",
        method_config_id="default",
        base_config=common,
        measurements_metadata=_measurements_metadata(),
        execution_profile="publication-v1",
        registry_path=REGISTRY,
    )
    diffusion = resolve_method_semantics(
        "gsdiff_diff",
        method_config_id="default",
        base_config=common,
        measurements_metadata=_measurements_metadata(),
        execution_profile="publication-v1",
        registry_path=REGISTRY,
    )
    assert tv.semantic_config["solver"]["prior_type"] == "tv"
    assert diffusion.method_id == "gsdiff_diffusion"
    assert diffusion.semantic_config["solver"]["prior_type"] == "diffusion"
    assert tv.method_config_sha256 != diffusion.method_config_sha256
```

- [ ] **Step 2: Run the registry tests and verify RED**

Run:

```powershell
D:\conda\envs\spi\python.exe -m pytest tests/experiments/test_methods.py -q
```

Expected: collection fails because `gsdiff.experiments.methods` does not
exist.

- [ ] **Step 3: Extend the registry document and protocol validator**

Replace each `{id, lane}` entry with the exact keys demonstrated by the
canonical DGI row:

```yaml
id: dgi
lane: analytic
execution_family: baseline
command_template:
  - ${PYTHON}
  - scripts/run_baselines.py
  - --method
  - dgi
  - --dataset
  - ${MEASUREMENTS_PATH}
  - --dataset-identity-sha256
  - ${DATASET_IDENTITY_SHA256}
  - --method-config
  - ${METHOD_CONFIG_PATH}
  - --algorithm-seed
  - ${ALGORITHM_SEED}
  - --device
  - ${DEVICE}
  - --output-dir
  - ${OUTPUT_DIR}
required_child_outputs: [reconstruction.npz, method-info.json]
checkpoints: []
profiles:
  publication-v1:
    method_config_id: default
    publication_eligible: true
    selection_eligible: true
    promotion_eligible: true
    convergence_status: not-applicable
    execution_ready: true
    execution_blockers: []
    semantic_config:
      algorithm: direct-dgi
      native_unit: pass
      native_budget: 1
  controller-cpu-smoke-v1:
    method_config_id: smoke-default-v1
    publication_eligible: false
    selection_eligible: false
    promotion_eligible: false
    convergence_status: smoke-only/not-convergence-assessed
    execution_ready: true
    execution_blockers: []
    semantic_config:
      algorithm: direct-dgi
      native_unit: pass
      native_budget: 1
```

Every remaining row uses the same exact key set. Baseline rows name
`scripts/run_baselines.py`; GSDiff rows name `train.py`; the fourth
`command_template` element (zero-based index 3, immediately after `--method`)
is the row's literal canonical `id`. The validator constructs this expected
tuple from the locked family and ID and compares it field-for-field, so a
copied DGI ID or wrong entry point is rejected.
The diffusion template additionally ends with the literal pair:

```yaml
  - --checkpoint
  - gsdiff-diffusion-prior-v1=${CHECKPOINT:gsdiff-diffusion-prior-v1}
```

Its publication profile has `publication_eligible: false` while provenance is
blocked, and also sets `selection_eligible: false` and
`promotion_eligible: false`, `execution_ready: false`, and exact blockers
`missing-reproducible-checkpoint-locator` and
`missing-checkpoint-training-provenance`. Its verified local checkpoint is
usable only through the nonpromotable smoke profile. Optimizing publication profiles use
`convergence_status: convergence-required`; analytic DGI uses
`not-applicable`. Every publication semantic mapping includes:

```yaml
compute_cap:
  wall_time_seconds: 1800
  peak_vram_bytes: 15032385536
  on_exceed: ineligible-retain-artifacts
```

The registry top level also binds:

```yaml
campaign_execution_profile_aliases:
  primary-full-v1: publication-v1
  supplement-full-v1: publication-v1
  ood-full-v1: publication-v1
  failure-budget-v1: publication-v1
  pilot-smoke-v1: controller-cpu-smoke-v1
  ablation-selection-v1: ablation-selection-v1
pilot_method_config_aliases:
  pilot-smoke-v1:
    default: smoke-default-v1
```

The profile-scoped config alias preserves the locked `pilot-v1` matrix, which
declares `default`, while normalizing it to a separate smoke config identity.
`ResolvedMethod.requested_method_config_id` remains `default`;
`ResolvedMethod.method_config_id` becomes `smoke-default-v1`; only the latter
enters `method_config_sha256`.

The six exact `joint_shortlist` mappings in `ablations-v1.yaml` are repeated
as `ablation-j1-v1` through `ablation-j6-v1` path-free semantic
configurations. They set all eligibility flags false,
`execution_ready: false`, and the sole blocker
`missing-versioned-ablation-native-budgets`. Their axis values and canonical
method IDs must match `ablations-v1.yaml` field-for-field. Task 4 hashes and
validates these identities but refuses to materialize them; it does not invent
an SGD/ADMM/HQS selection budget.

For `gsdiff_diffusion`, use:

```yaml
checkpoints:
  - logical_id: gsdiff-diffusion-prior-v1
    sha256: 667948800911acb9f9a7271e20af5692b0f007007d0fc32a15ac169eba32c5dd
    provenance_status: blocked-missing-training-provenance
```

Bind the exact native budgets:

| ID | Publication semantic values | Smoke semantic values |
| --- | --- | --- |
| `dgi` | `algorithm=direct-dgi`, `native_unit=pass`, `native_budget=1` | Same single pass |
| `static_cs` | `rho=0.5`, `n_admm=150`, `chambolle_iter=100`, lambda grid `[0.001,0.003,0.01,0.03,0.1,0.3,1.0]`, `refit_all_measurements=true` | `n_admm=1`, lambda `[0.001]` |
| `perframe_cs` | `rho=0.5`, `n_admm=120`, `chambolle_iter=100`, same lambda grid, `refit_all_measurements=true` | `n_admm=1`, lambda `[0.001]` |
| `tv3d` | `iterations=500`, `opnorm_iterations=30`, `lambda_xy=[0.003,0.03,0.3]`, `lambda_t=[0.001,0.01,0.1,1.0]`, `refit_all_measurements=true` | `iterations=1`, `opnorm_iterations=30`, first grid pair |
| `monin` | `rho=1.0`, `n_admm=150`, `chambolle_iter=100`, lambda grid `[0.0001,0.0003,0.001,0.003,0.01,0.03]`, `interpolation=bilinear`, `motion_blocks=5`, `polynomial_degree=1` | `n_admm=1`, first lambda, `motion_blocks=4` for the four-frame pilot |
| `gidc3dtv` | `optimizer=adam`, `n_steps=2500`, `lr=0.05`, `betas=[0.5,0.9]`, `eval_every=25`, `xi_xy=[0.003,0.03,0.3]`, `xi_t=[0.01,0.1]` | `n_steps=1`, `eval_every=1`, first grid pair |
| `recinr` | `hidden_dim=32`, `render_layers=3`, `basis=lowrank`, `basis_order=0`, `harmonics=2`, `flow_scale=0.5`, `position_encoding_space=2`, `position_encoding_time=5`, `anneal_fraction=0.6`, `anchor_tau=0.5`, `warm_steps=300`, `flow_steps=400`, `joint_steps=1200`, `lr_start=0.003`, `lr_end=0.001`, `snapshot_every=50`, `node_rule=round-1.7T` | warm/flow/joint each 1; representation values unchanged |
| `siren` | `scene_type=siren`, `w0=8`, `initialization=random`, `solver_type=sgd`, `sgd_steps=4000`, `lr_scene=0.003` | `sgd_steps=1`, `motion_warmup_steps=0` |
| `recinr_se2` | `scene_type=recinr`, `grid_size=20`, `initialization=random`, `solver_type=sgd`, `sgd_steps=3000`, `lr_scene=0.003`, `motion_warmup_steps=500` | `sgd_steps=1`, `motion_warmup_steps=0` |
| `gsdiff_tv` | `scene_type=gaussian`, `gaussian_count=1000`, `initialization=dgi_adaptive`, `solver_type=admm`, `outer_iterations=80`, `inner_iterations=50`, `splitting_warmup_outer=20`, `rho=0.1`, `rho_growth=1.1`, `prior_type=tv`, `tv_variant=tv3d_corrected`, `tv_weight=0.005`, `soft_tv_weight=0.006`, `temporal_tv_weight=0.1`, `lr_scene=0.009`, `lr_motion=0.15`, `motion_warmup_fraction=0.2`, `motion_warmup_outer=16` | outer/inner each 1; both warmups 0 |
| `gsdiff_diffusion` | Same Gaussian/ADMM/rho/LR base, `prior_type=diffusion`, splitting warmup 20, `motion_warmup_fraction=0.0`, `motion_warmup_outer=0`, proximal weight 0.005, soft-TV 0.006, temporal weight 0.05, logical checkpoint, `denoise_steps=1`, `clamp_range=[0.0,1.0]`, `sigma_start=0.3`, `sigma_end=0.05`, `renoise=false`, `ddim_spacing=linear` | outer/inner each 1; both warmups 0; verified local checkpoint only |

The baseline semantic mappings also bind the implementation constants that
were previously hidden in function bodies. CS uses nonnegativity, target
standard-deviation conditioning, and isotropic Chambolle TV. Monin binds
preview blur sigma 1.5 and NCC search radius 12. GIDC binds U-Net channels
`[16,32,64,128,128]`, Adam epsilon `1e-8`, and learning-rate decay
`0.90 ** (step/100)`. Native ReCINR binds output activation `softplus`,
`lam_flow_t=0.5`, `lam_flow_xy=0.2`, `lam_l1=0.05`, `tv_xy=1e-5`,
`lam_tv_canon=1e-4`, and `lam_ttv=0.0`. These values are part of the
path-free semantic hash.

Every GSDiff semantic mapping must also store every constructor value that was
previously an implicit default:

```yaml
motion:
  enable_rotation: true
  polynomial_degree: 1
  enable_affine: false
solver:
  loss_norm: zscore
  lr_motion: 0.15
  tv_weight: 0.005
```

SIREN additionally locks `hidden=128`, `hidden_layers=2`, and random
initialization with no DGI prefit. `recinr_se2` locks `channels=32`,
`render_layers=3`, grid 20, and random initialization with no DGI prefit.
Gaussian rows lock `init_scale=1.5` and `min_scale=0.0`. TV ADMM locks prior
proximal iterations 50. Diffusion locks `renoise=false` and
`ddim_spacing=linear`. SGD controls inherit `use_3dtv=true` and
`temporal_tv_weight=0.05`; the primary TV row overrides temporal weight to
0.1. No strict adapter may fall back to a Python constructor default for a
scientific field.

Update `_validate_methods_registry` to require the new exact key sets, exact
command tokens, exact two outputs, exact checkpoint declaration, all three
normalized method-profile names, the exact campaign/profile and pilot-config
alias maps, and the locked values above. Recompute `protocol_sha256` from
canonical document bytes with the existing `_protocol_sha256` implementation
and replace the YAML value.

- [ ] **Step 4: Implement immutable registry resolution**

In `gsdiff/experiments/methods.py`:

1. define `CANONICAL_METHOD_IDS` in the locked order;
2. load through `load_protocol`;
3. normalize aliases before lookup;
4. normalize a requested campaign execution profile through the exact alias
   mapping and hash the normalized method profile;
5. normalize only the exact pilot config alias above, record requested and
   normalized IDs separately, and otherwise reject a profile whose embedded
   method-config ID does not match the requested ID;
6. for publication/smoke, accept only `gaussian_count` in `base_config`:
   require its locked integer for Gaussian rows, remove `None` for
   non-Gaussian rows so absent and null hash identically, and reject every
   non-null non-Gaussian value and every other override key;
7. for `ablation-selection-v1`, require one of the six declared joint
   method-config IDs and an exact field-for-field `base_config`; return it
   blocked and noneligible;
8. require Gaussian `gaussian_count` and reject it for every non-Gaussian
   method;
9. reject native `recinr` if `base_config` contains generic `scene` or
   `solver`;
10. freeze nested mappings/lists as `MappingProxyType` and tuples;
11. hash canonical JSON containing `method_id`, `method_config_id`,
   `execution_profile`, and semantic config;
12. return exactly two required child outputs.

Use these exact hash bytes:

```python
payload = {
    "method_id": canonical_id,
    "method_config_id": method_config_id,
    "execution_family": execution_family,
    "execution_profile": execution_profile,
    "command_template": list(command_template),
    "semantic_config": thaw_json(frozen_semantics),
    "checkpoint_requirements": checkpoint_payload,
    "required_child_outputs": [
        "reconstruction.npz",
        "method-info.json",
    ],
    "profile_policy": {
        "publication_eligible": publication_eligible,
        "selection_eligible": selection_eligible,
        "promotion_eligible": promotion_eligible,
        "convergence_status": convergence_status,
        "execution_ready": execution_ready,
        "execution_blockers": list(execution_blockers),
    },
}
method_config_sha256 = sha256_bytes(canonical_json_bytes(payload))
```

`thaw_json` converts mapping proxies to dictionaries and tuples to lists
without changing scalar values.

- [ ] **Step 5: Verify GREEN for registry identity**

Run:

```powershell
D:\conda\envs\spi\python.exe -m pytest tests/experiments/test_methods.py tests/experiments/test_protocol.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Write RED path, inactive-field, and seed tests**

Add:

```python
def test_absolute_path_cannot_enter_semantic_identity() -> None:
    with pytest.raises(ValueError, match="path"):
        resolve_method_semantics(
            "dgi",
            method_config_id="default",
            base_config={"output_dir": r"D:\leak"},
            measurements_metadata=_measurements_metadata(),
            execution_profile="publication-v1",
            registry_path=REGISTRY,
        )


@pytest.mark.parametrize("method_id", [m for m in METHODS if not m.startswith("gsdiff_")])
def test_gaussian_count_is_rejected_when_inactive(method_id: str) -> None:
    with pytest.raises(ValueError, match="gaussian_count"):
        resolve_method_semantics(
            method_id,
            method_config_id="default",
            base_config={"gaussian_count": 1000},
            measurements_metadata=_measurements_metadata(),
            execution_profile="publication-v1",
            registry_path=REGISTRY,
        )


def test_null_inactive_gaussian_count_hashes_like_absent() -> None:
    absent = resolve_publication("siren", base_config={})
    null = resolve_publication("siren", base_config={"gaussian_count": None})
    assert absent.method_config_sha256 == null.method_config_sha256


def test_algorithm_seed_matches_locked_domain_derivation() -> None:
    result = derive_algorithm_seed(
        cell_seed=42,
        dataset_identity_sha256="1" * 64,
        method_id="gsdiff_tv",
        method_config_sha256="2" * 64,
    )
    payload = canonical_json_bytes({
        "domain": "algorithm-seed-v1",
        "cell_seed": 42,
        "dataset_identity_sha256": "1" * 64,
        "method_id": "gsdiff_tv",
        "method_config_sha256": "2" * 64,
    })
    digest = hashlib.sha256(payload).hexdigest()
    assert result.derivation_sha256 == digest
    assert result.seed_u32 == int.from_bytes(bytes.fromhex(digest)[:4], "big")
```

- [ ] **Step 7: Run seed tests and verify RED**

Run:

```powershell
D:\conda\envs\spi\python.exe -m pytest tests/experiments/test_methods.py -q
```

Expected: the new validation and seed assertions fail for their intended
missing behavior.

- [ ] **Step 8: Implement strict field validation and seed derivation**

Reject strings that are absolute Windows paths, absolute POSIX paths,
contain a staging UUID token, or use unknown `${...}` tokens inside semantic
config. Validate SHA-256 inputs with `fullmatch("[0-9a-f]{64}")`. Require
`cell_seed` to be a nonnegative integer and reject booleans.

Implement the seed derivation exactly as asserted in Step 6. Export the public
types/functions from `gsdiff/experiments/__init__.py`.

- [ ] **Step 9: Run Task 1 gates**

Run:

```powershell
D:\conda\envs\spi\python.exe -m pytest tests/experiments/test_methods.py tests/experiments/test_protocol.py tests/experiments/test_identity.py -q
D:\conda\envs\spi\python.exe -m compileall -q gsdiff/experiments
git diff --check
```

Expected: all pytest tests pass, compileall exits 0, and diff check is clean.

- [ ] **Step 10: Commit Task 1**

```powershell
git add -- configs/protocols/methods-v1.yaml gsdiff/experiments/methods.py gsdiff/experiments/protocol.py gsdiff/experiments/__init__.py tests/experiments/test_methods.py tests/experiments/test_protocol.py
git commit -m "feat: define canonical experiment method registry"
```

### Task 2: Raw Blind Objective and Versioned Child Artifacts

**Files:**

- Create: `gsdiff/experiments/objectives.py`
- Create: `gsdiff/experiments/child_outputs.py`
- Create: `schemas/method-info-v2.schema.json`
- Create: `tests/experiments/test_objectives.py`
- Create: `tests/experiments/test_child_outputs.py`
- Modify: `gsdiff/experiments/__init__.py`

**Interfaces:**

- Consumes: `ResolvedMethod`, `AlgorithmSeed`, `SPIAcquisitionData`, the
  existing internal serialization primitives
  `gsdiff.data._artifact_io.write_npz` and
  `gsdiff.data._artifact_io.read_npz_members`, array descriptor helpers,
  canonical JSON helpers, and atomic byte writes. Task 2 does not claim these
  private primitives are public `gsdiff.data.artifacts` exports.
- Produces:

```python
@dataclass(frozen=True)
class BlindObjective:
    formula_id: str
    numerator: float
    denominator: float
    value: float


@dataclass(frozen=True, kw_only=True)
class MethodChildResult:
    method_id: str
    reconstruction: np.ndarray
    estimated_motion_trajectory: np.ndarray | None
    dgi: np.ndarray | None
    info: Mapping[str, object]
    history: tuple[Mapping[str, object], ...]


def heldout_normalized_l2(
    reconstruction: np.ndarray,
    patterns: np.ndarray,
    measurements: np.ndarray,
    frame_indices: np.ndarray,
) -> BlindObjective:
    raise ValueError


def select_by_heldout_normalized_l2(
    candidates: Sequence[object],
    run_candidate: Callable[[object], np.ndarray],
    *,
    patterns: np.ndarray,
    measurements: np.ndarray,
    frame_indices: np.ndarray,
) -> tuple[object, np.ndarray, tuple[Mapping[str, object], ...]]:
    raise ValueError


def write_method_child_outputs_v2(
    output_dir: Path,
    *,
    method: ResolvedMethod,
    acquisition: SPIAcquisitionData,
    measurements_file_sha256: str,
    algorithm_seed: AlgorithmSeed,
    result: MethodChildResult,
    child_started_at_utc: str,
    child_finished_at_utc: str,
) -> Mapping[str, str]:
    raise ValueError


def validate_method_child_outputs_v2(
    output_dir: Path,
    *,
    expected_method: ResolvedMethod,
    expected_dataset_identity_sha256: str,
    expected_measurements_file_sha256: str,
    expected_algorithm_seed: AlgorithmSeed,
) -> Mapping[str, str]:
    raise ValueError
```

- [ ] **Step 1: Write RED objective tests**

Add numerical and validation tests:

```python
def test_raw_objective_matches_physical_float64_formula() -> None:
    reconstruction = np.array([[[1, 2], [3, 4]], [[4, 3], [2, 1]]], dtype=np.float32)
    patterns = np.array([
        [[1, 0], [0, 1]],
        [[0, 1], [1, 0]],
    ], dtype=np.float32)
    frame_indices = np.array([0, 1], dtype=np.int64)
    measurements = np.array([6.0, 5.0], dtype=np.float32)
    result = heldout_normalized_l2(
        reconstruction, patterns, measurements, frame_indices
    )
    predicted = np.array([5.0, 5.0], dtype=np.float64)
    expected_numerator = np.linalg.norm(predicted - measurements.astype(np.float64))
    expected_denominator = max(np.linalg.norm(measurements.astype(np.float64)), 1e-12)
    assert result.formula_id == "heldout-normalized-l2-v1"
    assert result.numerator == pytest.approx(expected_numerator)
    assert result.denominator == pytest.approx(expected_denominator)
    assert result.value == pytest.approx(expected_numerator / expected_denominator)


def test_zero_measurement_norm_uses_locked_floor() -> None:
    result = heldout_normalized_l2(
        np.ones((1, 1, 1)),
        np.ones((1, 1, 1)),
        np.zeros(1),
        np.zeros(1, dtype=np.int64),
    )
    assert result.denominator == 1e-12
    assert result.value == pytest.approx(1e12)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("bad-reconstruction-shape", "reconstruction"),
        ("bad-pattern-shape", "patterns"),
        ("bad-measurement-shape", "measurements"),
        ("noninteger-indices", "frame_indices"),
        ("out-of-range-index", "frame_indices"),
        ("nonfinite", "finite"),
    ],
)
def test_objective_rejects_invalid_inputs(mutation: str, message: str) -> None:
    reconstruction, patterns, measurements, frame_indices = valid_objective_arrays()
    reconstruction, patterns, measurements, frame_indices = mutate_objective_arrays(
        mutation, reconstruction, patterns, measurements, frame_indices
    )
    with pytest.raises((TypeError, ValueError), match=message):
        heldout_normalized_l2(
            reconstruction, patterns, measurements, frame_indices
        )
```

Add a two-candidate fixture whose z-scored ranking is the reverse of raw L2,
and assert the helper chooses the raw-L2 winner.

- [ ] **Step 2: Run objective tests and verify RED**

Run:

```powershell
D:\conda\envs\spi\python.exe -m pytest tests/experiments/test_objectives.py -q
```

Expected: collection fails because `objectives.py` does not exist.

- [ ] **Step 3: Implement the raw objective and candidate selector**

Implementation requirements:

```python
selected = reconstruction[frame_indices].astype(np.float64, copy=False)
physical_patterns = patterns.astype(np.float64, copy=False)
predicted = np.einsum("khw,khw->k", physical_patterns, selected)
target = measurements.astype(np.float64, copy=False)
numerator = float(np.linalg.norm(predicted - target))
denominator = max(float(np.linalg.norm(target)), 1e-12)
```

Validate exact ranks and matching dimensions before indexing. Require finite
arrays and integer frame indices in `[0, T)`. Candidate selection preserves
declared candidate order for ties and records only:

```python
{
    "candidate": 0.001,
    "formula_id": "heldout-normalized-l2-v1",
    "numerator": 1.0,
    "denominator": 8.0,
    "value": 0.125,
}
```

The three numbers above illustrate the exact finite-float fields; each row
stores its computed numerator, denominator, and quotient. The selector
signature has no `gt_frames` argument and imports no evaluator.

- [ ] **Step 4: Verify GREEN for objective tests**

Run:

```powershell
D:\conda\envs\spi\python.exe -m pytest tests/experiments/test_objectives.py -q
```

Expected: all objective tests pass.

- [ ] **Step 5: Write RED v2 artifact tests**

Tests must assert:

1. child output contains exactly `reconstruction.npz` and
   `method-info.json`;
2. `reconstruction-v2` makes DGI and motion trajectory independently
   optional;
3. no execution class, truth access, promotion flag, metrics, PSNR, SSIM, GT
   fields, NaN, Infinity, extra JSON keys, or extra ZIP members are accepted;
4. bounded history contains at most 21 samples and records the native sampling
   rule;
5. method ID, dataset identity, method-config hash, and algorithm seed
   cross-match the parent request;
6. a child cannot pre-create `metrics.json`, `stdout.log`, or `stderr.log`;
7. output writing is atomic and an interrupted first write cannot validate as
   complete.

Use this no-motion assertion:

```python
result = MethodChildResult(
    method_id="dgi",
    reconstruction=np.ones((4, 32, 32), dtype=np.float32),
    estimated_motion_trajectory=None,
    dgi=np.ones((32, 32), dtype=np.float32),
    info={
        "parameter_count": 0,
        "native_iteration_unit": "pass",
        "native_iteration_budget": 1,
        "convergence_status": "not-applicable",
        "selected_hyperparameters": None,
        "selection": None,
        "checkpoint_hashes": [],
    },
    history=(),
)
hashes = write_method_child_outputs_v2(
    output_dir,
    method=resolved_dgi,
    acquisition=acquisition,
    measurements_file_sha256="a" * 64,
    algorithm_seed=algorithm_seed,
    result=result,
    child_started_at_utc="2026-07-28T00:00:00Z",
    child_finished_at_utc="2026-07-28T00:00:01Z",
)
assert set(hashes) == {"reconstruction.npz", "method-info.json"}
loaded = load_reconstruction_v2(output_dir / "reconstruction.npz")
assert loaded.estimated_motion_trajectory is None
```

- [ ] **Step 6: Run v2 artifact tests and verify RED**

Run:

```powershell
D:\conda\envs\spi\python.exe -m pytest tests/experiments/test_child_outputs.py -q
```

Expected: collection fails because the v2 writer does not exist.

- [ ] **Step 7: Define exact v2 schemas and writer**

`reconstruction-v2` NPZ metadata exact keys:

```text
schema
dataset_identity_sha256
method_id
optional_arrays
array_descriptors
```

Required arrays are `reconstruction`, `frame_indices`, and `time_grid`.
Optional arrays are `dgi` and `estimated_motion_trajectory`.

`method-info-v2` exact top-level keys:

```text
schema
method_id
method_config_id
execution_family
execution_profile
dataset_identity_sha256
measurements_file_sha256
method_config_sha256
semantic_config
algorithm_seed
parameter_count
native_iteration
warmup
selected_hyperparameters
selection
convergence
checkpoints
motion_estimate
reconstruction
child_timing
```

Use Draft 2020-12 with `additionalProperties: false` at every object level.
`algorithm_seed` has exactly `domain`, `derivation_sha256`, and `seed_u32`.
`reconstruction` binds the file SHA-256 and exact array descriptors.
`convergence` embeds no more than 21 JSON-native history samples and records
`sampling_policy`, `observed_count`, and `serialized_count`.
For 21 or more observations, serialize indices
`floor(i * (observed_count - 1) / 20)` for integer `i=0..20`; for fewer
observations, serialize every row. Only a profile whose convergence status is
`convergence-required` can pass publication convergence with 21 or more
observations. Analytic DGI uses `not-applicable`, permits zero history rows,
and is not subjected to that threshold. Smoke explicitly records
`smoke-only/not-convergence-assessed`.
`child_timing` contains exact RFC 3339 UTC `started_at` and `finished_at`
strings and a nonnegative finite `elapsed_seconds` derived from them.
`motion_estimate` has exact keys `present` and `native_model`, and its presence
must match the optional trajectory member in `reconstruction-v2`.

Do not reuse the legacy substring blacklist. Validate exact schema fields and
explicitly reject evaluator-only fields through the v2 schema. Preserve v1
writers/loaders unchanged.

- [ ] **Step 8: Implement v2 loading and parent cross-validation**

The validator must:

- reject links and nonfiles;
- require exact two-file inventory;
- verify both hashes after loading;
- validate JSON with `Draft202012Validator`;
- reject duplicate JSON keys during parsing;
- reject nonfinite JSON numbers;
- verify reconstruction metadata against method info;
- verify all parent-known fields against `ResolvedMethod`, dataset identity,
  and `AlgorithmSeed`;
- verify optional-array flags against exact ZIP members;
- verify every array descriptor and expected shape.

- [ ] **Step 9: Run Task 2 gates**

Run:

```powershell
D:\conda\envs\spi\python.exe -m pytest tests/experiments/test_objectives.py tests/experiments/test_child_outputs.py tests/data/test_artifacts.py -q
D:\conda\envs\spi\python.exe -m compileall -q gsdiff/experiments
git diff --check
```

Expected: all tests pass, including unchanged legacy artifact tests.

- [ ] **Step 10: Commit Task 2**

```powershell
git add -- gsdiff/experiments/objectives.py gsdiff/experiments/child_outputs.py gsdiff/experiments/__init__.py schemas/method-info-v2.schema.json tests/experiments/test_objectives.py tests/experiments/test_child_outputs.py
git commit -m "feat: add blind objective and child artifact v2"
```

### Task 3: GT-Free Baseline Adapters

**Files:**

- Create: `gsdiff/experiments/adapters.py`
- Create: `tests/experiments/test_baseline_adapters.py`
- Modify: `gsdiff/baselines/common.py:103-148`
- Modify: `gsdiff/baselines/cs.py:1-94`
- Modify: `gsdiff/baselines/gidc.py:89-187`
- Modify: `gsdiff/baselines/monin.py:135-175`
- Modify: `gsdiff/baselines/recinr.py:117-245`
- Modify: `gsdiff/baselines/tv3d.py:100-140`
- Modify: `gsdiff/baselines/inr.py` only if exact registry parameters cannot be
  passed without changing its existing constructor
- Modify: `gsdiff/experiments/__init__.py`

**Interfaces:**

- Consumes: `ResolvedMethod`, `AlgorithmSeed`, `MethodChildResult`,
  `SPIAcquisitionData`, and the raw objective helpers.
- Produces:

```python
BASELINE_METHOD_IDS = (
    "dgi",
    "static_cs",
    "perframe_cs",
    "tv3d",
    "monin",
    "gidc3dtv",
    "recinr",
)


def run_baseline_method(
    method: ResolvedMethod,
    acquisition: SPIAcquisitionData,
    *,
    algorithm_seed: AlgorithmSeed,
    device: str,
) -> MethodChildResult:
    raise ValueError
```

Strict baseline cores accept only acquisition arrays, resolved semantic
values, algorithm seed, and runtime device. Legacy wrappers may still adapt
unblinded compatibility objects, but strict dispatch never calls them.

- [ ] **Step 1: Write RED adapter contract tests**

Build a real `SPIAcquisitionData` fixture with finite arrays, `T=5`, uniform
frame assignment, and a distinct holdout set. Parametrize all seven baseline
IDs with `controller-cpu-smoke-v1` and assert:

```python
@pytest.mark.parametrize("method_id", BASELINE_METHOD_IDS)
def test_smoke_baseline_accepts_blind_acquisition_only(
    method_id: str,
    blind_acquisition: SPIAcquisitionData,
) -> None:
    method = resolve_smoke(method_id, blind_acquisition)
    result = run_baseline_method(
        method,
        blind_acquisition,
        algorithm_seed=derive_for(method, blind_acquisition),
        device="cpu",
    )
    assert result.method_id == method_id
    assert result.reconstruction.shape == (
        blind_acquisition.T,
        blind_acquisition.H,
        blind_acquisition.W,
    )
    assert np.isfinite(result.reconstruction).all()
    forbidden = json.dumps(result.info, sort_keys=True).lower()
    assert "psnr" not in forbidden
    assert "ssim" not in forbidden
    assert "ground_truth" not in forbidden
    assert "gt_" not in forbidden
```

For methods whose tiny real solver is expensive, retain real array/operator
code and reduce only the registry smoke budget. Do not replace the solver with
a mocked reconstruction.

- [ ] **Step 2: Run adapter tests and verify RED**

Run:

```powershell
D:\conda\envs\spi\python.exe -m pytest tests/experiments/test_baseline_adapters.py -q
```

Expected: collection fails because `run_baseline_method` does not exist.

- [ ] **Step 3: Replace legacy selection helpers with the locked objective**

In `gsdiff/baselines/common.py`:

- remove `_zscore_np` from selection;
- make `holdout_residual` a compatibility alias that calls
  `heldout_normalized_l2(...).value`;
- remove `gt_frames` from `select_by_holdout`;
- remove evaluator imports and PSNR columns from selection tables;
- fail closed when holdout arrays are absent.

The compatibility-only `evaluate_video` symbol must not be imported at module
scope by `cs.py`, `gidc.py`, `monin.py`, or `tv3d.py`. Move each such import
inside the corresponding legacy evaluation wrapper (or remove it where dead)
so importing the strict adapter closure does not invoke
`common.__getattr__` and therefore does not import the excluded
`gsdiff.baselines._evaluation` module. Strict solver cores must have no
reference to `evaluate_video`.

Keep training-data normalization internal to the baseline solvers; it is not a
selection signal. Every solver that conditions measurements must invert that
conditioning on its candidate and final reconstruction before the raw
physical held-out objective or artifact writing. Tests must fail if a
normalized iterate is compared directly with raw held-out measurements.

- [ ] **Step 4: Refactor DGI, TV3D, GIDC, and native ReCINR strict cores**

Implement strict functions with explicit semantic configuration:

```python
run_dgi(acquisition, semantic_config, algorithm_seed, device)
run_tv3d(acquisition, semantic_config, algorithm_seed, device)
run_gidc3dtv(acquisition, semantic_config, algorithm_seed, device)
run_recinr(acquisition, semantic_config, algorithm_seed, device)
```

Requirements:

- DGI returns no motion trajectory and parameter count 0.
- TV3D uses 30 operator-norm iterations and the algorithm seed instead of
  hard-coded seed 0.
- GIDC uses exactly 2500 publication steps or one smoke step, exact candidate
  grids, Adam `lr=0.05`, betas `(0.5, 0.9)`, and the algorithm seed.
- ReCINR uses `round(1.7*T)` nodes, exact registry curriculum, and algorithm
  seed.
- GIDC and ReCINR use the raw held-out objective for candidate/snapshot
  selection.
- GIDC serializes every `{xi_xy, xi_t, snapshot_step}` trial in registry-xi
  and ascending 1-based completed-update order: 600 publication rows and one
  smoke row. ReCINR serializes every `{snapshot_step}` trial using the 1-based
  global step including warm-up: 25 publication rows and one smoke row.
- Each method chooses the strict first minimum, keeps only the current best
  reconstruction array, and marks the selected run's unique producing step
  with `reconstruction_source: true`; bounded history must retain that row.
- GIDC selected hyperparameters remain the two-xi projection. ReCINR selected
  hyperparameters remain null even though its snapshot selection is non-null.
- None of these four methods returns a fabricated trajectory.
- `info` includes exact native unit/budget, parameter count, convergence
  status, selected hyperparameters, selection table or `None`, and checkpoint
  hashes `[]`.

`parameter_count` is the exact count of native scalar degrees of freedom
fitted or optimized from measurements, excluding selected hyperparameters:

| Method | Exact count |
| --- | --- |
| DGI | `0` |
| static CS | `H * W` |
| per-frame CS | `T * H * W` |
| TV3D | `T * H * W` |
| Monin | `H * W + 2 * polynomial_degree` |
| GIDC/ReCINR and GSDiff-family methods | Sum of unique trainable tensor elements actually passed to the optimizer |

The Monin count represents one canonical image and two coefficients per
anchored polynomial degree; the reference-block gauge fits no intercept and
there is no rotation degree of freedom. A frozen diffusion prior is excluded
from GSDiff counts. Add exact formula assertions for every
classical method and direct optimizer-parameter count assertions for stochastic
methods.

Use a scoped RNG context that saves/restores Python, NumPy, and Torch RNG
states. Seed all three from `AlgorithmSeed.seed_u32`; do not mutate acquisition
seed state or leak a changed global RNG state to the caller.

- [ ] **Step 5: Write RED CS and TV3D refit tests**

Use a small real operator and a spy around only the expensive `admm_tv` kernel.
The test must prove call data, not merely call count:

```python
def test_static_cs_selects_on_train_then_refits_all_measurements(
    blind_acquisition: SPIAcquisitionData,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_rows: list[int] = []
    real = cs.admm_tv

    def recording_admm(A, y, H, W, lam, **kwargs):
        observed_rows.append(int(A.shape[0]))
        return real(A, y, H, W, lam, **kwargs)

    monkeypatch.setattr(cs, "admm_tv", recording_admm)
    method = resolve_smoke("static_cs", blind_acquisition)
    run_baseline_method(
        method,
        blind_acquisition,
        algorithm_seed=derive_for(method, blind_acquisition),
        device="cpu",
    )
    assert observed_rows[:-1] == [blind_acquisition.K]
    assert observed_rows[-1] == blind_acquisition.K + blind_acquisition.holdout_K
```

Add the corresponding per-frame assertion: candidate fits use each frame's
training rows, while the final selected-lambda refit partitions the concatenated
train+holdout measurements by frame.

For TV3D, spy on `_chambolle_pock` and record `op.M` for each candidate and
the final call. Candidate operators must have `acquisition.K` rows. The final
operator must be newly constructed from concatenated train and holdout arrays
and have `acquisition.K + acquisition.holdout_K` rows. The test must use a
distinct external holdout set so an all-true mask on the training-only
operator cannot pass.

- [ ] **Step 6: Run refit tests and verify RED**

Run:

```powershell
D:\conda\envs\spi\python.exe -m pytest tests/experiments/test_baseline_adapters.py -k "cs or tv3d_refit" -q
```

Expected: tests fail because current CS returns the selected training fit
without an all-measurement refit and current TV3D's final all-true mask still
uses only the training operator.

- [ ] **Step 7: Implement CS and TV3D all-measurement refit**

For both CS methods:

1. require a distinct holdout set;
2. fit every locked candidate on training arrays only;
3. select with raw held-out normalized L2;
4. concatenate train and holdout patterns, measurements, and frame indices;
5. run one final solve with the selected lambda on concatenated arrays;
6. return the final all-measurement reconstruction while preserving the
   pre-refit selection table.

Do not rescore the refitted reconstruction on the same holdout and overwrite
the pre-refit selection value.

For TV3D, select the `(lambda_xy, lambda_t)` pair using only the training
operator and the raw physical holdout objective. Then concatenate training and
holdout patterns, measurements, and frame indices; construct a new all-data
operator; recompute its measurement conditioning, operator norm, and warm
start; and run one final solve with the selected pair. Convert any internally
conditioned candidate and final iterate back to physical measurement scale
before applying the raw objective or returning the result. Preserve the
pre-refit selection table and do not treat an all-true mask on the
training-only operator as an all-measurement refit. Static CS uses one global
conditioning scale; per-frame CS deconditions each frame with the exact scale
used for that frame's solve.

- [ ] **Step 8: Write RED Monin trajectory tests**

Assert the strict Monin result contains a measurement-derived trajectory:

```python
result = run_baseline_method(
    resolve_smoke("monin", blind_acquisition),
    blind_acquisition,
    algorithm_seed=algorithm_seed,
    device="cpu",
)
assert result.estimated_motion_trajectory is not None
assert result.estimated_motion_trajectory.shape == (blind_acquisition.T, 3)
assert np.isfinite(result.estimated_motion_trajectory).all()
assert np.all(result.estimated_motion_trajectory[:, 2] == 0)
assert "velocity_error" not in result.info
assert "gt_velocity" not in result.info
```

Also assert DGI, both CS variants, TV3D, GIDC, and native ReCINR return
`estimated_motion_trajectory is None`.

- [ ] **Step 9: Run trajectory tests and verify RED**

Run:

```powershell
D:\conda\envs\spi\python.exe -m pytest tests/experiments/test_baseline_adapters.py -k "trajectory or monin" -q
```

Expected: current Monin accesses GT velocity and no optional-trajectory strict
result exists.

- [ ] **Step 10: Implement Monin strict output and canonical dispatch**

Keep its measurement-only preview/NCC/polyfit trajectory. For a publication
dataset require `T >= 5`; for the `T=4` pilot smoke profile resolve
`motion_blocks=4` explicitly. Form a `[T,3]` trajectory whose first two columns
are the estimated translation and whose rotation column is zero because
translation-only is the method's native model. Remove GT velocity and all
trajectory error calculations from strict output.

Implement `run_baseline_method` as an exact dictionary dispatch over the seven
IDs. Reject a GSDiff-family or unknown method before importing a baseline
module.

- [ ] **Step 11: Add deterministic RNG and import-capability tests**

Tests must prove:

- two runs with the same algorithm seed produce the same smoke result;
- changing the algorithm seed changes stochastic GIDC/ReCINR initialization;
- caller Python/NumPy/Torch RNG states are restored;
- importing `gsdiff.experiments.adapters` and running every strict baseline
  does not import `gsdiff.evaluation` or `gsdiff.baselines._evaluation`;
- importing `gsdiff.experiments.adapters` succeeds from a temporary strict
  source snapshot in which `gsdiff/baselines/_evaluation.py` and the other
  declared evaluator/truth files are absent; this test must exercise imports of
  `cs`, `gidc`, `monin`, and `tv3d`, not merely inspect `sys.modules`;
- the strict function signatures contain no `truth`, `gt`, or arbitrary
  `**kwargs`.

- [ ] **Step 12: Run Task 3 gates**

Run:

```powershell
D:\conda\envs\spi\python.exe -m pytest tests/experiments/test_baseline_adapters.py tests/experiments/test_objectives.py tests/experiments/test_child_outputs.py -q
D:\conda\envs\spi\python.exe -m pytest tests/data/test_artifacts.py -q
D:\conda\envs\spi\python.exe -m compileall -q gsdiff/baselines gsdiff/experiments
git diff --check
```

Expected: all tests pass and legacy v1 artifact tests remain green.

- [ ] **Step 13: Commit Task 3**

```powershell
git add -- gsdiff/experiments/adapters.py gsdiff/experiments/__init__.py gsdiff/baselines/common.py gsdiff/baselines/cs.py gsdiff/baselines/gidc.py gsdiff/baselines/monin.py gsdiff/baselines/recinr.py gsdiff/baselines/tv3d.py gsdiff/baselines/inr.py tests/experiments/test_baseline_adapters.py
git commit -m "refactor: make baseline adapters truth free"
```

If `gsdiff/baselines/inr.py` is unchanged, omit it from `git add`.

### Task 4: Strict GSDiff Adapter and Separate Motion Warmup

**Files:**

- Create: `gsdiff/experiments/gsdiff_adapter.py`
- Create: `tests/experiments/test_gsdiff_adapter.py`
- Modify: `gsdiff/experiments/adapters.py`
- Modify: `gsdiff/solver/admm.py:25-187`
- Modify: `tests/solver/test_gradient_groups.py`
- Modify: `train.py:193-539`
- Modify: `gsdiff/experiments/__init__.py`

**Interfaces:**

- Consumes: `ResolvedMethod`, `AlgorithmSeed`, `SPIAcquisitionData`,
  `MethodChildResult`, existing scene/motion/forward/prior/solver classes, and
  a runtime logical-checkpoint mapping.
- Produces:

```python
GSDIFF_METHOD_IDS = (
    "siren",
    "recinr_se2",
    "gsdiff_tv",
    "gsdiff_diffusion",
)


def run_gsdiff_method(
    method: ResolvedMethod,
    acquisition: SPIAcquisitionData,
    *,
    algorithm_seed: AlgorithmSeed,
    checkpoint_paths: Mapping[str, Path],
    device: str,
) -> MethodChildResult:
    raise ValueError


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

- [ ] **Step 1: Write RED ADMM warmup tests**

Extend `tests/solver/test_gradient_groups.py` with a tiny scene and motion
parameter fixture. Test separate semantics:

```python
solver = make_admm_solver(
    splitting_warmup_outer=0,
    motion_warmup_outer=1,
    n_outer=2,
    n_inner=1,
)
scene_before = clone_parameters(solver._scene_params)
motion_before = clone_parameters(solver._motion_params)
first = solver.step()
assert_parameters_equal(scene_before, solver._scene_params)
assert_parameters_changed(motion_before, solver._motion_params)
assert first["in_motion_warmup"] is True
assert first["in_splitting_warmup"] is False
second = solver.step()
assert_parameters_changed(scene_before, solver._scene_params)
assert second["in_motion_warmup"] is False
```

Add a second test with `splitting_warmup_outer=1` and
`motion_warmup_outer=0` to prove the existing splitting transition remains
independent. Add a constructor test that rejects either warmup count above
`n_outer`.

- [ ] **Step 2: Run warmup tests and verify RED**

Run:

```powershell
D:\conda\envs\spi\python.exe -m pytest tests/solver/test_gradient_groups.py -k "motion_warmup or splitting_warmup" -q
```

Expected: tests fail because ADMM currently exposes one `n_warmup` meaning
only splitting warmup.

- [ ] **Step 3: Implement separate ADMM/HQS warmups**

Rename the semantic constructor field to `splitting_warmup_outer` and add
`motion_warmup_outer`. Retain `n_warmup` only as a deprecated compatibility
keyword that cannot be supplied together with the new name.

During each theta inner step:

```python
(loss_d + loss_tv + loss_c).backward()
if in_motion_warmup:
    for parameter in self._scene_params:
        if parameter.grad is not None:
            parameter.grad.zero_()
```

Do not zero motion gradients. Do not add optimizer steps or outer iterations.
The z/u/rho behavior depends only on splitting warmup and HQS state. Report
both `in_motion_warmup` and `in_splitting_warmup`.

- [ ] **Step 4: Verify GREEN for solver tests**

Run:

```powershell
D:\conda\envs\spi\python.exe -m pytest tests/solver/test_gradient_groups.py tests/solver/test_sgd_parameters.py -q
```

Expected: all solver tests pass.

- [ ] **Step 5: Write RED canonical GSDiff binding tests**

For each GSDiff ID, resolve the smoke profile and inspect a real constructed
adapter using tiny acquisition arrays. Assert:

- `siren`: SIREN, `w0=8`, random, SGD one smoke step;
- `recinr_se2`: ReCINR canonical, grid 20, random, SGD one smoke step;
- `gsdiff_tv`: Gaussian count 1000, ADMM one-by-one smoke, corrected TV, no
  checkpoint;
- `gsdiff_diffusion`: Gaussian count 1000, ADMM, diffusion prior, exactly one
  logical checkpoint and matching hash;
- all outputs persist their canonical method ID rather than solver name;
- `gsdiff_tv` and `gsdiff_diffusion` never collapse to `gsdiff-admm`;
- estimated SE(2) trajectory is real `[T,3]`;
- strict output contains no evaluator fields.

Use `gsdiff_tv` for the real one-step CPU optimization. For diffusion, load the
real local checkpoint in a separate construction/hash test and verify the
declared prior fields without running a publication optimization. Do not add a
test-only scientific override to production configuration.

- [ ] **Step 6: Run GSDiff tests and verify RED**

Run:

```powershell
D:\conda\envs\spi\python.exe -m pytest tests/experiments/test_gsdiff_adapter.py -q
```

Expected: collection fails because the strict GSDiff adapter does not exist.

- [ ] **Step 7: Extract strict GSDiff execution from `train.py`**

Move measurement-only construction and optimization into
`run_gsdiff_method`. It must:

1. accept the already loaded acquisition object;
2. seed Python, NumPy, and Torch from `AlgorithmSeed` in a scoped context;
3. construct only the scene, motion, solver, and prior declared by
   `ResolvedMethod`;
4. use held-out arrays only for the locked measurement objective;
5. never load truth, call `evaluate`, save figures, or write metrics;
6. produce bounded scalar history with no embedded video;
7. calculate parameter count from trainable model/scene/motion parameters;
8. return the canonical method ID and real estimated trajectory;
9. use the runtime checkpoint mapping only after matching logical ID and
   SHA-256;
10. reject any unused or extra checkpoint path.

Keep legacy result/figure/evaluation code in a clearly named compatibility
function. The strict adapter must not import that function.

- [ ] **Step 8: Bind motion warmup counts**

Resolve:

```python
motion_warmup_outer = math.ceil(
    semantic_config["solver"]["motion_warmup_fraction"]
    * semantic_config["solver"]["outer_iterations"]
)
```

Require the resolved integer to equal the registry's stored
`motion_warmup_outer`. For `gsdiff_tv`, publication values are fraction 0.2,
outer 80, resolved 16. Keep splitting warmup 20 as an independent field.
SGD uses exact `motion_warmup_steps`; it does not use the ADMM rounding rule.

- [ ] **Step 9: Implement canonical family dispatch**

`run_canonical_method` dispatches baseline IDs to `run_baseline_method` and
GSDiff IDs to `run_gsdiff_method`. It rejects a checkpoint mapping for any
method with no checkpoint requirement and rejects a missing mapping for
diffusion.

- [ ] **Step 10: Add checkpoint failure tests**

Test exact pre-execution failures for:

- missing logical ID;
- extra logical ID;
- nonexistent path;
- directory instead of regular file;
- symlink/reparse point;
- SHA mismatch;
- correct local file hash but publication provenance still marked blocked.

The correct local hash permits only the nonpromotable smoke profile.

- [ ] **Step 11: Run Task 4 gates**

Run:

```powershell
D:\conda\envs\spi\python.exe -m pytest tests/solver tests/experiments/test_gsdiff_adapter.py tests/experiments/test_baseline_adapters.py tests/experiments/test_methods.py -q
D:\conda\envs\spi\python.exe -m compileall -q gsdiff train.py
git diff --check
```

Expected: all tests pass and strict GSDiff output uses canonical IDs.

- [ ] **Step 12: Commit Task 4**

```powershell
git add -- gsdiff/experiments/gsdiff_adapter.py gsdiff/experiments/adapters.py gsdiff/experiments/__init__.py gsdiff/solver/admm.py train.py tests/experiments/test_gsdiff_adapter.py tests/solver/test_gradient_groups.py
git commit -m "feat: add strict GSDiff method adapter"
```

### Task 5: Structured Materializer and Procedural Audit Boundary

**Files:**

- Create: `gsdiff/experiments/execution.py`
- Create: `gsdiff/experiments/audit.py`
- Create: `scripts/experiments/method_child_bootstrap.py`
- Create: `tests/experiments/test_method_execution.py`
- Create: `tests/experiments/fixtures/truth_seeking_child.py`
- Modify: `gsdiff/experiments/__init__.py`

**Interfaces:**

- Consumes: `ResolvedMethod`, `AlgorithmSeed`, exact measurement/checkpoint
  hashes, and a source code root.
- Produces:

```python
@dataclass(frozen=True)
class MaterializedMethodExecution:
    argv: tuple[str, ...]
    cwd: Path
    env: Mapping[str, str]
    measurements_path: Path
    method_config_path: Path
    child_output_dir: Path
    expected_acquisition_spec: Mapping[str, object]
    audit_log_path: Path
    stdout_path: Path
    stderr_path: Path
    read_allowlist: tuple[Path, ...]
    read_root_allowlist: tuple[Path, ...]
    write_root_allowlist: tuple[Path, ...]
    requested_runtime_device: str
    child_runtime_device: str
    audit_policy_path: Path
    audit_policy_sha256: str
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


def materialize_method_execution(
    method: ResolvedMethod,
    *,
    stage_root: Path,
    measurements_source: Path,
    measurements_file_sha256: str,
    dataset_identity_sha256: str,
    expected_acquisition_spec: Mapping[str, object],
    algorithm_seed: AlgorithmSeed,
    checkpoint_store: Mapping[str, Path],
    python_executable: Path,
    source_root: Path,
    requested_runtime_device: str,
) -> MaterializedMethodExecution:
    raise ValueError


def load_materialized_method_request(
    path: Path,
) -> MaterializedMethodRequest:
    raise ValueError


def validate_audit_log(
    path: Path,
    *,
    expected_policy_sha256: str,
) -> Mapping[str, object]:
    raise ValueError
```

- [ ] **Step 1: Write RED stable-materialization tests**

Assert two different absolute stage roots produce:

- identical method/config hashes;
- identical path-free semantic configuration;
- different materialized argv/cwd paths;
- the same logical materialization record after its documented runtime-path
  fields are removed.

Parametrize requested device mapping and assert:

- `cpu` produces no `CUDA_VISIBLE_DEVICES`, records both device fields as
  `cpu`, and materializes `--device cpu`;
- `cuda:0` produces `CUDA_VISIBLE_DEVICES=0`, records child device `cuda:0`,
  and materializes `--device cuda:0`;
- `cuda:1` produces `CUDA_VISIBLE_DEVICES=1`, records requested device
  `cuda:1` but child device `cuda:0`, and materializes `--device cuda:0`;
- malformed, negative, bare `cuda`, and whitespace-containing device strings
  fail before staging.

Assert the exact private tree:

```text
input/measurements.npz
config/method-config.json
checkpoints/
code/
work/
child-output/
parent/audit/policy.json
parent/audit/file-opens.jsonl
parent/logs/stdout.log
parent/logs/stderr.log
```

The logs and audit log may be empty files before launch; they are parent-owned
and are not inside `child-output`.

- [ ] **Step 2: Run materializer tests and verify RED**

Run:

```powershell
D:\conda\envs\spi\python.exe -m pytest tests/experiments/test_method_execution.py -k "material" -q
```

Expected: collection fails because `execution.py` does not exist.

- [ ] **Step 3: Implement fail-closed staging**

Materialization must:

1. reject `ResolvedMethod.execution_ready=false` and report its exact blockers;
2. require a new or empty `stage_root` and reject symlinks/reparse points in
   its ancestry;
3. copy, never hardlink, measurement bytes;
4. hash the source before copy, destination after copy, and source once more
   after copy; all three hashes must equal the expected hash;
5. copy the strict import closure under `gsdiff/**/*.py`, excluding
   `gsdiff/evaluation/**`, `gsdiff/baselines/_evaluation.py`, and
   `gsdiff/data/_artifact_truth.py`, plus
   `train.py`, `scripts/run_baselines.py`,
   `scripts/experiments/method_child_bootstrap.py`, and
   `schemas/method-info-v2.schema.json` into `code/`; compatibility-only
   matplotlib/evaluator imports in the two CLIs must be lazy so the strict
   process does not require excluded files;
6. reject source symlinks/reparse points and copy no `data`, `results`,
   `checkpoints`, `.git`, `.claude`, `.superpowers`, or external workspace
   files;
7. copy each declared checkpoint after exact regular-file and SHA validation;
8. write canonical `materialized-method-config-v1` JSON with path-free
   resolved semantics, exact expected acquisition spec, expected measurement
   file hash, and a separate runtime checkpoint mapping;
9. replace only the approved stable tokens and reject residual `${...}`;
10. wrap the child command with the staged bootstrap;
11. accept only requested physical device `cpu` or full-match
    `cuda:[0-9]+`, record it outside semantic identity, and derive the
    child-visible device and `CUDA_VISIBLE_DEVICES` exactly;
12. use an argv tuple and `shell=False` semantics.

The approved tokens are `${PYTHON}`, `${MEASUREMENTS_PATH}`,
`${OUTPUT_DIR}`, `${METHOD_CONFIG_PATH}`, `${DATASET_IDENTITY_SHA256}`,
`${ALGORITHM_SEED}`, `${DEVICE}`, `${AUDIT_LOG_PATH}`, and declared
`${CHECKPOINT:logical-id}` values.

Launch the bootstrap with Python isolated/no-site flags:

```text
${PYTHON} -I -S -B -X utf8 scripts/experiments/method_child_bootstrap.py
  --policy D:\stage\parent\audit\policy.json
  --code-root D:\stage\code
  --entrypoint scripts/run_baselines.py
  -- --method dgi --dataset ${MEASUREMENTS_PATH}
```

`materialize_method_execution` validates that the registry template starts
with the selected Python executable followed by exactly `train.py` or
`scripts/run_baselines.py`. It removes those first two tokens, supplies the
entrypoint to the bootstrap, and passes the remaining materialized tokens
after `--`. The bootstrap sets
`sys.argv = [entrypoint, *child_arguments]` and executes the exact staged
entrypoint through `runpy.run_path(entrypoint, run_name="__main__")` in the
same audited process. It never launches a nested process. Both the original
materialized child argv and the wrapped bootstrap argv are recorded.

Device mapping is exact and tested:

| Requested physical device | Child environment | `${DEVICE}` / child device |
| --- | --- | --- |
| `cpu` | no `CUDA_VISIBLE_DEVICES` | `cpu` |
| `cuda:N` | `CUDA_VISIBLE_DEVICES=N` | `cuda:0` |

The materialized config records both `requested_runtime_device` and
`child_runtime_device`. For example, a `cuda:1` request must produce
`CUDA_VISIBLE_DEVICES=1` and `--device cuda:0`; passing `cuda:1` into that
child is a hard validation failure.

The bootstrap installs the hook before calling `site.main()` and before adding
the staged code root/importing method code, and also sets
`sys.dont_write_bytecode=True`. The Python runtime root is
`Path(python_executable).resolve().parent`; the Windows system read root
is the resolved `SystemRoot\System32`. Record both as allowed runtime read
roots. Do not include the research workspace root.

Construct a fresh environment rather than copying `os.environ`. It may contain
only `SYSTEMROOT`, `WINDIR`, a PATH built from the selected Python environment
and `SystemRoot\System32`, `TEMP`/`TMP` redirected below staged `work`,
`HOME`/`USERPROFILE`, `XDG_CACHE_HOME`, `TORCH_HOME`, and `MPLCONFIGDIR`
redirected below `work`, and
`CUDA_VISIBLE_DEVICES` when the requested physical device needs it. Strip
`PYTHONPATH`, real user-home values, dataset/artifact variables, and upstream
workspace paths. Because `-I` ignores Python environment variables, UTF-8 mode
comes from `-X utf8`; the bootstrap explicitly reconfigures stdout/stderr to
UTF-8 with replacement disabled.

- [ ] **Step 4: Write RED audit policy tests**

Install the hook in short-lived subprocesses and assert:

- staged input/config/code/checkpoint reads are allowed and logged;
- writes are allowed only below `child-output` and `work`;
- `../evaluation-truth.npz` is denied;
- a known absolute truth path is denied;
- `os.listdir` and `os.scandir` on a sibling/upstream root are denied;
- `os.chdir` outside staged `code`/`work` is denied;
- `subprocess.Popen` and `os.system` are denied;
- a symlink/reparse escape is denied when the platform permits creating it;
- denied events contain operation, resolved path or command class, decision,
  and UTC timestamp;
- the audit hook is installed before importing strict method code.
- importing fresh staged modules creates no `.pyc`, no `__pycache__`, and no
  denied bytecode-write event;
- a UTF-8 stdout/stderr sentinel containing `盲态验证` round-trips exactly;
- `from gsdiff.data import load_evaluation_truth` fails because the strict
  source snapshot contains no truth-loader implementation, and the staged
  inventory contains no evaluator/truth implementation file.
- the log begins with `hook-installed`, uses contiguous integer sequence
  numbers, and ends with one `bootstrap-finished` event;
- `MaterializedMethodExecution.audit_policy_path` is the exact canonical
  policy file and `audit_policy_sha256` matches its bytes; passing that explicit
  hash to `validate_audit_log` succeeds, while any other expected hash fails;
- truncated JSON, duplicate/out-of-order sequence numbers, policy-hash
  mismatch, or a missing terminal event fails validation.

- [ ] **Step 5: Run audit tests and verify RED**

Run:

```powershell
D:\conda\envs\spi\python.exe -m pytest tests/experiments/test_method_execution.py -k "audit or truth or subprocess or symlink" -q
```

Expected: tests fail because no audit hook exists.

- [ ] **Step 6: Implement the audit hook**

The trusted bootstrap may use only stdlib, read `policy.json`, open the audit
log, and load the stdlib-only `audit.py` by exact staged file path with
`importlib.util.spec_from_file_location`; it must not import the `gsdiff`
package before hook installation. It then calls `sys.addaudithook` before
`site.main()`, before adding the staged code root to `sys.path`, and before
importing or running method code. The hook writes JSON lines through the
pre-opened file descriptor to avoid recursive `open` events.

Handle at least:

```text
open
os.listdir
os.scandir
os.chdir
subprocess.Popen
os.system
```

Path policy rules:

- construct a lexical absolute path without resolving links;
- inspect every existing lexical ancestor with `lstat` and Windows file
  attributes, rejecting every symlink/junction/reparse component;
- for a nonexistent write target, inspect through its nearest existing parent
  before accepting the suffix;
- only after lexical reparse checks, resolve the path, apply `normcase`, and
  compare it with the allowlists;
- exact read paths: staged measurement, method config, audit policy, declared
  checkpoints;
- read roots: staged code and Python runtime;
- write roots: staged child output and work;
- audit log writes are limited to the bootstrap's already-open descriptor;
- the default decision for every governed filesystem/process operation is
  deny; unrelated CPython audit event classes are logged only when the policy
  declares them.

Raise `PermissionError` on denial after writing the event. Reject all nested
process creation. A denied event makes the parent run fail even if child code
catches `PermissionError` and exits zero. Document native extensions as
outside the adversarial sandbox claim.

`validate_audit_log` parses with duplicate-key rejection, verifies the policy
hash/header/terminal event/contiguous sequence, rejects any denied event for a
successful run, and returns an immutable summary with exact log SHA-256 and
event count. The parent never trusts a child-reported audit result.

- [ ] **Step 7: Add materialized request loader tests**

The strict child loader must reject:

- method-config duplicate JSON keys;
- extra keys;
- semantic hash mismatch;
- inconsistent canonical method/config/profile fields inside the request;
- malformed dataset identity, measurement hash, acquisition-spec, or
  algorithm-seed fields;
- an algorithm-seed integer inconsistent with the first four bytes of its
  declared derivation digest;
- a runtime checkpoint mapping whose logical IDs/hashes disagree with the
  resolved method or whose staged files fail exact hash validation;
- staged measurement/output/checkpoint paths outside the materialized stage
  roots;
- any absolute path inside the semantic section.

It returns one frozen `MaterializedMethodRequest` only after validation. The
request binds the resolved method, algorithm seed, dataset identity,
measurement-file hash, exact acquisition spec, staged measurement/output
paths, exact runtime checkpoint paths, requested physical device, and
child-visible logical device. No strict CLI may reconstruct any of these
values independently from command-line defaults.

- [ ] **Step 8: Run Task 5 gates**

Run:

```powershell
D:\conda\envs\spi\python.exe -m pytest tests/experiments/test_method_execution.py tests/experiments/test_methods.py tests/experiments/test_child_outputs.py -q
D:\conda\envs\spi\python.exe -m compileall -q gsdiff/experiments scripts/experiments
git diff --check
```

Expected: all tests pass.

- [ ] **Step 9: Commit Task 5**

```powershell
git add -- gsdiff/experiments/execution.py gsdiff/experiments/audit.py gsdiff/experiments/__init__.py scripts/experiments/method_child_bootstrap.py tests/experiments/test_method_execution.py tests/experiments/fixtures/truth_seeking_child.py
git commit -m "feat: materialize audited blind method children"
```

### Task 6: Strict CLIs, Real Subprocess Proof, and Provenance

**Files:**

- Create: `docs/experiments/method-registry-v1.md`
- Create: `tests/experiments/test_method_subprocess.py`
- Modify: `scripts/run_baselines.py`
- Modify: `train.py`
- Modify: `tests/data/test_artifacts.py:2248-2670`
- Modify: `tests/experiments/test_campaign_cli.py`
- Modify: `configs/protocols/pilot-v1.yaml` only if all readiness gates are
  actually satisfied

**Interfaces:**

- Consumes: materialized request loader, canonical dispatcher, v2 writer, and
  the bootstrap/audit boundary.
- Produces strict CLI contracts:

```text
run_baselines.py --method dgi --dataset D:\stage\input\measurements.npz
  --dataset-identity-sha256 aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
  --method-config D:\stage\config\method-config.json
  --algorithm-seed 123456789 --device cpu
  --output-dir D:\stage\child-output

train.py --method gsdiff_tv --dataset D:\stage\input\measurements.npz
  --dataset-identity-sha256 aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
  --method-config D:\stage\config\method-config.json
  --algorithm-seed 123456789 --device cpu
  --output-dir D:\stage\child-output

train.py --method gsdiff_diffusion --dataset D:\stage\input\measurements.npz
  --dataset-identity-sha256 aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
  --method-config D:\stage\config\method-config.json
  --algorithm-seed 123456789 --device cpu
  --output-dir D:\stage\child-output
  --checkpoint gsdiff-diffusion-prior-v1=D:\stage\checkpoints\gsdiff-diffusion-prior-v1
```

- [ ] **Step 1: Write RED strict CLI tests**

Tests must assert:

- exactly one canonical method per process;
- baseline CLI rejects GSDiff IDs and train CLI rejects baseline IDs;
- `--truth-path`, `--baselines`, `--name`, `--solver`, and arbitrary
  publication YAML overrides cannot enter strict mode;
- CLI method/dataset/seed/checkpoint values must match materialized config;
- CLI output and child-visible device values must match the loaded request;
- method ID, dataset path and identity, output path, algorithm seed, device, or
  checkpoint CLI tampering is rejected by the strict entry point after loading
  the internally valid request;
- strict success writes exact two v2 files;
- legacy invocation is clearly labelled nonpromotable and cannot write v2
  output or be accepted by corrected campaign validation.

- [ ] **Step 2: Run strict CLI tests and verify RED**

Run:

```powershell
D:\conda\envs\spi\python.exe -m pytest tests/experiments/test_campaign_cli.py -k "method_child or strict_method" -q
```

Expected: tests fail because current baseline blind mode supports only DGI and
both CLIs accept arbitrary legacy configuration.

- [ ] **Step 3: Implement strict entry points**

For strict mode:

1. parse only the declared arguments;
2. set `request = load_materialized_method_request(method_config_path)` and
   compare every CLI method/dataset/output/seed/checkpoint argument to the
   corresponding bound request field and compare `--device` specifically to
   `request.child_runtime_device`;
3. call
   `load_acquisition_data(request.measurements_path,
   expected_dataset_identity_sha256=request.dataset_identity_sha256,
   expected_acquisition_spec=request.expected_acquisition_spec)`;
4. call `run_canonical_method` with `request.method`,
   `request.algorithm_seed`, `request.child_runtime_device`, and
   `request.checkpoint_paths`;
5. call `write_method_child_outputs_v2` with
   `request.measurements_file_sha256`, the bound method/seed, the exact child
   start/finish timestamps, and `request.child_output_dir`;
6. print only concise progress and output paths;
7. never load truth, evaluator modules, publication metrics, or figures.

Move current behavior behind an explicit `legacy_main`/compatibility branch.
Compatibility output keeps v1 labels and is permanently nonpromotable. Emit a
migration message when old flags are used without an explicit compatibility
marker.

- [ ] **Step 4: Write RED real subprocess tests**

Create a tiny immutable measurement artifact and use the production
materializer/bootstrap to launch:

1. baseline `dgi` with `controller-cpu-smoke-v1`;
2. `gsdiff_tv` with one ADMM outer and one inner iteration on CPU.

For each run assert:

- return code 0;
- `child-output` has exact two files;
- v2 parent validation succeeds;
- argv, env, cwd, and materialized semantic config contain no truth path or
  upstream dataset root;
- `validate_audit_log` accepts a complete header/sequence/terminal record,
  reports no denied event, and finds no evaluator/data-root access;
- output is nonpromotable by profile semantics.

Run the fixture child through the same bootstrap and assert sibling read,
absolute upstream read, directory scan, symlink/reparse escape, and nested
subprocess attempts each return nonzero with a denied audit event.

- [ ] **Step 5: Run subprocess tests and verify RED**

Run:

```powershell
D:\conda\envs\spi\python.exe -m pytest tests/experiments/test_method_subprocess.py -q
```

Expected: real strict children fail until both CLIs use the materialized
request and v2 writer.

- [ ] **Step 6: Make real subprocess proof GREEN**

Fix only behavior demonstrated by the RED tests. Keep production audit policy,
not a test-only wrapper, in both success and attack cases. Capture stdout and
stderr into parent-owned paths. Do not move logs into `child-output`.

- [ ] **Step 7: Record exact ReCINR provenance**

In `docs/experiments/method-registry-v1.md`, record:

- URL `https://github.com/liuqjjin/ReCINR`;
- upstream commit `9149d1d228db2e4eb3ae852a004f1d9e95ee0229`;
- exact remote and local content SHA-256 for every adapted ReCINR file;
- the local modification summary;
- GitHub `licenseInfo=null` and absence of `LICENSE`, `COPYING`, and `NOTICE`
  at the audited revision;
- repository declarations without converting them into a confirmed
  redistribution license;
- archive status `blocked-license-copyright-review`.

Collect the evidence read-only with `gh api` and local `Get-FileHash`. Do not
modify the ReCINR repository.

- [ ] **Step 8: Document the registry and blockers**

The same document must include:

- all eleven canonical IDs and exact publication/smoke values;
- raw held-out formula and its physical units;
- separate algorithm seed domain;
- exact two-child/five-final ownership;
- procedural-boundary limitation;
- diffusion checkpoint logical ID/hash;
- missing checkpoint locator/training-provenance blocker;
- legacy migration path;
- why CPU smoke cannot be publication evidence.

- [ ] **Step 9: Evaluate pilot readiness honestly**

Keep `pilot-v1.execution_ready=false` when any required clean-clone
checkpoint, all-method subprocess, CUDA, disk, or repository-cleanliness gate
is unsatisfied. Record exact readiness blockers in documentation and tests.
Only change it to true when every condition in the approved design is proven
in the current commit; do not infer readiness from local checkpoint presence
alone.

- [ ] **Step 10: Run targeted and full verification**

Run:

```powershell
D:\conda\envs\spi\python.exe -m pytest tests/experiments/test_methods.py tests/experiments/test_objectives.py tests/experiments/test_child_outputs.py tests/experiments/test_baseline_adapters.py tests/experiments/test_gsdiff_adapter.py tests/experiments/test_method_execution.py tests/experiments/test_method_subprocess.py tests/experiments/test_campaign_cli.py tests/solver -q
D:\conda\envs\spi\python.exe -m pytest -m "not cuda" -q
D:\conda\envs\spi\python.exe -m compileall -q gsdiff scripts train.py
D:\conda\envs\spi\python.exe -c "import json; from pathlib import Path; from jsonschema import Draft202012Validator; s=json.loads(Path('schemas/method-info-v2.schema.json').read_text(encoding='utf-8')); Draft202012Validator.check_schema(s); print('method-info-v2 schema: valid')"
git diff --check
git status --short
```

Expected:

- every targeted test passes;
- full non-CUDA suite passes with only declared skips;
- compileall exits 0;
- schema check prints `method-info-v2 schema: valid`;
- diff check is clean;
- status contains only Task 4 intended files before commit.

- [ ] **Step 11: Commit Task 6**

```powershell
git add -- docs/experiments/method-registry-v1.md scripts/run_baselines.py train.py tests/data/test_artifacts.py tests/experiments/test_campaign_cli.py tests/experiments/test_method_subprocess.py configs/protocols/pilot-v1.yaml
git commit -m "feat: enforce strict blind method entry points"
```

If `pilot-v1.yaml` is correctly unchanged, omit it from `git add`.

- [ ] **Step 12: Formal Task 4 review and push**

Generate a review package from the design commit through Task 6. A formal
reviewer must independently check SPEC, CODE, and SECURITY. Any fix round goes
back to the responsible implementer; the same reviewer re-reviews the scoped
fix. Stop after five failed rounds. Task 4 is complete only with:

```text
P0 = 0
P1 = 0
P2 = 0
SPEC findings = 0
CODE findings = 0
SECURITY findings = 0
```

After clean review:

```powershell
git push origin debug/admm-vs-sgd
```
