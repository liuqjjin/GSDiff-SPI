# GSDiff experiment protocol v1

This document describes the immutable, versioned declarations in
`configs/protocols/`. Task 1 freezes structure and scientific intent; it does
not generate a dataset or launch a method.

## Identity and hashing

Every expanded run carries the scientific-contract ID and its canonical
content SHA-256. The campaign ID is scheduling and aggregation metadata and is
not part of reusable run identity. In Task 1, acquisition-config and
method-config IDs are scheduling and logical campaign-membership dimensions.
Tasks 2–4 make the resolved acquisition and method content cryptographically
identity-bearing through dataset and resolved-config hashes; `campaign_id`
never enters `RunIdentity`.

`campaign_sha256` hashes the sorted set of logical membership records:
target, motion, seed, method, acquisition-config ID, and method-config ID.
Therefore source-list reordering does not change campaign membership.
`protocol_sha256` hashes the normalized document with its own hash omitted.
It remains list-order-sensitive so declared report order is auditable.

The loader accepts only restricted YAML in JSON's scalar domain. It rejects
anchors, aliases, merge keys, duplicate keys, timestamps, binary scalars,
non-finite numbers, non-string mapping keys, unknown schema keys, and
inconsistent hashes before a protocol can be consumed.

## Corrected scientific contract

All scheduling motion aliases resolve to corrected generator
`gsdiff-corrected-sim` version `generator-v1`, with generator motion type
`custom_se2` and time normalized to `[0, 1]`. The generator must never receive
`trans`, `rot`, `transrot`, or `accel` as legacy motion types. Their physical
parameters are:

| Alias | Velocity | Acceleration | Omega | Beta |
|---|---:|---:|---:|---:|
| `trans` | `[8, 8]` | `[0, 0]` | `0` | `0` |
| `rot` | `[0, 0]` | `[0, 0]` | `0.4` | `0` |
| `transrot` | `[8, 8]` | `[0, 0]` | `0.3` | `0` |
| `accel` | `[6, 6]` | `[3, 3]` | `0.2` | `0.1` |

The corrected acquisition uses Bernoulli `{0,1}` patterns, stratified pattern
order, uniform time assignment, and uniform-random holdout patterns. RNG uses
NumPy `PCG64` with
`SeedSequence(entropy=seed, spawn_key=(stream_id,))`: train pattern `0`, train
noise `1`, holdout pattern `2`, and holdout noise `3`.

Detector noise is absolute, not recalculated independently for each signal:

```text
sigma = sqrt(var(y_reference, ddof=0)) * 10**(-snr_db/20)
```

The Bernoulli reference cell's sigma is reused for its train, holdout, and
alternate-pattern measurements.

Targets resolve explicitly to `assets/tank.png`, `char:5`, `char:R`,
`assets/usaf_tar_small.png`, and the repository's `assets/cx_camera.png`,
`assets/cx_clutter.png`, `assets/cx_coins.png`, and `assets/cx_text.png`.

## Campaign matrices

The standard corrected acquisition is `64×64`, 20 frames, 2,560 training
measurements, 250 held-out measurements, 25 dB, and `metrics-v1`.

| Campaign | Matrix | Runs | Datasets |
|---|---|---:|---:|
| Main (`primary-v1`) | 3 targets × 3 rigid motions × 5 seeds × 11 methods | 495 | 45 |
| Supplement (`supplement-grid-v1`) | 4 targets × 4 motions × 3 seeds × 11 methods | 528 | 48 |
| OOD (`ood-v1`) | 2 targets × 3 rigid motions × 3 seeds × 11 methods | 198 | 18 |
| Failure (`failure-v1`) | 2 targets × 1 motion × 3 seeds × 6 budgets × 5 methods | 180 | 36 |
| Pilot (`pilot-v1`) | 1 target × 1 motion × 1 seed × 11 methods | 11 | 1 |

Main targets are `tank`, `digit5`, and `usaf`; motions are `trans`, `rot`, and
`transrot`; seeds are `7`, `11`, `42`, `73`, and `101`. The supplement adds
`letterR` and `accel` but uses only seeds `7`, `11`, and `42`. Its exact
overlap with main is 297 runs and 27 datasets. Supplement-only membership is
231 runs; the union is 726 runs and 66 datasets.

The versioned YAML files are authoritative. Their remaining matrix memberships
are:

- OOD: targets `[cx_camera, cx_clutter]`; motions
  `[trans, rot, transrot]`; seeds `[7, 11, 42]`; acquisition config `[base]`;
  and all 11 canonical methods (`dgi`, `static_cs`, `perframe_cs`, `tv3d`,
  `monin`, `gidc3dtv`, `recinr`, `siren`, `recinr_se2`, `gsdiff_tv`,
  `gsdiff_diffusion`). Every method uses method config `default`.
- Failure: targets `[cx_coins, cx_text]`; motion `[transrot]`; seeds
  `[7, 11, 42]`; methods `[dgi, tv3d, recinr_se2, gsdiff_tv,
  gsdiff_diffusion]`; and acquisition configs
  `[m320, m640, m1280, m2560, m3840, m5120]`, binding 320, 640, 1,280,
  2,560, 3,840, and 5,120 training measurements respectively. Every method
  uses method config `default`.

The failure matrix is an identifiability/measurement-budget study, not a
representation-win claim.

The structural pilot uses `32×32`, 4 frames, 128 training measurements, and 16
held-out measurements. It is intentionally `execution_ready: false` under
`pilot-smoke-v1`; runnable per-method budgets are not guessed in Task 1.

## Methods, selection, and confirmation

Canonical method order is `dgi`, `static_cs`, `perframe_cs`, `tv3d`, `monin`,
`gidc3dtv`, `recinr`, `siren`, `recinr_se2`, `gsdiff_tv`, and
`gsdiff_diffusion`. `gsdiff_tv` is the primary TV-prior identity.
Diffusion-prior configurations use the separate `gsdiff_diffusion` lane.
`gsdiff_diff` is recorded only as a compatibility alias and is never canonical.
The confirmation comparison remains exactly `gsdiff_tv` versus `recinr_se2`.

The ablation contract uses selection anchors `tank/trans`, `digit5/rot`, and
`usaf/transrot`, with seeds `[7, 11, 42]`. Confirmatory seeds `[73, 101]` are
forbidden during selection. Its ten axes are locked exactly:

- representation: `[gaussian, siren, grid, recinr_se2]`
- solver: `[sgd, hqs, admm]`
- prior: `[tv2d, tv3d_corrected, diffusion]`
- motion warmup fraction: `[0, 0.1, 0.2, 0.4]`
- temporal-TV weight: `[0, 0.05, 0.1, 0.3]`
- training measurements: `[640, 1280, 1920, 2560, 3840, 5120]`
- SNR in dB: `[15, 20, 25, 30]`
- pattern: `[bernoulli, gaussian, hadamard_natural, fourier]`
- motion fit: `[matched, translation_only, rotation_only]`
- Gaussian count: `[250, 500, 1000, 1500]`

The semantic anchor is `gaussian` representation, `admm` solver,
`tv3d_corrected` prior, motion-warmup fraction `0.2`, temporal-TV weight `0.1`,
and Gaussian count `1000`.

| ID | Representation | Solver | Prior | Warmup | Temporal TV | Gaussians | Method lane | Method config |
|---|---|---|---|---:|---:|---:|---|---|
| `j1` | `recinr_se2` | `hqs` | `diffusion` | 0.2 | 0.1 | null | `gsdiff_diffusion` | `ablation-j1-v1` |
| `j2` | `grid` | `admm` | `diffusion` | 0.2 | 0.1 | null | `gsdiff_diffusion` | `ablation-j2-v1` |
| `j3` | `siren` | `sgd` | `tv3d_corrected` | 0.1 | 0.05 | null | `gsdiff_tv` | `ablation-j3-v1` |
| `j4` | `gaussian` | `hqs` | `tv2d` | 0.1 | 0.05 | 1500 | `gsdiff_tv` | `ablation-j4-v1` |
| `j5` | `recinr_se2` | `sgd` | `tv2d` | 0.4 | 0.3 | null | `gsdiff_tv` | `ablation-j5-v1` |
| `j6` | `grid` | `hqs` | `tv3d_corrected` | 0.4 | 0.05 | null | `gsdiff_tv` | `ablation-j6-v1` |

The locked counts are 17 one-factor configurations and six shortlist
configurations; 207 decision cells; 207 post-freeze replay cells; 14 stress
configurations and 126 stress cells; 333 publication-commit cells; and 540
total executions before retries.

Selection minimizes held-out normalized L2 residual over all nine
anchor-by-seed cells, subject to the locked convergence and compute gates.
Confirmation uses the untouched seeds `73` and `101`, complete finite paired
results, and no inferential-significance claim.

## Legacy evidence boundary

Corrected protocol results do not inherit, reuse, or satisfy any legacy
random-pattern output. Existing `results/eval_matrix` artifacts are
legacy/incomplete evidence. Every corrected cell must be generated under the
versioned Bernoulli acquisition, RNG, absolute-noise, contract, and identity
rules above.
