# GSDiff-SPI Correctness, Reproducibility, and Publication Design

**Date:** 2026-07-27

**Status:** Approved direction; implementation requires a separate task plan

**Repository baseline:** `c03420784bc92b4e9b9eef8330cbd9571ebebc68` on `debug/admm-vs-sgd`

## 1. Objective

Complete the unfinished Claude Code handoff, correct the implementation before
generating final results, and turn GSDiff-SPI into a reproducible,
submission-ready research project.

The final state must include:

1. a tested implementation whose numerical operators satisfy their stated
   mathematics;
2. a versioned experiment runner that cannot silently reuse incompatible
   results or overwrite a partial table;
3. a complete multi-seed comparison in which GSDiff-TV is the primary method
   and the diffusion prior is presented as a secondary, distribution-dependent
   enhancement;
4. a reversible repository-cleaning record;
5. a complete English-language computational-imaging manuscript, figures,
   tables, references, and supplementary material;
6. independent code and manuscript reviews with no unresolved critical or
   major findings.

“Optimal” means best within a declared, finite search space under a
pre-registered GT-free selection rule. It does not mean a universal proof of
the global optimum of a non-convex reconstruction problem.

## 2. Scope and Non-goals

### In scope

- GSDiff scene, motion, forward-model, solver, prior, evaluation, experiment,
  plotting, and paper-generation code.
- All tracked source and documentation in this repository.
- Ignored experiment outputs needed to establish provenance or determine
  whether an artifact can be archived or deleted.
- The vendored ReCINR baseline and its relationship to upstream commit
  `9149d1d228db2e4eb3ae852a004f1d9e95ee0229` in `D:\SPI\ReCINR`.
- Simulation experiments required for the main paper and supplement.

### Out of scope unless later evidence makes it necessary

- Retraining a broad-domain diffusion model as a headline contribution.
- Claiming that the diffusion prior improves every target.
- Rewriting the independent ReCINR project.
- Deleting raw experiments before a regeneration and comparison gate passes.
- Using ground truth for hyperparameter or checkpoint selection.

## 3. Current-State Findings That Govern the Design

The unfinished Claude task is a final target-by-motion, multi-seed comparison.
It was never queued. The current runner cannot produce the promised table
because it reads a diffusion-only default for `gsdiff` and replaces
`results/eval_matrix/table.json` on every invocation.

The following correctness and evidence issues must be resolved before any
result is promoted to the final paper:

- `TVPrior3D` applies `temporal_weight` to the temporal forward difference but
  omits the same factor in the corresponding divergence. The current operator
  fails the weighted adjoint identity for every tested value except
  `temporal_weight=1`.
- The diffusion schedule does not evaluate the exact requested final noise
  level on its last call.
- Frozen motion parameters can retain gradients while excluded from the
  optimizer and can affect global gradient clipping.
- Per-frame independent min-max normalization removes an affine degree of
  freedom from every frame and is too permissive as the sole image metric.
- Result reuse is based on path existence rather than a complete experiment
  identity.
- Current tables are partial and can be overwritten by subset runs.
- Existing figures are diagnostic 150-dpi plots rather than manuscript
  figures.
- No manuscript source currently exists.

Legacy results remain useful as historical evidence but are not valid as final
post-fix measurements.

## 4. Workstream A — Correctness and Test Architecture

### 4.1 Test framework

Create a `pytest` suite under `tests/`. Tests must be deterministic on CPU
unless the behavior specifically requires CUDA. A separate marker will cover
CUDA smoke tests.

Required numerical tests:

- weighted TV gradient/divergence adjoint identity in float64 for
  `alpha ∈ {0, 0.05, 0.3, 1, 2}`, with relative error below `1e-10`;
- `alpha=0` 3D-TV behavior agreeing with independent per-frame 2D-TV within a
  documented numerical tolerance;
- finite-difference checks for scene and motion gradients on a tiny synthetic
  problem;
- forward-model consistency between batched and direct measurement
  evaluation;
- exact diffusion schedule endpoints for one-step and multi-step schedules;
- frozen motion parameters having no gradients and no influence on scene
  gradient clipping;
- deterministic data, noise, pattern, and holdout generation for a fixed seed;
- rectangular-image coordinate tests for every supported scene
  representation;
- cache-identity and aggregation tests that reproduce the old overwrite and
  stale-reuse failures before the implementation is changed.

Every behavior change follows red–green–refactor: the reproducing test must be
observed failing for the expected reason before production code is edited.

### 4.2 Correctness changes

Fix one root cause at a time:

1. weighted temporal divergence in `TVPrior3D`;
2. the stated isotropic/anisotropic TV definition mismatch;
3. diffusion schedule endpoint accounting and logging;
4. frozen-parameter gradient handling and group-specific clipping;
5. rectangular coordinate and padding inconsistencies;
6. evaluation metrics and explicit intensity-alignment policy;
7. result identity, incremental aggregation, and failure propagation.

Existing legacy metric names remain readable, but new results must expose
unambiguous names and definitions.

## 5. Workstream B — Reproducible Experiment System

### 5.1 Immutable experiment identity

Every run receives a manifest containing:

- Git commit and dirty-worktree flag;
- complete resolved configuration;
- SHA-256 of the resolved configuration;
- method, target, motion, seed, and protocol version;
- dataset, asset, and checkpoint SHA-256 values;
- Python, PyTorch, CUDA, GPU, OS, and dependency versions;
- exact command, start/end timestamps, return code, runtime, and peak VRAM;
- metric definitions and implementation version.

A cached result is reusable only when the complete identity matches. A path
match alone is never sufficient.

Runs write to a temporary directory and are atomically marked complete only
after all required files validate. Aggregation reads complete manifests rather
than directory names. Subset runs merge into the aggregate and cannot delete
unrelated cells.

### 5.2 Method identities

The experiment runner exposes separate method names:

- `dgi`
- `static_cs`
- `perframe_cs`
- `tv3d`
- `monin`
- `gidc3dtv`
- `recinr`
- `siren`
- `recinr_se2`
- `gsdiff_tv`
- `gsdiff_diffusion`

`gsdiff_tv` is the paper’s primary method. `gsdiff_diffusion` is a secondary
in-distribution enhancement and may not be relabelled as the default method.

All methods receive the same generated measurements for a given
target/motion/seed cell. GT-free tuning uses a held-out measurement split and
never PSNR, SSIM, trajectory error, or any ground-truth image statistic.

### 5.3 Locked primary protocol

The primary simulated protocol is:

- image size: `64 × 64`;
- frames: `20`;
- measurements: `2560` training plus `250` held-out;
- primary pattern: Bernoulli `0/1`;
- measurement SNR: `25 dB`;
- seeds: `7, 11, 42`;
- confirmatory headline seeds: `73, 101` in addition to the three primary
  seeds;
- primary rigid motions: translation, rotation, and translation+rotation;
- accelerated motion: supplementary full-grid experiment;
- GSDiff-TV: main row;
- GSDiff-diffusion: secondary row, interpreted jointly with in-/out-of-
  distribution status.

The Claude grid is retained in the supplement:

- targets: `tank`, `digit5`, `letterR`, `usaf`;
- motions: `trans`, `rot`, `transrot`, `accel`;
- seeds: `7, 11, 42`.

The main text uses the three rigid motions and targets that provide distinct
scene regimes. OOD analysis uses `cx_camera` and `cx_clutter` with three
primary seeds. `cx_coins` and `cx_text` are retained as measurement-budget
failure cases rather than evidence of representation superiority.

Existing random-pattern measurements are legacy evidence. Pattern choice is
tested explicitly rather than mixed silently into the primary table.

### 5.4 Metrics and statistics

Primary image metrics use one global affine calibration for the full video,
not independent per-frame min-max normalization:

- PSNR;
- SSIM;
- normalized RMSE.

Secondary metrics:

- legacy per-frame normalized PSNR, labelled as such;
- held-out measurement residual;
- training residual;
- translation, rotation, acceleration, and angular-acceleration error;
- runtime, peak VRAM, parameter count, and convergence/failure rate.

Report paired per-seed results, mean ± standard deviation, and a 95% bootstrap
confidence interval for headline effects. Do not use a significance claim
when the available number of independent seeds is insufficient.

### 5.5 Ablations

Required controlled studies:

- scene representation: Gaussian, SIREN, ReCINR, ReCINR-SE2;
- solver: SGD, HQS, ADMM;
- prior: 2D TV, corrected 3D TV, diffusion;
- motion warmup and temporal-TV weight;
- measurement budget around the observed phase transition;
- SNR;
- Bernoulli, random Gaussian, and ordered pattern families;
- motion complexity and model misspecification;
- Gaussian count and computational cost.

Searches use a declared finite grid. Selection is based on held-out
measurements, followed by a separate untouched confirmatory seed set.

## 6. Workstream C — Repository and Artifact Governance

### 6.1 Tracked repository

Track:

- source, tests, locked paper configs, manifest schemas, aggregation code;
- compact final tables and figure-source data;
- paper sources, references, captions, and reproducibility documentation;
- checkpoint/data cards and SHA-256 manifests.

Do not track:

- full checkpoints, raw generated datasets, per-frame run dumps, Python
  caches, or temporary render artifacts.

Large reproducibility assets receive a documented generation command or an
external location plus checksum. The vendored ReCINR baseline records its
upstream commit, local modifications, authorship, and license status.

### 6.2 Two-gate cleanup

No ambiguous result is deleted directly.

1. Produce an inventory with size, timestamps, completeness, configuration
   identity, duplicates, and downstream references.
2. Move proven scratch, cache, duplicate, corrupt, and incomplete candidates
   into a dated recoverable `_trash` area.
3. Regenerate all locked tables and figures from the retained authoritative
   artifacts and compare their hashes/data.
4. Delete the quarantined files only when regeneration is complete and no
   manifest or paper asset refers to them.

Immediately reproducible caches such as `__pycache__` are still recorded in
the cleanup report.

## 7. Workstream D — Manuscript and Figure System

### 7.1 Default publication target

Until a specific venue is supplied, the manuscript is an English-language
computational-imaging journal article built in venue-neutral LaTeX. Figure
assets support single-column `85 mm` and double-column `178 mm` widths so the
source can be moved to an Optica, IEEE, or Nature-family template without
redrawing the science.

### 7.2 Manuscript structure

Create:

- title and abstract;
- introduction and evidence-based related work;
- dynamic SPI forward model;
- Gaussian scene and SE(2) motion representation;
- ADMM/HQS optimization and corrected TV/diffusion priors;
- experimental protocol and GT-free model selection;
- main comparison;
- ablations and computational analysis;
- OOD behavior and limitations;
- discussion and conclusion;
- data/code availability, reproducibility, author contribution, funding,
  conflict, and ethics statements;
- supplementary methods, full grids, hyperparameters, and failure cases.

Every numerical claim maps to a locked result cell. Every novelty claim maps
to a verified reference or is weakened/removed.

### 7.3 Figure set

Produce manuscript figures from locked machine-readable data:

1. acquisition, scene, motion, and reconstruction overview;
2. representative reconstructions with identical display limits, error maps,
   motion trajectories, and zoomed regions;
3. multi-seed main comparison and uncertainty;
4. measurement-budget, noise, pattern, and solver/prior ablations;
5. convergence, runtime, memory, and motion recovery;
6. in-distribution versus OOD crossover and failure analysis.

Line art is vector PDF/SVG. Raster panels are exported at the final physical
size with at least 600 dpi for line-rich content and 300 dpi for continuous
tone. Fonts, line weights, grayscale behavior, panel labels, and a
color-vision-safe palette are shared through one plotting style module.

### 7.4 Low-AI writing standard

The prose must:

- state the physical problem before the method name;
- use measured quantities and explicit conditions instead of promotional
  adjectives;
- avoid canned signposting, repetitive topic sentences, fake quotations,
  sweeping novelty claims, and generic “significant improvement” language;
- distinguish observation, mechanism hypothesis, and demonstrated cause;
- include negative results and limitations where they change interpretation;
- retain natural variation in sentence length without sacrificing precision;
- be reviewed against the source code, equations, figures, and result
  manifests rather than polished in isolation.

## 8. Delivery Sequence

1. Preserve the `c034207` state and label its existing results as legacy.
2. Establish tests and reproduce each confirmed failure.
3. Fix correctness defects with one red–green cycle per behavior.
4. Implement manifest-backed running and aggregation.
5. Run a tiny corrected pilot and compare legacy versus corrected behavior.
6. Freeze the protocol and run the primary and supplementary campaigns.
7. Generate locked tables and figures.
8. Quarantine and verify cleanup candidates.
9. Write and typeset the manuscript from the locked evidence.
10. Run independent code review, reproducibility review, figure QA, citation
    verification, and three simulated peer reviews.

## 9. Acceptance Gates

The project is not complete until all gates pass.

### Correctness gate

- all CPU and CUDA tests pass from a clean environment;
- numerical adjoint and gradient tolerances pass;
- no unresolved critical or major code-review finding remains;
- documented equations match the implementation.

### Experiment gate

- every required cell has the declared seeds and a matching manifest;
- no aggregate contains a stale, incomplete, dirty-tree, or identity-mismatched
  run;
- headline conclusions survive the confirmatory seeds;
- failures and OOD behavior are reported rather than excluded post hoc.

### Repository gate

- a clean clone can install, run tests, execute the smoke protocol, and
  regenerate tracked tables and figures;
- every retained large artifact has provenance and a checksum;
- quarantined junk has passed the regeneration comparison before deletion;
- `git status` is clean at handoff.

### Publication gate

- figures pass physical-size rendering and visual inspection;
- tables match locked machine-readable data;
- manuscript equations, captions, cross-references, and bibliography compile
  without warnings that affect correctness;
- every factual and numerical claim has an evidence pointer;
- independent reviewer simulation has no unresolved major concern;
- the final prose passes a manual low-AI-style review.

## 10. Safety and Change Control

- Implementation occurs in an isolated worktree after user consent.
- Each task is independently testable and committed separately.
- GPU campaigns begin only after correctness and experiment-runner gates pass.
- Destructive cleanup requires exact targets, a dated inventory, and the
  two-gate process above.
- No push, merge, release, or deletion is inferred from test success; each
  follows the user-authorized project workflow.
