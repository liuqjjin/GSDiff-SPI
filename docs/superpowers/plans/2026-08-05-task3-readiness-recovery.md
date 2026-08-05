# Task 3 Readiness Recovery Plan

## Purpose

This plan replaces the assumption that Task 3 is only a readiness-flag change.
The repository audit at `986e599` showed that the pilot, diffusion checkpoint,
and selection campaign are blocked for different reasons. They must be closed
in dependency order without weakening the publication boundary.

The shortest safe order is:

1. make the eleven-method CPU smoke pilot executable but permanently
   non-publication;
2. train and verify a target-disjoint diffusion prior v2;
3. run and verify the real eleven-method smoke pilot;
4. freeze the complete semantics of the 207-cell selection campaign;
5. execute later phases only from exact `PhasePlan` authority.

Paper production, repository cleanup, and publication-scale campaigns remain
out of scope until Task 3 is closed.

## Audit findings that change the implementation order

- `pilot-v1` is structurally complete, but the validator hard-codes it as not
  ready, the native-budget gate always raises, and the runner rejects its
  intentional `default -> smoke-default-v1` alias and nonpromotion policy.
- Smoke profiles do not carry the compute-cap object required by the runner.
- The local diffusion-prior v1 bytes match the registry, but their training
  commit, dataset, environment, and loss history cannot be proven.
- The legacy prior dataset includes the exact `tank` and `usaf` evaluation
  targets. V1 therefore remains historical and blocked even if its old logs
  are later recovered.
- `selection-decision-v1` has 207 logical cells, but the 17 one-factor
  configurations and six joint configurations do not yet resolve to complete
  executable scene/motion/solver semantics. The campaign runner also calls the
  campaign-only expander on an ablation document.
- Stress, confirmatory, and supplement-only phases require conditional
  authority that cannot be replaced by a readiness boolean.

## Fixed decisions for the immediate engineering work

- The existing content-addressed artifact store is retained. Pilot identities
  bind `phase_id=pilot-v1`, the smoke execution profile, and nonpublication
  policy. No parallel result-store architecture is introduced.
- A smoke completion remains publication-, selection-, and promotion-
  ineligible. No eligibility flag is relaxed to make the pilot run.
- `pilot-smoke-v1` is CPU-only. A CUDA request must fail before artifact or
  source-snapshot creation.
- Smoke profiles use the already frozen safety ceiling of 1,800 seconds and
  15,032,385,536 peak-VRAM bytes. This is a run guard, not evidence of
  convergence or expected resource use.
- Native budgets are derived from the resolved semantic configuration through
  one shared function. Campaign `method_budgets` must equal the derivation;
  copied integers are never an independent authority.
- Diffusion prior v1 is not edited or relabelled. V2 is a new logical artifact.
- V2 uses 5,000 videos of shape `20 x 64 x 64`, preserving the former training
  scale while removing target leakage. Its sources are the repository's four
  procedural descriptors `char:7`, `char:L`, `char:T`, and `shape:circle`.
  No external image asset is needed.
- The v2 final checkpoint is the predeclared epoch-200 EMA state. Training-time
  model selection is forbidden.
- Raw datasets, checkpoints, and per-step logs remain ignored local artifacts;
  only compact contracts and verified provenance are tracked.

## Work package A: close the pilot execution contract

### RED tests

- A ready campaign is rejected when any method budget differs from the native
  budget derived from its resolved profile.
- The pilot alias is accepted only as
  `pilot-smoke-v1/default -> controller-cpu-smoke-v1/smoke-default-v1`.
- The same alias or a nonpromotable method is rejected for every other phase.
- Pilot execution rejects CUDA before filesystem writes.
- Missing or changed smoke compute caps fail closed.
- Pilot completions retain all three false eligibility flags and cannot satisfy
  publication aggregate or results-lock inputs.

### Implementation

- Move native-unit/budget derivation to one public experiment-contract helper
  and use it from child-output validation and campaign validation.
- Replace the unconditional budget placeholder with an exact resolved-profile
  comparison.
- Narrowly authorize nonpromotion only for the identity-bound CPU pilot.
- Add compute caps to all smoke profiles and refresh the registry's canonical
  hashes and human-readable machine record.
- Keep `pilot-v1.execution_ready=false` until work package B succeeds.

### Focused gate

```powershell
D:\conda\envs\gsdiff-spi\python.exe -m pytest `
  tests\experiments\test_methods.py `
  tests\experiments\test_child_outputs.py `
  tests\experiments\test_protocol.py `
  tests\experiments\test_runner.py `
  tests\experiments\test_campaign_cli.py -q
```

Commit and push this package separately before generating training data.

## Work package B: train a defensible diffusion prior v2

### RED tests

- Reject unknown contract fields, noncanonical content, invalid device/seed,
  and any training descriptor that intersects the evaluation-target registry.
- Reject dataset shape, dtype, range, finite-value, size, or hash disagreement.
- Reject a dirty or different training source commit and a different strict
  environment lock.
- Reject silent CPU fallback, missing epochs, nonfinite losses, wrong optimizer
  step count, incomplete state dicts, and provenance written before the final
  checkpoint.
- Prove the full control path with a tiny fixture; ordinary tests must not
  depend on ignored v1 or v2 files.

### Implementation

- Add a canonical v2 training contract and strict schema.
- Refactor dataset generation to preallocate its tensor and atomically publish
  the dataset followed by its manifest.
- Refactor training to require explicit `cuda:0`, exact clean Git evidence,
  the dedicated environment, deterministic controls, and the verified dataset
  manifest.
- Persist complete epoch losses locally and publish the final compact
  provenance record last.
- Add an independent verifier that recomputes every content hash and validates
  the model signature and finite tensors.

### Training gates

Before training: clean committed control code, strict environment verification,
target-disjoint contract, verified 5,000-video dataset, a real-size one-batch
CUDA test, and at least 6 GiB free disk.

After training: 200 complete epochs, exactly 125,000 optimizer steps, finite
losses, verified epoch-200 EMA checkpoint, independent CUDA inference, and a
canonical provenance record.

## Work package C: promote v2 authority and run the pilot

- Update the diffusion logical ID, hash, command template, and provenance
  status only after the independent v2 verifier passes.
- Remove publication checkpoint blockers without changing unrelated method
  semantics.
- Populate pilot budgets from the shared native-budget derivation and set only
  the pilot readiness flag.
- Build the one canonical pilot dataset, run exactly eleven CPU smoke cells,
  and verify all eleven complete manifests.
- Report smoke timings as controller evidence only. Do not extrapolate them to
  publication GPU ETA; use completed native-budget GPU cells later for ETA.
- Re-run focused, full non-CUDA, and real-CUDA gates, obtain an independent
  review, commit, and push.

## Work package D: freeze selection semantics before 207 runs

This package starts with a one-page scientific decision table. It must bind all
23 candidate configurations to complete nested scene, motion, solver, prior,
initialization, native-budget, and compute-cap values without constructor
defaults. It must also add all twelve stress acquisition configurations.

Three selection semantics are not currently fixed by repository evidence and
must be approved before implementation because they can change the winner:

1. the exact convergence-history metric;
2. whether all nine runs must pass the convergence predicate;
3. the nine-run peak-VRAM reduction used by the tie break.

No selection readiness flag is changed before that approval and the complete
candidate table passes parameterized adapter tests.

## Work package E: one phase-authority path

- Runner, dataset planner, aggregation, verification, and results locking must
  consume the same exact `PhasePlan`.
- `selection-stress-v1` requires a canonical frozen selection produced from
  all 207 verified decision cells.
- `primary-confirmatory-v1` requires verifier-produced in-memory evidence for
  replay 207, stress 126, and primary-selection 297 from one publication
  commit.
- `supplement-grid-v1` is the 231-cell scientific-identity difference from the
  528-cell full grid, not the full campaign.
- OOD and failure phases expand their complete protocols directly.
- Conditional refusal of stress before selection and confirmatory before all
  630 prerequisites is correct completion behavior, not an unfinished gate.

## Global verification and stop rules

Every package follows RED -> GREEN -> focused -> full -> real CUDA -> clean
diff -> independent review -> commit -> non-force push.

Stop before numerical execution if any configuration depends on an implicit
constructor default, if a checkpoint or dataset cross-hash cannot be rebuilt,
if the source tree is dirty, or if a protocol change alters the already
approved grid/counts. Never access `D:\SPI\ReCINR`, and never modify
`D:\conda\envs\spi`.
