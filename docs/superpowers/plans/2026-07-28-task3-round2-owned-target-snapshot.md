# Task 3 Round-2 Owned Target Snapshot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ensure request resolution and dataset generation never reread caller-owned `TargetSnapshot` state after validation.

**Architecture:** `_validate_generation_inputs` will capture each target field once into a private frozen owned snapshot. Assets and renderer become deep-frozen native copies, while the canonical image becomes a fresh immutable array copy that is validated after copying; all downstream resolver and generator work consumes only this owned value.

**Tech Stack:** Python 3.11, frozen dataclasses, NumPy, pytest.

## Global Constraints

- Edit only Task 3 owned-snapshot production, tests, this plan, and the ignored Task 3 report.
- Do not edit the round-1 fix plan.
- Do not commit, push, run Task 4/GPU work, or touch `D:\SPI\ReCINR`.
- Use `D:\conda\envs\spi\python.exe` and `apply_patch` for every file edit.

---

### Task 1: Deterministic post-validation RED

**Files:**
- Test: `tests/data/test_artifacts.py`

**Interfaces:**
- Consumes: real `gsdiff.data._corrected_generation._validate_generation_inputs`.
- Produces: two regression tests that mutate the caller snapshot immediately after validation returns.

- [ ] Add a resolver test that mutates target ID, assets and renderer after the real validator returns, then requires rejection or the exact pre-mutation owned request/config values.
- [ ] Add a generator test that mutates target ID, assets and canonical image after the real validator returns, then requires request, config, truth and generated arrays to derive exclusively from the pre-mutation owned snapshot.
- [ ] Run both named tests and record the exact expected RED.

### Task 2: Frozen owned target snapshot

**Files:**
- Modify: `gsdiff/data/_corrected_generation.py`

**Interfaces:**
- Produces: a private frozen owned target value returned by `_validate_generation_inputs`.
- Consumed by: `resolve_corrected_dataset_request` and `generate_corrected_dataset`.

- [ ] Capture caller fields once, make deep-owned assets/renderer copies and a fresh canonical-image copy, then validate and freeze only those owned values.
- [ ] Return the owned target from `_validate_generation_inputs`.
- [ ] Replace every post-validation `target_snapshot.*` read in resolver and generator code with the owned target.
- [ ] Run the two named tests and the focused target/request regression set to GREEN.

### Task 3: Gates and review

**Files:**
- Append: `.superpowers/sdd/2026-07-27-gsdiff-experiments-artifacts/task-3-report.md`

- [ ] Run complete data, campaign CLI, Task 3 joint, and full non-CUDA pytest gates.
- [ ] Run compileall, all three Schema checks/roundtrips, and `git diff --check`.
- [ ] Obtain an independent read-only review and resolve every concrete P0/P1/P2.
- [ ] Append exact RED/GREEN/gate evidence as Phase 3C3g without editing this plan.
