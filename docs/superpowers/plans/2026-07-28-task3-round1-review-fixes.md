# Task 3 Round-1 Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:executing-plans and strict test-driven development. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the five formal Task 3 round-1 findings without entering
Task 4, running a GPU campaign, touching `D:\SPI\ReCINR`, or committing.

**Architecture:** Keep all scientific semantic validation in
`_corrected_generation.py`, reuse the existing descriptor-bound safe file
snapshot in `_artifact_io.py`, and make the build CLI classify verified
on-disk manifests before deciding whether generation is necessary. Directory
discovery remains a recheckable snapshot, never a publication lock.

**Tech Stack:** Python 3.11, NumPy, Pillow, JSON Schema Draft 2020-12,
pytest, immutable dataset directories.

## Global Constraints

- Use `D:\conda\envs\spi\python.exe`.
- Every production behavior change requires a test observed failing first.
- Preserve all existing threat-boundary and concurrency behavior.
- Do not modify Task 4, run GPU work, touch `D:\SPI\ReCINR`, commit, or push.
- Append final evidence to
  `.superpowers/sdd/2026-07-27-gsdiff-experiments-artifacts/task-3-report.md`.

---

### Task 1: Exact Shared Renderer and Authoritative Semantic Validation

**Files:**
- Modify: `tests/data/test_artifacts.py`
- Modify: `gsdiff/data/_corrected_generation.py`
- Modify: `gsdiff/data/_artifact_bundle.py`
- Modify: `schemas/dataset-manifest-v1.schema.json`

**Interfaces:**
- Produces: `_validate_target_renderer(renderer, *, field)` returning the
  exact accepted renderer mapping.
- Produces: authoritative `validate_dataset_identity_spec()` checks for the
  scientific contract and nonempty target assets.
- Consumes: existing exact-native, opaque-ID, SHA-256 and canonical JSON
  validators.

- [ ] Add parameterized direct `TargetSnapshot` and request-resolution RED
  tests for missing, extra, null, empty, subclassed, out-of-range and wrong
  constant renderer values.
- [ ] Add coherent manifest rehash/parser/directory RED tests proving runtime
  and Draft 2020-12 schema return the same verdict.
- [ ] Run the named tests and record the expected validation holes.
- [ ] Add one exact renderer validator: glyph requires exactly
  `font_family`, `fill_fraction`, `resample`, `supersample`, with a path-free
  nonempty family, `0 < fill_fraction <= 1`, exact integer supersample at
  least one, and exact `"lanczos"`; file requires exactly `color_mode` and
  `resample`, with exact `"grayscale"` and `"lanczos"`.
- [ ] Invoke it from `TargetSnapshot`, request resolution, corrected config,
  truth validation and manifest validation paths.
- [ ] Add RED tests for path-bearing/empty scientific-contract IDs, malformed
  contract SHA-256, empty identity target assets, and non-float64 calibration
  reference descriptors through direct identity, coherent manifest,
  standalone truth serializer and truth loader paths.
- [ ] Make `validate_dataset_identity_spec()` authoritative for exact
  scientific contract and nonempty target assets; make corrected truth use
  `np.dtype(np.float64).str` exactly and share the same calibration-record
  validator used by manifests.
- [ ] Run the focused Task 1 tests to GREEN.

### Task 2: Exact-Native Blind Loader Anchors

**Files:**
- Modify: `tests/data/test_artifacts.py`
- Modify: `gsdiff/data/_artifact_dataset.py`

**Interfaces:**
- Consumes: `validate_exact_json_native`,
  `_validate_blind_acquisition`, and the stored blind dimensions validator.
- Preserves: public `load_acquisition_data()` and
  `load_acquisition_data_bytes()` signatures.

- [ ] Add RED tests passing an exact-string subclass as the expected identity,
  a dict subclass as the expected acquisition spec, and nested `np.int64`
  values.
- [ ] Add RED tests for malformed expected dimensions/acquisition values that
  currently survive canonical coercion.
- [ ] Run the named tests and verify failures are due to acceptance, not setup.
- [ ] Require an exact native SHA string and exact dict or
  `MappingProxyType`; recursively call `validate_exact_json_native`.
- [ ] Validate expected dimensions with the same exact integer/minimum rules
  and expected acquisition with `_validate_blind_acquisition` before canonical
  comparison.
- [ ] Run the focused loader tests to GREEN.

### Task 3: Bounded Safe Target and Font Snapshots

**Files:**
- Modify: `tests/data/test_artifacts.py`
- Modify: `gsdiff/data/_corrected_generation.py`

**Interfaces:**
- Consumes: `read_safe_file_snapshot()` and
  `verify_safe_file_snapshot()` from `_artifact_io.py`.
- Produces: explicit target-image and bundled-font byte limits.

- [ ] Add RED tests proving an oversized target/font is rejected before the
  bulk read or decoder, restored-mtime same-size replacement is rejected via
  ctime, and target/font hardlinks and leaf replacement races fail closed.
- [ ] Run the named tests and record the custom-reader gaps.
- [ ] Replace `_read_regular_snapshot()` with the shared descriptor-bound safe
  snapshot, using explicit role-specific byte limits.
- [ ] Recheck the snapshot after decode/render and before returning
  `TargetSnapshot`, so leaf substitution after the read fails closed.
- [ ] Run the focused snapshot tests and existing target-security tests to
  GREEN.

### Task 4: Pre-Generation Build Reuse and Final Unique Catalog Proof

**Files:**
- Modify: `tests/experiments/test_campaign_cli.py`
- Modify: `scripts/experiments/build_datasets.py`

**Interfaces:**
- Consumes: recheckable dataset discovery, full sequential
  `verify_dataset_directory()`, and canonical request/manifest projections.
- Produces: per-request classification as unique current, missing or
  ambiguous without retaining verified arrays.

- [ ] Add a second-pilot RED test with generator and publisher forbidden and
  exact before/after byte, mtime and ctime inventory.
- [ ] Add RED tests for two coherent semantic twins, a valid current dataset
  plus corrupt unrelated dataset, corrupt exact final data, publisher winner
  after the scan, and candidate disappearance/replacement.
- [ ] Assert missing classification generates and publishes, ambiguous
  classification fails without generation/write, and unrelated corruption is
  reported without blocking a unique current request.
- [ ] Run the named tests and record each expected failure.
- [ ] Add one sequential catalog scan that fully verifies every canonical
  candidate, releases its arrays immediately, records valid semantic
  projections and safely classifies corrupt unrelated candidates.
- [ ] For each request, reuse a unique current candidate with no generator or
  publisher call; generate/publish only when no valid current candidate
  exists; reject ambiguity before generation.
- [ ] Never treat global discovery as a lock. Before trusting a unique reuse,
  fully reverify the selected exact canonical path/identity and reconfirm its
  projection; unrelated staging churn must not block. Let the publisher
  resolve winner-after-scan races for missing candidates.
- [ ] After publication, perform a fresh sequential final catalog scan and
  prove every requested projection has exactly one valid current identity,
  without retaining all verified arrays. Keep corrupt unrelated candidates in
  stable diagnostics rather than blocking `errors`.
- [ ] Run the complete campaign CLI test file to GREEN.

### Task 5: Verification and Report

**Files:**
- Modify:
  `.superpowers/sdd/2026-07-27-gsdiff-experiments-artifacts/task-3-report.md`

- [ ] Run focused data and CLI tests.
- [ ] Run the Task 3 joint gate:
  `pytest tests/data/test_artifacts.py tests/experiments/test_campaign_cli.py
  tests/experiments/test_identity.py -q`.
- [ ] Run the complete non-CUDA gate: `pytest -m "not cuda" -q`.
- [ ] Run `compileall`, Draft 2020-12 schema checks, canonical schema
  round-trips and `git diff --check`.
- [ ] Obtain an independent read-only review of the final diff.
- [ ] Append exact RED, GREEN, gate and review evidence to the Task 3 report.
- [ ] Do not commit.
