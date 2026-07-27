# GSDiff-SPI Publication Package Implementation Plan

> **For agentic workers:** REQUIRED SKILLS: use `nature-writing` for manuscript
> architecture, `nature-polishing` for the low-AI language and LaTeX-layout
> passes, `nature-figure` with the saved Python backend for every scientific
> figure, `nature-citation` plus broader primary-source academic search for
> claim support, `nature-ref-verifier` for final bibliography QA,
> `nature-reviewer` for the three-report pre-submission review, and `pdf` for
> rendered-page inspection. Use `superpowers:verification-before-completion`
> before claiming the package is publication-ready.

**Goal:** Build an evidence-locked, venue-neutral English computational-imaging
manuscript whose quantitative claims, figures, tables, references, prose, and
PDF layout are reproducible from the corrected experiment artifacts and ready
for journal-specific adaptation.

**Architecture:** The results lock is the only source for empirical numerical
results. Scientific-contract, configuration, code, dependency, data, and
checkpoint metadata come from their separately hashed provenance bundle. Tidy
figure-source data feed deterministic Python plotting scripts;
numerical-claim YAML feeds generated LaTeX macros and tables. A terminology
ledger and claim-evidence map govern prose. LaTeX builds a blinded review
package until authors, affiliations, funding, conflicts, data-license choices,
and target-journal metadata are supplied. Automated checks and rendered-page
inspection are followed by three simulated reviewer-emphasis reports, one
genuinely independent frozen-artifact review, and a resolution ledger.

**Tech Stack:** the exact versions frozen by Task 0: `latexmk` driving
`pdflatex` with `-no-shell-escape`, BibTeX, Python 3.12, matplotlib, NumPy,
pandas, PyYAML, Pillow, pypdf/pdfplumber, Poppler, and pytest 9.

## Publication Position and Confirmation Gate

Detected writing axes:

- paper type: `algorithmic`
- sections: title, abstract, introduction/related work, method, experiments,
  discussion, conclusion
- source language: mixed Chinese/English notes reconstructed into English
- journal: `generic` until the user names a venue
- primary readers: computational imaging and inverse-problems researchers
- figure backend: saved preference `Python`, used exclusively for plotting,
  preview, export, and figure visual QA

Provisional one-sentence argument:

> In dynamic single-pixel imaging, GSDiff-SPI jointly represents a canonical
> scene and rigid motion under a measurement-consistent optimization scheme;
> corrected multi-seed experiments will determine where its TV-regularized
> variant improves reconstruction and motion estimation, where a learned
> diffusion prior helps only in distribution, and where sampling or model
> mismatch limits both.

This sentence is a working boundary, not a result claim. It must be revised
only after the locked evidence exists. No performance direction, numerical
gain, novelty superlative, or causal explanation may be drafted before its
support is in the claim-evidence map.

The primary confirmation rule is imported verbatim from
`selection-policy-v1`: under scientific contract `gsdiff-sim-v1`, compare
`gsdiff_tv` with `recinr_se2` on `psnr_global_affine` across all nine primary
target-motion cells for each untouched seed `73` and `101`. A directional
"improves" claim is allowed only if both seed-level mean paired deltas are
strictly positive, both methods have complete finite coverage, and GSDiff-TV
has no greater failure count. This is a direction/coverage rule with
`minimum_effect_db=0.0`, not an inferential significance test. Report both raw
seed-level effects; a magnitude claim cannot exceed the smaller raw effect.
If any condition fails, the confirmatory performance claim is rejected.

## Preconditions and Red Lines

- Complete both earlier implementation plans and verify `results-lock-v1`.
- Never type a result number directly into manuscript prose, captions, or
  tables. Import it from generated LaTeX macros.
- Never cite a metadata-only candidate. Verify the abstract, full text, or
  publisher page and the bibliographic fields.
- Never invent authors, affiliations, funding, conflicts, licenses,
  contributions, acknowledgements, or repository DOI.
- Never hide OOD regressions, failed convergence, or measurement-budget
  collapse.
- Do not call the algorithm universally optimal. Use
  "grid-optimal under protocol v1" only when the finite search and confirmation
  establish it.
- No local selective contrast manipulation of reconstruction images. All
  image scaling is global, declared, and reproduced by script.
- AI-assisted text remains author-verification material. The scientific
  argument, claim boundaries, and interpretation must stay traceable to the
  project's evidence and the user's approval.
- Execute every native command through the fail-closed `Invoke-Checked` pattern
  in the completion gate; a later successful build/check may never mask an
  earlier nonzero exit.

## File Structure

Create:

```text
paper/
  main.tex
  supplement.tex
  metadata.yaml
  manuscript-settings.tex
  references.bib
  sections/
    00_title_abstract.tex
    01_introduction.tex
    02_results.tex
    03_discussion.tex
    04_methods.tex
    05_declarations.tex
  supplement/
    s01_methods.tex
    s02_full_grid.tex
    s03_ablations.tex
    s04_ood_failures.tex
    s05_reproducibility.tex
  claims/
    terminology-ledger.yaml
    argument-map.yaml
    numerical-claims.yaml
    numerical-claims.tex
    citation-claims.csv
    verified-references.yaml
    prose-audit.json
  legends/
    fig01.tex
    fig02.tex
    fig03.tex
    fig04.tex
    fig05.tex
    fig06.tex
    table01.tex
    table02.tex
    table_s01.tex
    table_s02.tex
    table_s03.tex
    table_s04.tex
  figures/
    fig01_overview.{svg,pdf,tiff,png}
    fig02_reconstruction.{svg,pdf,tiff,png}
    fig03_main_comparison.{svg,pdf,tiff,png}
    fig04_ablations.{svg,pdf,tiff,png}
    fig05_compute_convergence.{svg,pdf,tiff,png}
    fig06_ood_failures.{svg,pdf,tiff,png}
  figure_data/
    <locked tidy CSV/JSON from results-lock>
  tables/
    table01_main.tex
    table02_compute.tex
    table_s01_full_grid.tex
    table_s02_hyperparameters.tex
    table_s03_ablations.tex
    table_s04_failures.tex
  cards/
    data-card.md
    checkpoint-card.md
    code-card.md
    baseline-provenance.md
  qa/
    build-report.json
    figure-qa.json
    reference-verification.md
    claim-evidence-report.md
    pdf-layout-report.md
    anonymity-allowlist.yaml
    anonymity-attestation-pre-review.json
    submission-readiness.md
  reviews/
    review-1.md
    review-2.md
    review-3.md
    cross-review-synthesis.md
    resolution-ledger.md
  dist/
    gsdiff-spi-review.pdf
    gsdiff-spi-supplement.pdf
    source-data.zip
    review-source.zip
    reproducibility-bundle.zip
    reproducibility-manifest.json
scripts/publication/
  common_style.py
  validate_source_data.py
  generate_claim_macros.py
  generate_tables.py
  figure01_overview.py
  figure02_reconstruction.py
  figure03_main_comparison.py
  figure04_ablations.py
  figure05_compute_convergence.py
  figure06_ood_failures.py
  validate_figures.py
  lint_manuscript.py
  verify_citations.py
  build_paper.ps1
  render_pdf_pages.py
  scan_anonymity.py
  verify_publication_package.py
requirements-paper.lock
docs/reproducibility/paper-toolchain-lock.json
tests/publication/
  test_source_data.py
  test_claim_macros.py
  test_tables.py
  test_figures.py
  test_manuscript_lint.py
  test_citations.py
  test_latex_references.py
  test_package_manifest.py
  test_toolchain_smoke.py
  test_anonymity.py
```

Generated render intermediates go under `tmp/pdfs/` and are removed after the
QA report is complete. Only stable deliverables remain in `paper/dist/`.

## QA Dependency Graph and Invalidation Rules

The workflow is a strict directed acyclic graph:

```text
results/provenance locks
  -> validated evidence bundle
  -> claims, tables, figures, legends, and cards
  -> manuscript prose and citations
  -> deterministic LaTeX build
  -> rendered-page/anonymity/package QA
  -> frozen review candidate
  -> simulated reviews + genuinely independent review
  -> resolution and release candidate
```

Every generated file declares its direct input hashes and generator source
commit. A changed upstream hash invalidates every descendant. The verifier
computes the dependency closure and refuses a package containing a stale
descendant even if its path and file hash still exist. Any review-driven
change reruns all applicable claim, citation, figure, table, build, rendered
page, anonymity, and package checks; the candidate is refrozen and all affected
review criteria are re-rated. Narrowing a claim is a manuscript change, not an
automatic resolution.

## Global Figure Style Contract

Use one immutable style module:

```python
FINAL_WIDTH_MM = {"single": 85.0, "double": 178.0}
FONT_FAMILY = ["DejaVu Sans"]
FONT_SIZE_PT = {"body": 7.0, "small": 6.0, "panel": 8.0}
LINE_WIDTH_PT = {"axis": 0.8, "data": 1.2, "highlight": 1.8}
```

The conservative 85/178-mm widths remain readable under common 89/183-mm
two-column journal specifications. A venue adapter may enlarge them after the
target journal is named; it may not shrink text below the declared minimum.
Task 0 locks the exact `DejaVu Sans` file hash and the TeX text/math font files.
Runtime fallback to a same-named or platform-dependent font is forbidden; a
missing or mismatched font fails the build.

Fixed semantic palette:

```python
METHOD_COLORS = {
    "gsdiff_tv": "#0072B2",
    "gsdiff_diffusion": "#D55E00",
    "dgi": "#B3B3B3",
    "static_cs": "#7F7F7F",
    "perframe_cs": "#4D4D4D",
    "tv3d": "#009E73",
    "monin": "#CC79A7",
    "gidc3dtv": "#E69F00",
    "recinr": "#56B4E9",
    "siren": "#882255",
    "recinr_se2": "#332288",
}
```

Rules:

- white background, except grayscale reconstruction/error image plates;
- no rainbow colormap, 3D chart, pie chart, radar chart, or dual y-axis;
- red/green is never the only encoding; use marker, line, hatch, or direct
  label redundancy;
- top/right spines off; frameless legends; shared legends where possible;
- editable SVG text (`svg.fonttype = "none"`) and TrueType PDF text
  (`pdf.fonttype = 42`);
- vector SVG/PDF primary, 600-dpi TIFF and 300-dpi PNG secondary;
- panel labels are lowercase bold, 8 pt at final size;
- every quantitative legend names seeds, center, spread/CI, and metric version;
- figure scripts close every figure and are deterministic.

## Figure Evidence Contracts

### Figure 1: Measurement-consistent scene-motion reconstruction

- core conclusion: the method separates canonical scene, rigid motion, and
  measurement consistency, with the prior acting on rendered video rather than
  replacing the forward model;
- archetype: schematic-led composite;
- panels:
  - `a` temporal single-pixel acquisition and measurement stream;
  - `b` canonical 2D Gaussian scene plus SE(2) trajectory;
  - `c` differentiable rendering and SPI forward operator;
  - `d` SGD/HQS/ADMM loop with TV primary and diffusion secondary branches;
  - `e` outputs and GT-free held-out selection;
- risk: do not imply an unimplemented operation or hide that acceleration is a
  model extension/misspecification study.

### Figure 2: Representative reconstructions and motion

- core conclusion: representative image quality must agree with global-affine
  metrics and trajectory estimates under the same selected cells;
- archetype: image plate + quant;
- panels: GT/reconstruction/error/zoom for predeclared cells, followed by
  translation/rotation trajectories;
- identical crop coordinates and intensity scale across methods within a cell;
- error maps share one fixed scale; no per-method contrast optimization;
- the main calibrated plate uses the exact evaluation-only global-affine
  coefficients and clipping policy from `metrics-v1`; a hashed sidecar records
  slope, intercept, clipping bounds, and source run for every displayed method;
- raw uncalibrated reconstructions appear in the supplement whenever intensity
  fidelity or photometric bias is discussed;
- representative cells are selected by a predeclared rule (median seed or
  fixed seed), never by visual attractiveness.

### Figure 3: Main multi-seed comparison

- core conclusion: show the paired effect and its heterogeneity across target
  and motion, not merely a grand average;
- archetype: quantitative grid with one hero paired-effect panel;
- panels: global-affine PSNR, SSIM, nRMSE, per-cell paired differences,
  convergence/failure rate, and a compact method-rank view;
- seed-level points remain visible under summary intervals;
- selection seeds `7/11/42` and untouched confirmation seeds `73/101` are
  plotted and summarized separately; the two raw confirmatory paired effects
  remain visible, and any pooled five-seed view is labelled descriptive;
- no significance stars with five seeds; report sample SD and bootstrap CI.

### Figure 4: What drives the method

- core conclusion: the three or four predeclared decision-critical factors have
  bounded, protocol-dependent effects;
- archetype: one compact main-text decision figure;
- show candidate diagnostics on selection seeds only, then a separate panel
  with the frozen winner's two untouched confirmatory effects against the
  predeclared comparator; never imply every ablation candidate ran on
  confirmatory seeds;
- mark the selected setting and compute cap without hiding tied candidates.
- move the complete representation/solver/prior/warmup/temporal-weight/K/SNR/
  pattern/model-order/Gaussian-count grid to supplementary figures and tables;
  the main figure is at most four panels, no taller than 150 mm at 178-mm final
  width, with a caption budget of 220 words.

### Figure 5: Convergence and computational cost

- core conclusion: any quality gain must be interpreted with runtime, memory,
  parameter count, convergence, and failure rate;
- panels: residual-vs-iteration, quality-vs-runtime, peak VRAM, parameter count,
  and failure frequency;
- log axes only where declared and clearly labelled;
- do not connect failed or missing observations as if they converged.

### Figure 6: In-distribution, OOD, and failure boundaries

- core conclusion: learned-prior gains are distribution-dependent, while
  insufficient measurements and motion misspecification expose limits;
- main panels: ID/OOD paired effects, one representative OOD reconstruction
  comparison, and one measurement-budget failure sweep; the second OOD target,
  full K examples, and full diagnostic matrix move to the supplement;
- the main figure is at most four panels, no taller than 150 mm at 178-mm final
  width, with a caption budget of 220 words;
- keep negative diffusion effects visible;
- apply the predeclared failure rubric, multiple deterministic initializations,
  residual/convergence histories, and measurement-budget counterfactual before
  assigning a cause. Otherwise label evidence `undetermined` or
  `consistent with`; do not claim that representation failure, insufficient
  data, or optimizer non-convergence was causally identified.

## Task 0: Re-enter the Approved Workspace and Prove the Paper Toolchain

No manuscript, figure, or generated artifact work starts until this gate
passes.

**Files:**

- Create `requirements-paper.lock`.
- Create `docs/reproducibility/paper-toolchain-lock.json`.
- Create `tests/publication/test_toolchain_smoke.py`.
- Create minimal smoke fixtures under `tests/publication/fixtures/latex-smoke/`.

- [ ] **Step 1: Re-enter the approved isolated implementation workspace**

Reuse the worktree choice and provenance recorded by the correctness plan. If
this plan begins in a new normal checkout with no recorded preference, repeat
that plan's explicit worktree-consent gate. Record the starting clean commit
and SHA-256 of the results lock, publication-artifacts manifest, protocol
bundle, environment lock, and this plan.

- [ ] **Step 2: Probe without installing**

Record exact executable paths and versions for `latexmk`, `pdflatex`, `bibtex`,
`pdffonts`, `pdfimages`, `pdfinfo`, `pdftoppm`, and the paper Python
dependencies. The July 2026 planning probe found no LaTeX toolchain and missing
`pandas`/`pdfplumber`; treat that as an expected blocker to verify, not as a
license to install globally. If required tooling is absent, report the exact
gap and obtain explicit user authorization before installing it. Prefer a
project-local or otherwise version-pinned toolchain.

- [ ] **Step 3: Freeze one exact build backend**

The v1 build uses BibTeX and this `latexmk` contract:

```text
latexmk -pdf -bibtex -halt-on-error \
  -pdflatex="pdflatex -interaction=nonstopmode -halt-on-error -file-line-error -no-shell-escape %O %S"
```

Record executable SHA-256/version, TeX package-file hashes, Python wheels or
exact installed distributions, Poppler version, fonts plus font-file hashes,
locale, time zone, and platform in `paper-toolchain-lock.json`. The build
environment sets `LC_ALL=C`, `TZ=UTC`, `PYTHONHASHSEED=0`, a fixed
`SOURCE_DATE_EPOCH`, and a declared PDF trailer/metadata-ID policy.

- [ ] **Step 4: Compile and render a representative smoke document**

The fixture contains Unicode author-neutral text, equations, cross-references,
one BibTeX citation, one generated macro, a pre-generated PDF figure, and a
table. The same figure's SVG is parsed and rendered separately by figure QA;
the `pdflatex` build never invokes SVG/Inkscape conversion or shell escape.
Compile with the exact command, render every page, and prove selectable text,
embedded fonts, working citation/reference resolution, no shell escape,
expected page boxes, and deterministic semantic PDF content across two clean
builds. Exact PDF bytes are required only when the locked backend supports the
declared deterministic metadata/ID policy; otherwise record both byte hashes
and require an equal normalized semantic hash.

- [ ] **Step 5: Verify and commit**

```powershell
D:\conda\envs\spi\python.exe -m pytest tests/publication/test_toolchain_smoke.py -q
git diff --check
git add requirements-paper.lock docs/reproducibility/paper-toolchain-lock.json tests/publication/test_toolchain_smoke.py tests/publication/fixtures/latex-smoke
git commit -m "build: freeze publication toolchain"
```

## Task 1: Establish Evidence Interfaces and Manuscript Metadata

**Files:**

- Create `paper/metadata.yaml`.
- Create `paper/claims/terminology-ledger.yaml`.
- Create `paper/claims/argument-map.yaml`.
- Create the four files under `paper/cards/`.
- Create `scripts/publication/validate_source_data.py`.
- Create `tests/publication/test_source_data.py`.

- [ ] **Step 1: Write RED source-lock tests**

Reject figure data when the results-lock hash differs, a row lacks a source run
ID, a run manifest is incomplete, a metric version is not `metrics-v1`, or a
claimed seed/cell cannot be traced to an immutable dataset. Require and validate
the locked `publication-artifacts-v1` manifest, including every raw
reconstruction/GT/error array, trajectory, iteration history, K-sweep example,
failure diagnostic, schema, shape, dtype, units/range, source identity, and
content hash needed by Figures 2, 5, and 6. A tidy metric table cannot
substitute for missing non-tabular evidence.

- [ ] **Step 2: Build the terminology ledger**

At minimum resolve canonical forms and first-use definitions for:

```text
single-pixel imaging (SPI)
GSDiff-SPI
canonical scene
two-dimensional Gaussian splatting
special Euclidean group SE(2)
three-dimensional total variation (3D-TV)
plug-and-play diffusion prior
held-out measurement residual
global-affine calibration
out of distribution (OOD)
```

Record variants from README, THEORY, source identifiers, config names, and
legacy Claude notes. One concept receives one name throughout.

- [ ] **Step 3: Build the argument map**

For every planned paragraph, record one job:
`context`, `gap`, `approach`, `result`, `comparison`, `mechanism`,
`implication`, or `limitation`.

For every major claim:

```yaml
claim_id: C001
text: ""
section: ""
evidence:
  - run_or_figure_id: ""
status: supported | inferred | needs-evidence | rejected
boundary: ""
```

The initial status of numerical/result claims is `needs-evidence`. Promotion
requires the results lock.

- [ ] **Step 4: Separate known and missing metadata**

`metadata.yaml` supports an anonymous review build. Missing authors,
affiliations, funding, conflicts, contributions, target venue, license, and
repository DOI appear in a machine-readable `submission_blockers` list. They
must not be replaced with fabricated placeholder people or organizations.

Also represent, without guessing: corresponding-author contact, ORCIDs,
keywords, acknowledgements, code/data availability and licenses, ethics
applicability, human/animal consent applicability, third-party permissions, and
the target venue's AI-assistance disclosure. Applicability fields are
`confirmed-applicable`, `confirmed-not-applicable`, or
`unresolved-review-required`; never auto-fill `N/A`.

- [ ] **Step 5: Build provenance cards before scientific drafting**

The data, checkpoint, code, and baseline cards record purpose/scope,
source/provenance and license state, generation/acquisition command, exact
hashes, expected size and untracked location, preprocessing,
train/holdout/confirmation use, limitations/prohibited claims, and verification
command. The checkpoint card distinguishes the learned diffusion checkpoint
from GSDiff-TV. The baseline card records ReCINR upstream commit
`9149d1d228db2e4eb3ae852a004f1d9e95ee0229`, local changes, authorship, and
license status. Unresolved redistribution rights remain a blocker. Never print
credentials, private drive roots, usernames, or network paths.

- [ ] **Step 6: Verify and commit**

```powershell
D:\conda\envs\spi\python.exe -m pytest tests/publication/test_source_data.py -q
git add paper/metadata.yaml paper/claims paper/cards scripts/publication/validate_source_data.py tests/publication/test_source_data.py
git commit -m "docs: establish manuscript evidence interfaces"
```

## Task 2: Generate Numerical Claims and Tables from Locked Data

**Files:**

- Create `scripts/publication/generate_claim_macros.py`.
- Create `scripts/publication/generate_tables.py`.
- Create `tests/publication/test_claim_macros.py`.
- Create `tests/publication/test_tables.py`.
- Generate `paper/claims/numerical-claims.yaml`.
- Generate `paper/claims/numerical-claims.tex`.
- Generate all files under `paper/tables/`.

**Interface:**

```python
def load_locked_evidence(lock_path: Path) -> EvidenceBundle:
    ...

def make_numerical_claims(bundle: EvidenceBundle) -> dict[str, object]:
    ...

def render_latex_macros(claims: Mapping[str, object]) -> str:
    ...

def render_tables(bundle: EvidenceBundle) -> Mapping[str, str]:
    ...
```

- [ ] **Step 1: Write RED provenance tests**

Each macro and table cell must map to exact source rows. Tests intentionally
alter one CSV value and assert lock verification fails rather than silently
changing the paper.

- [ ] **Step 2: Use explicit macro names**

Examples:

```latex
\newcommand{\SelectionGSDiffTVPSNRMean}{...}
\newcommand{\SelectionGSDiffTVPSNRSD}{...}
\newcommand{\ConfirmVOneGSDiffTVVersusReCINRSETwoSeedSeventyThreePSNRDelta}{...}
\newcommand{\ConfirmVOneGSDiffTVVersusReCINRSETwoSeedOneHundredOnePSNRDelta}{...}
\newcommand{\ConfirmVOneGSDiffTVVersusReCINRSETwoPSNRMean}{...}
\newcommand{\DescriptiveVOneGSDiffTVPooledFiveSeedPSNRMean}{...}
```

Generated comments include claim ID and source-data hash. The manuscript may
only use named macros, not copied numbers. Selection (`7/11/42`), confirmation
(`73/101`), and pooled-five-seed descriptive summaries have distinct macro
prefixes and cannot be substituted for each other.

All numeric inputs are parsed as finite `Decimal` values with declared units,
range, precision, and missing-value policy. Text fields pass a centralized
LaTeX-escaping function. Generated TeX uses an allowlist of macro names and
literal-safe arguments; reject backslashes/control sequences, `\input`,
`\include`, `\write18`, path traversal, and unexpected braces from data fields.
Tests inject `%`, `&`, `_`, `#`, braces, Unicode, NaN/Infinity, and TeX payloads
to prove escaping or refusal.

- [ ] **Step 3: Generate tables with deterministic ordering**

- Table 1: all main methods, global-affine PSNR/SSIM/nRMSE with selection and
  confirmation blocks; the two raw confirmation effects are shown, while a
  pooled five-seed block is explicitly labelled descriptive;
- Table 2: runtime, VRAM, parameters, convergence/failure;
- Supplementary Table 1: complete Claude grid;
- Supplementary Table 2: exact method hyperparameters and selection objective;
- Supplementary Table 3: all selection-seed ablations plus a separate block for
  the frozen winner's two confirmatory effects;
- Supplementary Table 4: OOD/failure cases.

Bold only a uniquely best grid value. Ties under declared tolerance receive
the same typographic treatment. Do not bold legacy metrics.

- [ ] **Step 4: Verify round-trip**

Parse generated TeX values back to decimals and compare with locked source
within formatting tolerance. Check no `nan`, `inf`, missing cell, manual
`\SI{number}` result, or inconsistent decimal precision remains.

Generate the six table captions as separate source-owned files under
`paper/legends/`, and define the figure-legend schema/validator only. Figure
legends are written after the final panel/data/display contracts in Tasks 4 and
5. Contract tests require every completed legend/caption to stand alone:
define abbreviations, identify panels, state conditions/comparators, sample
grain and exact `n`, center/spread/CI, seed split, metric/alignment version,
error-map scaling, and source-data/lock identifier as applicable.

- [ ] **Step 5: Verify and commit**

```powershell
D:\conda\envs\spi\python.exe -m pytest tests/publication/test_claim_macros.py tests/publication/test_tables.py -q
git add scripts/publication paper/claims/numerical-claims.yaml paper/claims/numerical-claims.tex paper/tables paper/legends/table*.tex tests/publication
git commit -m "data: generate manuscript claims and tables"
```

## Task 3: Implement the Shared Python Figure System

**Files:**

- Create `scripts/publication/common_style.py`.
- Create `scripts/publication/validate_figures.py`.
- Create `tests/publication/test_figures.py`.
- Create figure-contract records inside `paper/qa/figure-qa.json`.

- [ ] **Step 1: Write RED style tests**

Assert saved backend is Python, required packages import, final size is exact,
SVG contains selectable `<text>`, PDF fonts are embedded, PNG/TIFF dimensions
match DPI, panel labels exist, and no disallowed colormap/chart type is used.
Parse SVG with namespace-aware XML: require all intended labels to remain text,
resolve transforms and effective font sizes at final dimensions, reject
unexpected external links/scripts and whole-panel rasterization, and verify
declared raster image panels are embedded with sufficient effective DPI. Use
`pdffonts`/`pdfimages` for the PDF and fail on unembedded/substituted fonts,
unexpected image downsampling, or undeclared color spaces.

- [ ] **Step 2: Implement style and export helpers**

```python
def new_figure(width: str, height_mm: float, **kwargs):
    ...

def label_panel(ax, label: str) -> None:
    ...

def save_publication_figure(
    fig,
    stem: Path,
    *,
    source_data_sha256: str,
    contract: Mapping[str, object],
) -> None:
    ...
```

Exports SVG first, then PDF, TIFF, and PNG from the same Python figure object.
Write a sidecar JSON with dimensions, data hash, script hash, palette, fonts,
font-file hashes, matplotlib/Python versions, locale/time zone, deterministic
seed, normalized semantic-render hash, and integrity notes. Source-owned inputs
(contracts, crop choices, legend prose) are never overwritten by generators;
generated outputs carry a header and are replaced only by their owning script.

- [ ] **Step 3: Add render validation**

Rasterize exported SVG/PDF using Python-compatible tooling or Poppler for
inspection only; do not redraw with another plotting backend. Check cropped
content bounds, blank output, clipped text, overlaps, and final-size
legibility. Run two clean exports and require identical source-data/script
hashes and normalized render hashes; require byte equality for SVG/raster
outputs after deterministic metadata normalization.

- [ ] **Step 4: Verify and commit**

```powershell
D:\conda\envs\spi\python.exe -m pytest tests/publication/test_figures.py -q
git add scripts/publication/common_style.py scripts/publication/validate_figures.py tests/publication/test_figures.py paper/qa/figure-qa.json
git commit -m "feat: add deterministic publication figure system"
```

## Task 4: Produce Figures 1 and 2

**Files:**

- Create `scripts/publication/figure01_overview.py`.
- Create `scripts/publication/figure02_reconstruction.py`.
- Generate Figure 1 and Figure 2 bundles.

- [ ] **Step 1: Encode the two figure contracts**

Each contract contains conclusion, archetype, final size, panel map, evidence
hierarchy, statistics/source data, integrity notes, and reviewer risks.

- [ ] **Step 2: Draw Figure 1 from implemented components**

Use simple vector geometry and mathematical labels in matplotlib. Every arrow
and branch must map to actual code. Compare the schematic with `train.py`,
forward model, scene, motion, solver, and prior code before approval.

- [ ] **Step 3: Assemble Figure 2 from immutable arrays**

The script loads only the predeclared representative cells and crop
coordinates. It applies the same global-affine transformation used by
`metrics-v1`, records slope/intercept/clipping and source identity in a hashed
sidecar, and uses a shared error scale. It never fits display calibration from
the selected crop. Generate a supplementary raw-output plate with no affine
display calibration when photometric fidelity is mentioned.

- [ ] **Step 4: Inspect at final size**

Open PNG previews and PDF/SVG exports. Confirm no misleading crop, clipped
label, invisible error contrast, inconsistent trajectory scale, or rasterized
text.

- [ ] **Step 5: Write and validate Figures 1-2 legends**

Write the two source-owned legends only after the final exports and sidecars
exist. Run the self-contained legend schema and verify each panel, calibration,
scale, `n`, metric, and source hash matches the figure contract.

- [ ] **Step 6: Commit**

```powershell
git add scripts/publication/figure01_overview.py scripts/publication/figure02_reconstruction.py paper/figures/fig01_* paper/figures/fig02_* paper/legends/fig01.tex paper/legends/fig02.tex paper/qa/figure-qa.json
git commit -m "fig: add method and reconstruction evidence"
```

## Task 5: Produce Figures 3-6

**Files:**

- Create the four remaining figure scripts.
- Generate Figure 3-6 bundles.

- [ ] **Step 1: Render Figure 3 with seed-level observations**

Do not use bars alone. Show paired seed points or compact interval plots and
retain target/motion heterogeneity. Any aggregate view must link to the same
observations.

- [ ] **Step 2: Render Figure 4 with selection/confirmation separation**

Use different marker fill or panel separation for selection and untouched
confirmation. Never join them into one tuning curve. Limit the main-text figure
to the three or four decision-critical panels declared by its contract; export
the remaining locked axes as supplementary figures. Only the frozen selected
configuration appears on seeds `73/101`; do not fabricate a confirmatory
candidate sweep.

- [ ] **Step 3: Render Figure 5 with honest failures**

Failed runs are explicit marks or counts, not dropped records. Runtime/VRAM
axes state hardware and inclusion policy.

- [ ] **Step 4: Render Figure 6 with OOD and failure boundaries**

Use the same method palette and metric scale as the main comparison. Negative
effects stay negative and visible. Limit the main-text figure to at most four
panels; export the second target/full K grid/diagnostic matrix to the
supplement. Only rubric-supported multi-initialization/counterfactual diagnoses
receive causal labels; otherwise display `undetermined` or `consistent with`.

- [ ] **Step 5: Run figure QA**

For all six figures verify:

- each panel answers a unique question;
- font is readable at 100% final width;
- shared scales are actually shared;
- uncertainty and `n` are defined;
- source-data sidecars resolve to locked files;
- PDF/SVG text is editable;
- no image was contrast-tuned per method;
- each separately owned legend file passes the self-contained legend contract;
- main Figures 4 and 6 satisfy the declared height/panel/caption budgets.

- [ ] **Step 6: Write and validate Figures 3-6 legends**

Write the four source-owned legends from the final figure contracts and
sidecars, then require exact panel/encoding/seed split/statistic/source-hash
agreement.

- [ ] **Step 7: Commit**

```powershell
git add scripts/publication/figure03_main_comparison.py scripts/publication/figure04_ablations.py scripts/publication/figure05_compute_convergence.py scripts/publication/figure06_ood_failures.py paper/figures paper/legends/fig03.tex paper/legends/fig04.tex paper/legends/fig05.tex paper/legends/fig06.tex paper/qa/figure-qa.json
git commit -m "fig: add comparison ablation and boundary evidence"
```

## Task 6: Build and Verify the Literature Base

**Files:**

- Create `paper/claims/citation-claims.csv`.
- Create `paper/claims/verified-references.yaml`.
- Create `paper/references.bib`.
- Create `paper/qa/reference-verification.md`.
- Create `scripts/publication/verify_citations.py`.
- Create `tests/publication/test_citations.py`.

- [ ] **Step 1: Inventory existing literature**

Treat repository notes and PDFs as discovery candidates, not verified support.
Record whether the full text was actually read, which claim it supports, and
whether the source is primary or review.

- [ ] **Step 2: Search claim by claim**

For high-level CNS/Nature-family context, use strict `CNS及子刊` scope with
date limits stated at search time. For domain completeness, separately search
primary SPI, dynamic computational imaging, Gaussian representations,
inverse-problem optimization, plug-and-play priors, diffusion restoration, and
relevant baseline papers without an artificial publisher restriction.

Use only primary/official sources for technical assertions. Search the opposite
result or limitation as well as confirming evidence.

- [ ] **Step 3: Grade support**

For each citable segment record:

```text
claim ID
exact claim text
canonical claim-text SHA-256
paper
BibTeX key and DOI
support grade: strong / partial / background / limiting
evidence basis: abstract / full text / publisher page
exact support boundary
insertion location
search and verification dates
```

Metadata-only candidates cannot enter `references.bib`.

- [ ] **Step 4: Verify bibliographic fields from multiple sources**

Check author order, title, year, journal, volume, issue, pages/article number,
and DOI. Classify every item as Verified, Check suggested, Needs fix, or
Unverifiable. `Needs fix` must be zero; `Unverifiable` must be explicitly
justified or removed.

Before finalizing, query authoritative publisher/Crossref/retraction sources
for corrections, expressions of concern, or retractions; record the check date
and outcome. Detect duplicate DOI/title records, duplicate BibTeX keys, cited
keys absent from the bibliography, uncited/orphan bibliography entries, and
claim rows whose exact text hash no longer matches the manuscript. Final QA is
bidirectional: every citation supports at least one mapped claim, and every
citation-requiring claim has one or more verified supporting/limiting sources.

`verify_citations.py` works at each in-text occurrence or citation cluster, not
merely at BibTeX-key level. Every occurrence maps current segment ID,
canonical segment-text SHA-256, cited key/DOI, support grade/boundary, and
verification date. It rejects stale text hashes, missing or misplaced support,
uncited keys, orphan records, and one paper being valid for one sentence but
silently reused for an unsupported second sentence. Task 6 tests the mechanism
on fixtures; the full manuscript audit runs after Task 9 and after every review
revision.

- [ ] **Step 5: Check intellectual debt and licenses**

Credit upstream baselines and source methods precisely. Verify permissions for
any adapted conceptual material. Prefer original schematics to adapted images.

- [ ] **Step 6: Commit**

```powershell
git add paper/claims/citation-claims.csv paper/claims/verified-references.yaml paper/references.bib paper/qa/reference-verification.md scripts/publication/verify_citations.py tests/publication/test_citations.py
git commit -m "docs: verify manuscript claims and references"
```

## Task 7: Draft Methods and Results from Evidence

**Files:**

- Create `paper/sections/02_results.tex`.
- Create `paper/sections/04_methods.tex`.
- Create the five supplement sections.

- [ ] **Step 1: Confirm the final one-sentence argument**

Update the provisional argument from the locked main, confirmatory, OOD, and
failure evidence. If no stable advantage survives confirmation, frame the
paper around the joint representation/diagnostic finding rather than inventing
a performance win. Present the exact proposed sentence, its claim IDs, and its
evidence boundaries to the user for explicit author approval. Drafting may
continue provisionally, but the Results/Abstract/title cannot be frozen or
released until that approval is recorded in the argument map.

- [ ] **Step 2: Draft Methods in reproducible order**

1. dynamic SPI task formulation and notation;
2. canonical 2D Gaussian scene;
3. SE(2) motion and optional acceleration/misspecification;
4. differentiable temporal measurement operator;
5. SGD/HQS/ADMM optimization;
6. corrected isotropic proximal TV and distinct anisotropic soft TV;
7. diffusion prior as a secondary PnP branch;
8. data generation and locked protocol;
9. GT-free held-out selection;
10. metrics, statistics, runtime, and software versions.

Every module states motivation, mechanism, and ablation role. Remove vague
phrases such as "standard settings" or "the model was validated".
State the method-child/evaluator separation, the blind-tuning audit, and which
quantities were visible to selection. Data/checkpoint/code/baseline cards and
license/provenance checks from Task 1 must already pass before this step.

- [ ] **Step 3: Draft Results as an evidence ladder**

1. numerical/operator validation;
2. main five-seed comparison;
3. representation/solver/prior ablations;
4. measurement and noise robustness;
5. compute/convergence;
6. OOD behavior and failure cases.

Each subsection opens with one bounded claim, names conditions and comparator,
then cites a figure/table and generated numerical macros.
Selection-seed findings and the two untouched confirmation effects occupy
separate paragraphs and macros. A pooled five-seed value, if shown, is
descriptive only.

- [ ] **Step 4: Draft supplement**

Include the complete Claude grid, all hyperparameters, full per-seed results,
all ablations, failed cells, provenance, metric definitions, and reproduction
commands. Do not use the supplement to hide a result needed for a main claim.

- [ ] **Step 5: Run claim-evidence audit**

Every result sentence maps to a macro, figure, table, or explicit qualitative
observation. Results report what occurred; interpretation stays in Discussion.

- [ ] **Step 6: Commit**

```powershell
git add paper/sections/02_results.tex paper/sections/04_methods.tex paper/supplement paper/claims/argument-map.yaml paper/qa/claim-evidence-report.md
git commit -m "docs: draft evidence-locked methods and results"
```

## Task 8: Draft Introduction, Discussion, Abstract, and Title

**Files:**

- Create remaining section files.
- Create `paper/main.tex`, `paper/supplement.tex`, and
  `paper/manuscript-settings.tex`.

- [ ] **Step 1: Draft the Introduction**

Use the open-with-challenge variant:

1. why dynamic SPI is difficult under temporal multiplexing;
2. why static reconstruction or framewise recovery loses scene-motion
   structure;
3. what implicit/explicit representations and priors solve and leave open;
4. the exact bounded contribution and evaluation scope.

Fold related work by technical topic unless the target journal later requires a
separate section.
Immediately before submission adaptation, re-open the current official author
guide for the named venue and record the access date, template/version, length,
figure, reference, data/code, AI-disclosure, and anonymity requirements. Until
then, do not imply generic formatting equals venue acceptance.

- [ ] **Step 2: Draft the Discussion**

Interpret the scene-motion representation, corrected prior behavior,
distribution-dependent diffusion crossover, finite search result, compute
tradeoff, and failure modes. Address rival explanations such as optimization
conditioning, representation capacity, and metric calibration. Do not repeat
Results panel by panel.

- [ ] **Step 3: Draft the Conclusion**

Use contribution, decisive evidence, implication, and boundary. Add no new
data and no universal claim.

- [ ] **Step 4: Draft Abstract last**

Use context/problem, gap, approach, strongest confirmed result, implication,
and boundary. Include at least one generated quantitative comparison only if
the exact primary confirmation rule passes; show no significance implication
from two confirmatory seeds.

- [ ] **Step 5: Generate and audit titles**

Produce 3-5 concrete candidates, then select the most defensible with the
terminology ledger and evidence. Avoid "novel", "revolutionary", "toward", or
a quantitative adjective unsupported by the full grid.

- [ ] **Step 6: Build anonymous review LaTeX**

Use standard packages only, keep style venue-neutral, and isolate all
venue-dependent margins/fonts/caption settings in
`manuscript-settings.tex`. Missing submission metadata is listed in
`metadata.yaml`, not fabricated in TeX.

- [ ] **Step 7: Commit**

```powershell
git add paper/main.tex paper/supplement.tex paper/manuscript-settings.tex paper/sections paper/claims/argument-map.yaml
git commit -m "docs: complete the evidence-bounded manuscript"
```

## Task 9: Remove AI-Like Prose Without Flattening the Author's Voice

**Files:**

- Create `scripts/publication/lint_manuscript.py`.
- Create `tests/publication/test_manuscript_lint.py`.
- Generate `paper/claims/prose-audit.json`.

**Structural order:** paper type → section job → paragraph logic →
claim/evidence/boundary → sentence polish.

- [ ] **Step 1: Write RED lint tests**

Hard errors:

- `In recent years`, `has garnered considerable attention`,
  `plays a pivotal/crucial role`, `it is worth noting`,
  `in conclusion`, `underscores the importance`, `groundbreaking`, or
  `revolutionary` as unconditionally banned boilerplate/marketing phrases;
- inconsistent method/metric terms against the terminology ledger;
- bare result numbers not expressed through generated macros;
- unresolved `TODO`, `TBD`, `XX`, placeholder citation, or tool token.

Advisory findings include em dashes, repeated three-part rhetoric, repeated
paragraph openers, generic throat-clearing, heavy nominalization, unusually
uniform sentence length, conclusion-like restatement at every subsection,
`remarkable`/`comprehensive`/`novel`, marketing verbs such as
`leverages`/`empowers`/`enables`, and `significant/significantly`. Contextual
terms become allowable only through a structured allowlist citing a bounded
novelty claim ID, defined test ID, or literal technical meaning; regex must not
guess semantics. The linter never fails solely on a probabilistic "AI-like"
score and never mechanically rewrites long or stylistically unusual sentences.
A human reviews each advisory item; variation in rhythm is preferable to a
uniform synthetic cadence.

- [ ] **Step 2: Reverse-outline every paragraph**

Record one sentence per paragraph describing its job. Split paragraphs with
two jobs; delete paragraphs with no role in the argument.

- [ ] **Step 3: Polish from Chinese/technical notes into direct English**

Translate intent, not syntax. Prefer concrete subjects and verbs, explicit
comparators, precise conditions, and evidence-calibrated hedging. Keep a stable
authorial stance (`we` for author actions where appropriate) rather than
oscillating between passive and impersonal constructions.

- [ ] **Step 4: Perform a human-sounding read-aloud pass**

Check that adjacent sentences differ naturally in length and construction,
transitions state real logical relations, and the prose does not sound like a
template. Do not introduce colloquialisms or reduce technical precision merely
to evade an AI detector.

- [ ] **Step 5: Refresh evidence/citation maps, then verify**

After the final prose pass, rebuild every citable segment and canonical text
hash. Refresh `citation-claims.csv`, remove stale/orphan occurrences, verify
new or materially changed claims against full text/authoritative sources, and
update `verified-references.yaml`, `references.bib`, and
`reference-verification.md` when needed. Re-run the whole-manuscript
claim-evidence audit and update `argument-map.yaml` plus
`claim-evidence-report.md` before invoking strict verifiers; do not merely
rewrite hashes to silence a stale-map failure.

```powershell
D:\conda\envs\spi\python.exe scripts\publication\lint_manuscript.py --paper paper --strict
D:\conda\envs\spi\python.exe -m pytest tests/publication/test_manuscript_lint.py -q
D:\conda\envs\spi\python.exe scripts\publication\verify_citations.py --paper paper --occurrence-level --strict
D:\conda\envs\spi\python.exe -m pytest tests/publication/test_citations.py -q
```

- [ ] **Step 6: Commit**

```powershell
git add scripts/publication/lint_manuscript.py tests/publication/test_manuscript_lint.py paper/sections paper/claims/prose-audit.json paper/claims/citation-claims.csv paper/claims/verified-references.yaml paper/claims/argument-map.yaml paper/references.bib paper/qa/reference-verification.md paper/qa/claim-evidence-report.md
git commit -m "docs: tighten manuscript logic and authorial voice"
```

## Task 10: Freeze Provenance, Metadata, and Reproducibility Inputs

**Files:**

- Create `paper/dist/reproducibility-manifest.json`.
- Modify and validate `paper/metadata.yaml` and all four cards from Task 1.

- [ ] **Step 1: Revalidate cards against locked evidence**

Verify every card hash against disk or mark the asset externally unavailable
with a reproducible creation path. Recheck licenses/permissions and the blind
tuning audit. Never emit credentials, drive roots, usernames, email addresses,
UNC paths, or private repository URLs into the anonymous package.

- [ ] **Step 2: Resolve or retain exact submission blockers**

Validate the author-supplied fields named in Task 1. The anonymous build may
omit identity fields, but the full metadata record must distinguish redacted,
unresolved, and confirmed values. Unresolved authorship, corresponding-author
contact, ORCID, funding, conflicts, contributions, acknowledgements, ethics/
consent applicability, permissions, licenses, AI disclosure, target venue, and
repository/DOI decisions remain explicit blockers; do not auto-convert them to
`N/A`.

- [ ] **Step 3: Write the reproducibility manifest**

Record `candidate_source_commit`, experiment/aggregation/lock-tool commits,
results lock, scientific-contract/config/environment/toolchain hashes,
generated-file DAG, cards, figures/tables/legends, manuscript sources, and
exact regeneration commands. This Task 10 value is the candidate parent, not
the final `source_commit` defined after review resolution in Task 13. Do not
include a self-referential hash or a future commit. Task 13 regenerates/finalizes
the manifest from its clean source export and attests the relationship between
the two commits.

- [ ] **Step 4: Commit**

```powershell
git add paper/metadata.yaml paper/cards paper/dist/reproducibility-manifest.json
git commit -m "docs: freeze publication provenance and metadata status"
```

## Task 11: Compile and Visually Inspect Main and Supplement PDFs

**Files:**

- Create `scripts/publication/build_paper.ps1`.
- Create `scripts/publication/render_pdf_pages.py`.
- Create `scripts/publication/scan_anonymity.py`.
- Create `tests/publication/test_latex_references.py`.
- Create `tests/publication/test_anonymity.py`.
- Create `paper/qa/anonymity-allowlist.yaml`.
- Generate `paper/qa/anonymity-attestation-pre-review.json`.
- Generate `paper/qa/build-report.json`.
- Generate `paper/qa/pdf-layout-report.md`.

- [ ] **Step 1: Reverify the frozen toolchain**

Run Task 0's smoke and compare every executable/package/font hash with
`paper-toolchain-lock.json`. No installation, fallback backend, template
substitution, or version drift is allowed at this stage.

- [ ] **Step 2: Compile in the correct order**

Use the exact locked `latexmk`/BibTeX/`pdflatex -no-shell-escape` command.
Cross-document references are one-way only: the supplement is self-contained
and may not import main-document labels; the main document may import the
supplement's fixed `paper/build/supplement.aux` via the locked external-reference
mechanism. Compile supplement first, then main to reference convergence. Reject
main↔supplement cycles, unexpected `.aux` locations, or stale auxiliary hashes.
Fail on undefined references/citations, multiply defined labels, missing
figures, missing macros, `Float too large`, overfull boxes beyond the declared
tolerance, underfull boxes above the declared badness threshold, substituted or
unembedded fonts, duplicate destinations, malformed URLs, or nonzero LaTeX
exit. The warning allowlist names exact benign messages and maximum counts; no
blanket warning suppression is permitted.

- [ ] **Step 3: Extract logical checks**

Verify page count, selectable text, embedded fonts, figure count/order, table
count/order, equation and reference numbering, bookmarks, page size, and that
no submission blocker accidentally appears as visible placeholder text.
Inspect MediaBox/CropBox/BleedBox consistency, PDF Info/XMP policy, image color
spaces, effective DPI, font substitution, active links/URLs, annotations, and
attachments. Compare two clean builds under the locked locale/time/epoch/font
environment using both byte hash (when promised) and normalized semantic hash.

- [ ] **Step 4: Pass the pre-review anonymity scan**

Require an author-confirmed sensitive-identity denylist supplied outside the
repository/package (names and variants, emails, ORCIDs, affiliations,
acknowledgements/grants, Git author/config/history tokens, usernames, private
paths/URLs). Record only its hash and confirmation status. Without it,
anonymous readiness is blocked.

Write `anonymity-attestation-pre-review.json` containing the reviewed candidate SHA-256,
scanner source SHA-256/version plus parent commit, external denylist SHA-256,
tracked allowlist SHA-256, complete scan scope, timestamp, and pass/fail counts. The sensitive
denylist contents remain outside Git and every deliverable. A changed
denylist/allowlist/scanner/candidate invalidates the attestation.

Scan PDF Info/XMP, annotations, attachments, bookmarks, links, and embedded
file names, plus TeX, BibTeX, SVG/XML, JSON/YAML/CSV/MD, scripts, cards, logs,
and manifests. Structured cited-author fields in `references.bib` and neutral
in-text citations are allowed as reference metadata; project-identity tokens
outside that exact context are not. Also scan for credentials, drive/UNC paths,
and private repository tokens. Only a clean scan may be frozen or sent for
Task 12 review.

- [ ] **Step 5: Render every page to PNG**

Use `tmp/pdfs/`, build contact sheets, then inspect every full-resolution page:

- no clipped/overlapping text;
- no black squares or broken symbols;
- no unreadable figure labels;
- no orphaned headings or captions;
- no half-empty float pages caused by wide-short figures;
- no figure/caption split that harms reading;
- even main/SI page density;
- references and URLs remain human-readable.

- [ ] **Step 6: Iterate layout**

Change → compile → render → inspect. Regenerate an overly wide/short figure
taller at the Python source instead of stretching it in LaTeX. Use top-aligned
float glue and surgical `\clearpage`/`[H]` only when measured layout requires
it. Do not blanket-force floats or use landscape unless a genuinely
unsquarable supplementary artifact remains.

Any source/layout change invalidates both clean builds, all page renders, the
layout report, anonymity scan, and the frozen review candidate.

- [ ] **Step 7: Commit**

```powershell
git add scripts/publication/build_paper.ps1 scripts/publication/render_pdf_pages.py scripts/publication/scan_anonymity.py tests/publication/test_latex_references.py tests/publication/test_anonymity.py paper/qa/build-report.json paper/qa/pdf-layout-report.md paper/qa/anonymity-allowlist.yaml paper/qa/anonymity-attestation-pre-review.json
git commit -m "docs: verify manuscript pdf layout"
```

The reviewed PDF bytes remain in the ignored/local candidate directory with
their hashes in QA records; they are not tracked until the optional Task 13
artifact-commit authorization.

## Task 12: Run Simulated Assessments, Obtain Independent Review, and Resolve

**Files:**

- Generate all files under `paper/reviews/`.
- Generate `paper/qa/submission-readiness.md`.

Use `nature-reviewer` on one shared frozen manuscript fact base with three
different emphases, without calling the reports independent and without
inventing reviewer identities, institutions, or biographies:

1. originality, significance, and computational-imaging readership;
2. numerical soundness, inverse-problem formulation, and algorithmic fairness;
3. reproducibility, statistics, figures, clarity, and boundary reporting.

Each report includes artifact hashes, overall assessment, interested
readership, summary of contribution, strengths, numbered major concerns,
numbered minor concerns, technical failings, originality/significance/
methodological-soundness/presentation/reproducibility criteria, required
evidence or claim narrowing, confidence, and recommendation posture. Each
begins with `Review setup` (input scope, assessment boundary, visible evidence,
and missing materials affecting confidence) and ends with a separate
`Risk / unsupported claims` section. Missing evidence is
`not-assessable`/`author-input-needed`, not hidden inside a confidence score.
The cross-review synthesis records consensus, weighting differences,
conflicting recommendations, and unsupported claims.

- [ ] **Step 1: Freeze the reviewed artifact**

Record PDF, source, results lock, and figure hashes. Reviewers assess the same
version.

- [ ] **Step 2: Produce exactly three simulated reports plus synthesis**

Ground every concern in the manuscript or artifact. Do not fabricate missing
experiments or call the report an editorial decision.

- [ ] **Step 3: Obtain one genuinely independent frozen-artifact review**

At least one reviewer who did not draft or edit the manuscript—an external
human or a genuinely separate review task/session given only the frozen
candidate and reviewer rubric—must assess the exact hashes. Internal subagents
sharing the authoring context count only as simulated QA. If an independent
review is unavailable, record it as a publication-readiness blocker rather than
mislabeling the three internal reports.

- [ ] **Step 4: Resolve concerns one by one**

For each issue record severity, evidence, action, changed files, validation,
and status. A requested new experiment returns to the locked experiment
workflow and creates a new results lock; it is never patched into prose.

- [ ] **Step 5: Rebuild, refreeze, and re-review**

Every change follows the dependency invalidation graph, regenerates applicable
outputs, rebuilds/renders/scans the PDFs, and produces a new candidate hash.
Re-rate each original concern against that new candidate and re-run all review
criteria affected by the change. Critical unresolved issues must be zero. Major
unresolved issues must be zero for the claimed scope. Honest limitations can
remain when the corresponding claim is narrowed and the limitation is explicit,
but narrowing itself must pass the new review.

Hard invariant:

```text
independent_review.candidate_sha256 == final_review_candidate.sha256
```

Any post-review change affecting claims, results, figures, methods, title, or
abstract requires the independent reviewer to assess the new hash or provide
explicit issue-by-issue sign-off on that exact candidate. Otherwise
`review_package_ready=false`. Re-run the occurrence-level citation verifier,
build/render/anonymity checks, and simulated criteria after every such change.
For prose changes, repeat Task 9's refresh workflow first: rebuild segment
hashes, recheck new/changed support, refresh occurrence/claim maps and
bibliography/verification reports, then run strict verification and commit all
updated evidence artifacts with the revision.

- [ ] **Step 6: Commit**

```powershell
git add paper/reviews paper/qa/submission-readiness.md
git commit -m "docs: resolve pre-submission review findings"
```

## Task 13: Package the Anonymous Review Artifact and Report Submission Blockers

**Files:**

- Create `scripts/publication/verify_publication_package.py`.
- Create `tests/publication/test_package_manifest.py`.
- Reuse `scripts/publication/scan_anonymity.py` and
  `tests/publication/test_anonymity.py` from Task 11.
- Generate `paper/dist/source-data.zip`.
- Generate `paper/dist/review-source.zip`.
- Generate `paper/dist/reproducibility-bundle.zip`.
- Finalize `paper/dist/reproducibility-manifest.json`.
- Generate `paper/dist/build-attestation.json`.
- Generate `paper/dist/anonymity-attestation.json`.

- [ ] **Step 1: Freeze the source commit**

First write RED package tests for manifest-DAG mismatch, missing required
archive roles, unsafe/colliding ZIP paths, symlink escape, undeclared files,
hash mismatch, identity leak, and candidate-review hash mismatch. Implement
`verify_publication_package.py`, run those tests plus the existing anonymity
tests, and commit the verifier/tests with every remaining source-owned
manuscript file, generator, config, schema, review resolution, and toolchain
lock:

```powershell
D:\conda\envs\spi\python.exe -m pytest tests/publication/test_package_manifest.py tests/publication/test_anonymity.py -q
git diff --check
git add scripts/publication/verify_publication_package.py tests/publication/test_package_manifest.py
git commit -m "feat: freeze publication package verifier"
```

Require a clean tree and record this exact HEAD as `source_commit`. Thus the
clean export contains the verifier and all tests used to validate itself.
`source_commit` is distinct from `experiment_code_commit` and the later
optional artifact commit.

- [ ] **Step 2: Build from a clean export and attest externally**

Use a temporary export of exact `source_commit`, verify the lock/toolchain/DAG,
then regenerate claims, tables, legends, figures, PDFs, and archives. Compare
semantic and promised byte hashes with working-tree outputs. Write
`build-attestation.json` with `source_commit`, results/protocol/toolchain hashes,
commands, output hashes, verifier versions, final candidate SHA, scanner source
SHA/version,
and the anonymity denylist/allowlist/scan-attestation hashes. It must not claim
the hash of a future commit containing itself.

- [ ] **Step 3: Keep the archives semantically separate**

- `source-data.zip`: only compact source data, data dictionary, schemas, lock
  references, and checksums needed to reproduce published plots/tables;
- `review-source.zip`: anonymous manuscript TeX, bibliography, figures,
  legends, tables, generation scripts, cards safe for review, tests, and build
  instructions;
- `reproducibility-bundle.zip`: redistributable code/config/protocol/schema
  sources and manifests, excluding any dependency/checkpoint/data whose license
  does not permit redistribution.

Use sorted POSIX relative paths, fixed ZIP timestamps/permissions/compression,
no duplicate or case-colliding entries, no absolute/drive/UNC paths, no `..`,
and no symlink/junction/hard-link escape. Test extraction into a fresh temporary
directory and reject archive bombs or path traversal. Exclude secrets, raw
checkpoints, raw datasets, caches, `_trash`, transient logs, and private
repository metadata.

- [ ] **Step 4: Run separate anonymity and security scans**

Require the same external denylist runtime input used in Task 11 and assert its
SHA-256 equals the pre-review anonymity attestation; likewise require the same
tracked allowlist and scanner-source SHA values. A missing or changed input
invalidates the reviewed candidate and returns to build → scan → refreeze →
independent review. The denylist contents never enter the repository or any ZIP.

Scan anonymous PDF Info/XMP, annotations, attachments, bookmarks, links, and
embedded file names. Scan TeX, BibTeX, SVG/XML metadata/text, JSON/YAML/CSV/MD,
scripts, cards, logs, manifests, and every uncompressed ZIP member for author
names, emails, ORCIDs, affiliations, acknowledgements, grant numbers,
usernames, Windows drive roots, UNC paths, private URLs/tokens, and repository
history clues. Use an explicit allowlist for scientific names/legitimate paths;
do not rely on deleting all metadata blindly. The full author package is
validated separately and must not be substituted for the anonymous one.
Generate `paper/dist/anonymity-attestation.json` with packaged candidate SHA
and full archive-member scope; do not overwrite the tracked pre-review
attestation. The package verifier and external build attestation require both
attestations, their candidate relationship, pass status, and exact
denylist/allowlist hashes. If the user declines the optional artifact commit,
the final attestation remains only in the external clean-build directory.

- [ ] **Step 5: Produce two readiness statuses**

- `review_package_ready`: code/evidence/prose/layout/references/reviews pass;
- `submission_package_ready`: additionally requires target venue, authors,
  affiliations, corresponding author, funding, conflicts, contributions,
  acknowledgements, data/code license, and repository/DOI decisions.

`review_package_ready` also requires the genuinely independent frozen-artifact
review from Task 12, a clean anonymity scan, and
`independent_review.candidate_sha256 == packaged_candidate.sha256`. The
anonymous review package may be complete while submission metadata remains
blocked. Report that distinction plainly.

- [ ] **Step 6: Verify and create the optional artifact commit**

```powershell
D:\conda\envs\spi\python.exe -m pytest tests/publication -q
D:\conda\envs\spi\python.exe scripts\publication\verify_publication_package.py --paper paper --strict
git diff --check
```

Ask the user whether generated PDFs/ZIPs/attestation should be committed. Only
after explicit approval run:

```powershell
git add paper/dist paper/qa/submission-readiness.md
git commit -m "release: package publication-ready review artifact"
```

If the user declines, keep deliverables in the verified external clean-build
directory and leave the repository clean; `artifact_commit` is absent. If
approved, the resulting commit is `artifact_commit`. A later annotated tag may bind
`artifact_commit` to the external attestation. This plan authorizes neither a
tag, push, GitHub release, journal upload, nor public redistribution; each
requires its own user-authorized action.

## Publication Completion Gate

Automated:

```powershell
$ErrorActionPreference = "Stop"
function Invoke-Checked {
    param([string]$Label, [scriptblock]$Command)
    & $Command
    $code = $LASTEXITCODE
    if ($code -ne 0) { throw "$Label failed with exit code $code." }
}

Invoke-Checked "CPU tests" { & 'D:\conda\envs\spi\python.exe' -m pytest -m "not cuda" -q }
Invoke-Checked "Publication tests" { & 'D:\conda\envs\spi\python.exe' -m pytest tests/publication -q }
Invoke-Checked "Results lock" { & 'D:\conda\envs\spi\python.exe' scripts\experiments\lock_results.py --verify-only }
Invoke-Checked "Prose lint" { & 'D:\conda\envs\spi\python.exe' scripts\publication\lint_manuscript.py --paper paper --strict }
Invoke-Checked "Citation verification" { & 'D:\conda\envs\spi\python.exe' scripts\publication\verify_citations.py --paper paper --occurrence-level --strict }
Invoke-Checked "Figure verification" { & 'D:\conda\envs\spi\python.exe' scripts\publication\validate_figures.py --paper paper --strict }
Invoke-Checked "Package verification" { & 'D:\conda\envs\spi\python.exe' scripts\publication\verify_publication_package.py --paper paper --strict }
Invoke-Checked "Whitespace check" { git diff --check }
$dirty = git status --porcelain=v1 --untracked-files=all
if ($LASTEXITCODE -ne 0 -or $dirty) { throw "Publication gate requires a clean worktree." }
```

Manual:

- [ ] The one-sentence argument is confirmed and bounded by locked evidence.
- [ ] The user explicitly approved the final one-sentence argument as an author.
- [ ] Every major claim has a supported claim-evidence entry.
- [ ] Every empirical result number is generated from the results lock; protocol
  and provenance numbers resolve to their separately hashed source bundle.
- [ ] All references are field-verified; no metadata-only citation remains.
- [ ] All six figures pass contract, source-data, statistics, and final-size QA.
- [ ] Figure legends are self-contained and name `n`, interval, metrics, and
      source data.
- [ ] Methods are reimplementable; exact configs/hashes are available.
- [ ] Results and Discussion are not mixed.
- [ ] OOD regressions and failure cases are explicit.
- [ ] No universal-optimality, universal-diffusion, or unsupported novelty claim
      remains.
- [ ] Prose passes the structural and low-AI audit without sounding mechanically
      homogenized.
- [ ] Every PDF page has been rendered and visually inspected.
- [ ] For venue-neutral `review_package_ready`, `target_venue: unresolved` and
  generic constraints are explicit. For `submission_package_ready` only, the
  named venue's current official author guide and disclosure rules were
  rechecked and recorded immediately before adaptation.
- [ ] Three simulated reviewer-emphasis reports, synthesis, and one genuinely
  independent frozen-artifact review are complete.
- [ ] The independent review's candidate SHA-256 equals the final packaged
  review candidate SHA-256.
- [ ] Critical and major review concerns are zero after rebuilding, refreezing,
  and re-rating the changed candidate.
- [ ] Anonymous review readiness and submission metadata readiness are reported
      separately.
- [ ] Anonymous PDF/source/archive scans find no non-allowlisted identity,
  private-path, credential, annotation, attachment, or metadata leak.
- [ ] `source_commit`, external build attestation, and optional
  `artifact_commit` are distinct and non-self-referential.
