# Method registry v1

This note is the human-readable companion to
`configs/protocols/methods-v1.yaml`. The YAML registry remains authoritative.
The machine record at the end binds every raw publication and CPU-smoke
profile object by canonical-JSON SHA-256, so a changed value cannot pass the
documentation test unnoticed.

## Canonical methods and locked budgets

There are eleven canonical IDs. Compatibility aliases are not additional
methods.

| Canonical ID | Family | `publication-v1` native budget | `controller-cpu-smoke-v1` native budget |
| --- | --- | --- | --- |
| `dgi` | baseline | one direct-DGI pass | one direct-DGI pass |
| `static_cs` | baseline | ADMM 150; Chambolle 100; lambda `[0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]` | ADMM 1; Chambolle 100; lambda `[0.001]` |
| `perframe_cs` | baseline | ADMM 120; Chambolle 100; lambda `[0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]` | ADMM 1; Chambolle 100; lambda `[0.001]` |
| `tv3d` | baseline | primal-dual iterations 500; operator-norm iterations 30; lambda-xy `[0.003, 0.03, 0.3]`; lambda-t `[0.001, 0.01, 0.1, 1.0]` | iterations 1; operator-norm iterations 30; lambda-xy `[0.003]`; lambda-t `[0.001]` |
| `monin` | baseline | ADMM 150; Chambolle 100; lambda `[0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03]`; motion blocks 5 | ADMM 1; Chambolle 100; lambda `[0.0001]`; motion blocks 4 |
| `gidc3dtv` | baseline | Adam steps 2500; evaluation every 25; xi-xy `[0.003, 0.03, 0.3]`; xi-t `[0.01, 0.1]` | Adam step 1; evaluation every 1; xi-xy `[0.003]`; xi-t `[0.01]` |
| `recinr` | baseline | warm 300; flow 400; joint 1200 | warm 1; flow 1; joint 1 |
| `siren` | gsdiff | SGD steps 4000; motion warm-up 0 | SGD step 1; motion warm-up 0 |
| `recinr_se2` | gsdiff | SGD steps 3000; motion warm-up 500 | SGD step 1; motion warm-up 0 |
| `gsdiff_tv` | gsdiff | outer 80; inner 50; splitting warm-up 20; motion warm-up 16; proximal iterations 50 | outer 1; inner 1; both warm-ups 0; proximal iterations 50 |
| `gsdiff_diffusion` | gsdiff | outer 80; inner 50; splitting warm-up 20; motion warm-up 0; one denoise step | outer 1; inner 1; both warm-ups 0; one denoise step |

The table highlights each method's native budget and locked candidate grid.
Architecture, optimizer, regularization, eligibility, convergence, compute-cap,
and every remaining exact value are part of the raw profile objects bound in
the machine record. Publication profiles carry a 1,800-second wall-time cap
and a 15,032,385,536-byte peak-VRAM cap. Exceeding either cap retains
diagnostics but makes the run ineligible.

CPU smoke is a controller check, never publication evidence. Its method-config
ID is `smoke-default-v1`; it is always publication-, selection-, and
promotion-ineligible and records
`smoke-only/not-convergence-assessed`. A successful one-step run says that the
entry point and basic data path work. It says nothing about convergence or
scientific performance.

## Blind selection and algorithm randomness

The only child-side selection score is `heldout-normalized-l2-v1`:

```text
pred[k] = sum(P[k] * reconstruction[frame_indices[k]])
heldout_normalized_l2 =
    ||pred - y||_2 / max(||y||_2, 1e-12)
```

`P`, `pred`, and `y` are evaluated as float64 values in the raw physical
detector measurement units. The numerator and denominator therefore have the
same detector unit; their ratio is dimensionless. A solver may condition an
internal iterate, but it must return to physical forward-model scale before
selection or artifact writing. PSNR, SSIM, ground truth, trajectory error,
z-scored residuals, and evaluator data are not child selection signals.

Algorithm randomness has a separate `algorithm-seed-v1` domain:

```text
sha256(canonical_json({
  "domain": "algorithm-seed-v1",
  "cell_seed": <campaign cell seed>,
  "dataset_identity_sha256": <dataset identity>,
  "method_id": <canonical method ID>,
  "method_config_sha256": <path-free method hash>
}))
```

The first four digest bytes, read as an unsigned big-endian integer, form
`algorithm_seed_u32`. The full digest and integer are recorded. Python, NumPy,
and Torch use this seed; acquisition RNG streams 0--3 are not reused.

## Entry points, ownership, and boundary

Strict execution is the default. `scripts/run_baselines.py` accepts baseline
IDs and `train.py` accepts GSDiff IDs. A strict child consumes one materialized
request and produces exactly:

```text
reconstruction.npz
method-info.json
```

The parent validates those two files and owns the completed five-file
`outputs/` bundle:

```text
reconstruction.npz
metrics.json
method-info.json
stdout.log
stderr.log
```

Metrics and logs are parent-owned. Run manifests, resolved configuration,
audit evidence, and lifecycle state live outside `outputs/`. Evaluator truth
is loaded only after a successful child exit.

The audit bootstrap is a reproducible procedural boundary for trusted research
code, not an OS sandbox. It tests the Python-visible access surface and staged
closure. The claim does not cover native extensions or direct system calls;
both can bypass Python hooks and are outside this adversarial claim.

Old batch flags require the explicit `--legacy-compatibility` migration path.
That route is visibly labelled, emits legacy v1 output rather than the strict
two-file v2 contract, and is nonpromotable by the v2 parent validator. It
cannot silently enter a publication profile.

## Diffusion and pilot blockers

The frozen checkpoint requirement is:

- logical ID `gsdiff-diffusion-prior-v1`;
- SHA-256
  `667948800911acb9f9a7271e20af5692b0f007007d0fc32a15ac169eba32c5dd`;
- provenance status `blocked-missing-training-provenance`.

The publication profile remains non-executable and nonpromotable because
`missing-reproducible-checkpoint-locator` and
`missing-checkpoint-training-provenance` are unresolved. Knowing a desired
hash is not the same as having a reproducible locator or training record.

`configs/protocols/pilot-v1.yaml` therefore remains
`execution_ready: false`. The exact current readiness blockers are:

1. `all-eleven-method-clean-clone-subprocess-evidence-missing`;
2. `diffusion-reproducible-checkpoint-locator-missing`;
3. `diffusion-checkpoint-training-provenance-missing`;
4. `required-cuda-preflight-evidence-missing`;
5. `required-disk-preflight-evidence-missing`.

The DGI and GSDiff-TV CPU smoke proofs do not discharge these blockers. Pilot
readiness requires all eleven clean-clone method paths, required checkpoint
resolution, and the campaign's CUDA and disk preflights—not a local
checkpoint inference or two successful smoke cases.

## ReCINR provenance record

The audited upstream is <https://github.com/liuqjjin/ReCINR>, pinned at commit
`9149d1d228db2e4eb3ae852a004f1d9e95ee0229` and tree
`61df3a42e83f3145892ca8bba0aadfc88dc38c08`. The evidence below was obtained
read-only through GitHub's API; no excluded local ReCINR workspace was used.

| Pinned upstream path | SHA-256 |
| --- | --- |
| `README.md` | `ce118bfc1514ae9a15d688ab5c4333a73600b756370bd666593e140dffefdfa2` |
| `pyproject.toml` | `30c8d4e440f9e6ba18b345e035cef7b974ed858d3109364ab8eae99b235a8521` |
| `src/recinr/model.py` | `9c0d85bbc7e634038c9e060e4458f1df5d9d72f21069d5a924915b570a08660a` |
| `src/recinr/train.py` | `a9a066b454e6be1cbbf67cfe30b2311767560edd5e6d5c96ca842c0b838c399c` |

| Current tracked path | SHA-256 | Defensible mapping |
| --- | --- | --- |
| `gsdiff/baselines/recinr_model.py` | `7cfa1c7f20809634bc71fc14b143f81512358198af946f0039a38ad119c94eb7` | local header plus an earlier upstream model snapshot |
| `gsdiff/baselines/recinr.py` | `6bcb79509e1dcfe07e857aa5a5f92cb1cf32211c8017e4b0b692c80a64a6875a` | local semantic adapter; no one-to-one remote blob |
| `gsdiff/baselines/inr.py` | `600c26aec1545c0f7aab36d0cad5f95d4f4005ad0d3b9042e6c09f9ccdeabdf4` | earlier local control; not labelled as vendored ReCINR |

For the first mapping, removing local bytes 3--643 (a 641-byte header while
retaining the opening triple quote) produces 18,537 bytes with SHA-256
`bf80fc0b2573839ef500c45511e6692c5a669aef5fd4619974ddbb220b396047`.
That payload is byte-identical to upstream
`src/recinr/model.py` at ancestor commit
`847cca7cafded24ffc36522e92bc504090e48ab0`. The pinned commit is seven commits
ahead of that merge base. It is not byte-identical to the current local file:
the pinned model adds `strict_holdout`, `test_frac`, `val_frac`,
`forward(..., norm_mask=...)`, and train-mask-only normalization.

GitHub reports `licenseInfo=null` and `isArchived=false` for the repository.
The complete 285-entry pinned tree was not truncated and contains none of
`LICENSE`, `LICENSE.md`, `LICENSE.txt`, `COPYING`, or `NOTICE`. README and
pyproject MIT statements are declarations only, not a confirmed redistribution grant.
GitHub's repository-archive flag is distinct from this
project's redistribution decision: the project archive status is
`blocked-license-copyright-review` pending explicit license/copyright review.

## Machine-checked record

The following JSON is test data as well as documentation. Profile hashes use
canonical JSON with sorted keys, compact separators, UTF-8, and finite JSON
numbers.

<!-- method-registry-machine-v1:start -->
```json
{
  "registry": {
    "source": "configs/protocols/methods-v1.yaml",
    "source_sha256": "b262455370f0ff49f3f43dbf45c6f1871b5b6906b3b6560c13bc12ace7299b81",
    "canonical_ids": [
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
      "gsdiff_diffusion"
    ],
    "profile_hash_algorithm": "sha256(canonical-json(raw-profile-object))",
    "profile_sha256": {
      "dgi": {
        "publication-v1": "348efa2975e983845808ece2fdc354a7f88bb9e3a14a52e660252df142f3d973",
        "controller-cpu-smoke-v1": "ffbe93229f95744d5f29681b9c969df231950f5f1d3a931385723e2af3e5d816"
      },
      "static_cs": {
        "publication-v1": "c9486f89d5d6109c0b7b23f3de6b267171052329b5e62af88b6d96269099ad26",
        "controller-cpu-smoke-v1": "673394faf9fb6450ea546fdaa774612cef7eb1e8a123388aef053267bb155e0c"
      },
      "perframe_cs": {
        "publication-v1": "4b4ae41c407384e4e20bcc71db2cc2e28fe526208ba80eb39db91ffe8c6c3878",
        "controller-cpu-smoke-v1": "673394faf9fb6450ea546fdaa774612cef7eb1e8a123388aef053267bb155e0c"
      },
      "tv3d": {
        "publication-v1": "5fa29a994d7d7dbc4d502417e51f80f02231af7b6b9f5f94b7af0a1b69056297",
        "controller-cpu-smoke-v1": "e504f765caab65774f7c380f96549e84bcf2a05d03b8b909cd36d5303dfb0c37"
      },
      "monin": {
        "publication-v1": "e058d8ec005c9deea5947e9845ae4cff27a14bc032c63c1c7d6469b2d5f50e69",
        "controller-cpu-smoke-v1": "cd1ecc69f3fe682fcd923e432467a253f1585866600c785207f55db1b277e6fb"
      },
      "gidc3dtv": {
        "publication-v1": "b90eef64761c941edb22d8b691dcd436179961162aaf66e9aa57cad7ac4a7e69",
        "controller-cpu-smoke-v1": "95f2a6186e4cff4706eefdc581c141b80b0f9b85a3258fcb15d8b7fa22db0ecc"
      },
      "recinr": {
        "publication-v1": "62c7360bef2662be40b4625ea13512aedd3c3e24379703d56c57797cf4653afe",
        "controller-cpu-smoke-v1": "a07693cff64acfad92ed6863e48fa99175a0ba2f385016bad8f733cd10e29419"
      },
      "siren": {
        "publication-v1": "82f1884075e4a3e7e62b6316064dcd8fa79c0d9bbe3ab1b1b22fe901e04254c9",
        "controller-cpu-smoke-v1": "869cf756e4b8ac04cb6fdf55c32a9737dd2af1023cdf556edb852d2f82bfa648"
      },
      "recinr_se2": {
        "publication-v1": "c1c72fbeaf8f0c232777f2edec23c491e86b00e646de2c731217c6f7f48c9e06",
        "controller-cpu-smoke-v1": "d372d7926bee705836cd38cf29ee80d39c8d397541b4222a60de53ef8ab1045f"
      },
      "gsdiff_tv": {
        "publication-v1": "6fb974d71fc08efe62f0686c7a99ae9411c4a9d7cb49d20f61e0f80b2f98997b",
        "controller-cpu-smoke-v1": "20541c92836a9d56dac66ffa4648e8cf73953b8bc8349cc6a502346916251117"
      },
      "gsdiff_diffusion": {
        "publication-v1": "2d9a1204187dac604506ceb954b234e8b6c7b1bdf40ff0989cbb1ca26f7e641c",
        "controller-cpu-smoke-v1": "2180f3f406ca370ffa92e3b42e75eced092b2fdf1f81e08cc1ec52bd1815b335"
      }
    },
    "profiles": {
      "dgi": {"publication-v1":{"method_config_id":"default","publication_eligible":true,"selection_eligible":true,"promotion_eligible":true,"convergence_status":"not-applicable","execution_ready":true,"execution_blockers":[],"semantic_config":{"algorithm":"direct-dgi","native_unit":"pass","native_budget":1,"compute_cap":{"wall_time_seconds":1800,"peak_vram_bytes":15032385536,"on_exceed":"ineligible-retain-artifacts"}}},"controller-cpu-smoke-v1":{"method_config_id":"smoke-default-v1","publication_eligible":false,"selection_eligible":false,"promotion_eligible":false,"convergence_status":"smoke-only/not-convergence-assessed","execution_ready":true,"execution_blockers":[],"semantic_config":{"algorithm":"direct-dgi","native_unit":"pass","native_budget":1}}},
      "static_cs": {"publication-v1":{"method_config_id":"default","publication_eligible":true,"selection_eligible":true,"promotion_eligible":true,"convergence_status":"convergence-required","execution_ready":true,"execution_blockers":[],"semantic_config":{"solver":{"rho":0.5,"n_admm":150,"chambolle_iter":100,"lambda_grid":[0.001,0.003,0.01,0.03,0.1,0.3,1.0],"refit_all_measurements":true,"nonnegative":true,"target_std_conditioning":true,"chambolle_tv":"isotropic"},"compute_cap":{"wall_time_seconds":1800,"peak_vram_bytes":15032385536,"on_exceed":"ineligible-retain-artifacts"}}},"controller-cpu-smoke-v1":{"method_config_id":"smoke-default-v1","publication_eligible":false,"selection_eligible":false,"promotion_eligible":false,"convergence_status":"smoke-only/not-convergence-assessed","execution_ready":true,"execution_blockers":[],"semantic_config":{"solver":{"rho":0.5,"n_admm":1,"chambolle_iter":100,"lambda_grid":[0.001],"refit_all_measurements":true,"nonnegative":true,"target_std_conditioning":true,"chambolle_tv":"isotropic"}}}},
      "perframe_cs": {"publication-v1":{"method_config_id":"default","publication_eligible":true,"selection_eligible":true,"promotion_eligible":true,"convergence_status":"convergence-required","execution_ready":true,"execution_blockers":[],"semantic_config":{"solver":{"rho":0.5,"n_admm":120,"chambolle_iter":100,"lambda_grid":[0.001,0.003,0.01,0.03,0.1,0.3,1.0],"refit_all_measurements":true,"nonnegative":true,"target_std_conditioning":true,"chambolle_tv":"isotropic"},"compute_cap":{"wall_time_seconds":1800,"peak_vram_bytes":15032385536,"on_exceed":"ineligible-retain-artifacts"}}},"controller-cpu-smoke-v1":{"method_config_id":"smoke-default-v1","publication_eligible":false,"selection_eligible":false,"promotion_eligible":false,"convergence_status":"smoke-only/not-convergence-assessed","execution_ready":true,"execution_blockers":[],"semantic_config":{"solver":{"rho":0.5,"n_admm":1,"chambolle_iter":100,"lambda_grid":[0.001],"refit_all_measurements":true,"nonnegative":true,"target_std_conditioning":true,"chambolle_tv":"isotropic"}}}},
      "tv3d": {"publication-v1":{"method_config_id":"default","publication_eligible":true,"selection_eligible":true,"promotion_eligible":true,"convergence_status":"convergence-required","execution_ready":true,"execution_blockers":[],"semantic_config":{"solver":{"iterations":500,"opnorm_iterations":30,"lambda_xy":[0.003,0.03,0.3],"lambda_t":[0.001,0.01,0.1,1.0],"refit_all_measurements":true},"compute_cap":{"wall_time_seconds":1800,"peak_vram_bytes":15032385536,"on_exceed":"ineligible-retain-artifacts"}}},"controller-cpu-smoke-v1":{"method_config_id":"smoke-default-v1","publication_eligible":false,"selection_eligible":false,"promotion_eligible":false,"convergence_status":"smoke-only/not-convergence-assessed","execution_ready":true,"execution_blockers":[],"semantic_config":{"solver":{"iterations":1,"opnorm_iterations":30,"lambda_xy":[0.003],"lambda_t":[0.001],"refit_all_measurements":true}}}},
      "monin": {"publication-v1":{"method_config_id":"default","publication_eligible":true,"selection_eligible":true,"promotion_eligible":true,"convergence_status":"convergence-required","execution_ready":true,"execution_blockers":[],"semantic_config":{"solver":{"rho":1.0,"n_admm":150,"chambolle_iter":100,"lambda_grid":[0.0001,0.0003,0.001,0.003,0.01,0.03],"interpolation":"bilinear","motion_blocks":5,"polynomial_degree":1,"preview_blur_sigma":1.5,"ncc_search_radius":12},"compute_cap":{"wall_time_seconds":1800,"peak_vram_bytes":15032385536,"on_exceed":"ineligible-retain-artifacts"}}},"controller-cpu-smoke-v1":{"method_config_id":"smoke-default-v1","publication_eligible":false,"selection_eligible":false,"promotion_eligible":false,"convergence_status":"smoke-only/not-convergence-assessed","execution_ready":true,"execution_blockers":[],"semantic_config":{"solver":{"rho":1.0,"n_admm":1,"chambolle_iter":100,"lambda_grid":[0.0001],"interpolation":"bilinear","motion_blocks":4,"polynomial_degree":1,"preview_blur_sigma":1.5,"ncc_search_radius":12}}}},
      "gidc3dtv": {"publication-v1":{"method_config_id":"default","publication_eligible":true,"selection_eligible":true,"promotion_eligible":true,"convergence_status":"convergence-required","execution_ready":true,"execution_blockers":[],"semantic_config":{"solver":{"optimizer":"adam","n_steps":2500,"lr":0.05,"betas":[0.5,0.9],"eval_every":25,"xi_xy":[0.003,0.03,0.3],"xi_t":[0.01,0.1],"unet_channels":[16,32,64,128,128],"adam_epsilon":1e-08,"lr_decay":"0.90 ** (step/100)"},"compute_cap":{"wall_time_seconds":1800,"peak_vram_bytes":15032385536,"on_exceed":"ineligible-retain-artifacts"}}},"controller-cpu-smoke-v1":{"method_config_id":"smoke-default-v1","publication_eligible":false,"selection_eligible":false,"promotion_eligible":false,"convergence_status":"smoke-only/not-convergence-assessed","execution_ready":true,"execution_blockers":[],"semantic_config":{"solver":{"optimizer":"adam","n_steps":1,"lr":0.05,"betas":[0.5,0.9],"eval_every":1,"xi_xy":[0.003],"xi_t":[0.01],"unet_channels":[16,32,64,128,128],"adam_epsilon":1e-08,"lr_decay":"0.90 ** (step/100)"}}}},
      "recinr": {"publication-v1":{"method_config_id":"default","publication_eligible":true,"selection_eligible":true,"promotion_eligible":true,"convergence_status":"convergence-required","execution_ready":true,"execution_blockers":[],"semantic_config":{"representation":{"hidden_dim":32,"render_layers":3,"basis":"lowrank","basis_order":0,"harmonics":2,"flow_scale":0.5,"position_encoding_space":2,"position_encoding_time":5,"output_activation":"softplus"},"solver":{"anneal_fraction":0.6,"anchor_tau":0.5,"warm_steps":300,"flow_steps":400,"joint_steps":1200,"lr_start":0.003,"lr_end":0.001,"snapshot_every":50,"node_rule":"round-1.7T","lam_flow_t":0.5,"lam_flow_xy":0.2,"lam_l1":0.05,"tv_xy":1e-05,"lam_tv_canon":0.0001,"lam_ttv":0.0},"compute_cap":{"wall_time_seconds":1800,"peak_vram_bytes":15032385536,"on_exceed":"ineligible-retain-artifacts"}}},"controller-cpu-smoke-v1":{"method_config_id":"smoke-default-v1","publication_eligible":false,"selection_eligible":false,"promotion_eligible":false,"convergence_status":"smoke-only/not-convergence-assessed","execution_ready":true,"execution_blockers":[],"semantic_config":{"representation":{"hidden_dim":32,"render_layers":3,"basis":"lowrank","basis_order":0,"harmonics":2,"flow_scale":0.5,"position_encoding_space":2,"position_encoding_time":5,"output_activation":"softplus"},"solver":{"anneal_fraction":0.6,"anchor_tau":0.5,"warm_steps":1,"flow_steps":1,"joint_steps":1,"lr_start":0.003,"lr_end":0.001,"snapshot_every":50,"node_rule":"round-1.7T","lam_flow_t":0.5,"lam_flow_xy":0.2,"lam_l1":0.05,"tv_xy":1e-05,"lam_tv_canon":0.0001,"lam_ttv":0.0}}}},
      "siren": {"publication-v1":{"method_config_id":"default","publication_eligible":true,"selection_eligible":true,"promotion_eligible":true,"convergence_status":"convergence-required","execution_ready":true,"execution_blockers":[],"semantic_config":{"scene":{"scene_type":"siren","hidden":128,"hidden_layers":2,"w0":8,"initialization":"random","dgi_prefit":false},"motion":{"enable_rotation":true,"polynomial_degree":1,"enable_affine":false},"solver":{"solver_type":"sgd","sgd_steps":4000,"lr_scene":0.003,"motion_warmup_steps":0,"use_3dtv":true,"temporal_tv_weight":0.05,"loss_norm":"zscore","lr_motion":0.15,"tv_weight":0.005},"compute_cap":{"wall_time_seconds":1800,"peak_vram_bytes":15032385536,"on_exceed":"ineligible-retain-artifacts"}}},"controller-cpu-smoke-v1":{"method_config_id":"smoke-default-v1","publication_eligible":false,"selection_eligible":false,"promotion_eligible":false,"convergence_status":"smoke-only/not-convergence-assessed","execution_ready":true,"execution_blockers":[],"semantic_config":{"scene":{"scene_type":"siren","hidden":128,"hidden_layers":2,"w0":8,"initialization":"random","dgi_prefit":false},"motion":{"enable_rotation":true,"polynomial_degree":1,"enable_affine":false},"solver":{"solver_type":"sgd","sgd_steps":1,"lr_scene":0.003,"motion_warmup_steps":0,"use_3dtv":true,"temporal_tv_weight":0.05,"loss_norm":"zscore","lr_motion":0.15,"tv_weight":0.005}}}},
      "recinr_se2": {"publication-v1":{"method_config_id":"default","publication_eligible":true,"selection_eligible":true,"promotion_eligible":true,"convergence_status":"convergence-required","execution_ready":true,"execution_blockers":[],"semantic_config":{"scene":{"scene_type":"recinr","channels":32,"render_layers":3,"grid_size":20,"initialization":"random","dgi_prefit":false},"motion":{"enable_rotation":true,"polynomial_degree":1,"enable_affine":false},"solver":{"solver_type":"sgd","sgd_steps":3000,"lr_scene":0.003,"motion_warmup_steps":500,"use_3dtv":true,"temporal_tv_weight":0.05,"loss_norm":"zscore","lr_motion":0.15,"tv_weight":0.005},"compute_cap":{"wall_time_seconds":1800,"peak_vram_bytes":15032385536,"on_exceed":"ineligible-retain-artifacts"}}},"controller-cpu-smoke-v1":{"method_config_id":"smoke-default-v1","publication_eligible":false,"selection_eligible":false,"promotion_eligible":false,"convergence_status":"smoke-only/not-convergence-assessed","execution_ready":true,"execution_blockers":[],"semantic_config":{"scene":{"scene_type":"recinr","channels":32,"render_layers":3,"grid_size":20,"initialization":"random","dgi_prefit":false},"motion":{"enable_rotation":true,"polynomial_degree":1,"enable_affine":false},"solver":{"solver_type":"sgd","sgd_steps":1,"lr_scene":0.003,"motion_warmup_steps":0,"use_3dtv":true,"temporal_tv_weight":0.05,"loss_norm":"zscore","lr_motion":0.15,"tv_weight":0.005}}}},
      "gsdiff_tv": {"publication-v1":{"method_config_id":"default","publication_eligible":true,"selection_eligible":true,"promotion_eligible":true,"convergence_status":"convergence-required","execution_ready":true,"execution_blockers":[],"semantic_config":{"scene":{"scene_type":"gaussian","gaussian_count":1000,"initialization":"dgi_adaptive","init_scale":1.5,"min_scale":0.0},"motion":{"enable_rotation":true,"polynomial_degree":1,"enable_affine":false},"solver":{"solver_type":"admm","outer_iterations":80,"inner_iterations":50,"splitting_warmup_outer":20,"rho":0.1,"rho_growth":1.1,"prior_type":"tv","tv_variant":"tv3d_corrected","tv_weight":0.005,"soft_tv_weight":0.006,"temporal_tv_weight":0.1,"lr_scene":0.009,"lr_motion":0.15,"motion_warmup_fraction":0.2,"motion_warmup_outer":16,"prior_proximal_iterations":50,"loss_norm":"zscore"},"compute_cap":{"wall_time_seconds":1800,"peak_vram_bytes":15032385536,"on_exceed":"ineligible-retain-artifacts"}}},"controller-cpu-smoke-v1":{"method_config_id":"smoke-default-v1","publication_eligible":false,"selection_eligible":false,"promotion_eligible":false,"convergence_status":"smoke-only/not-convergence-assessed","execution_ready":true,"execution_blockers":[],"semantic_config":{"scene":{"scene_type":"gaussian","gaussian_count":1000,"initialization":"dgi_adaptive","init_scale":1.5,"min_scale":0.0},"motion":{"enable_rotation":true,"polynomial_degree":1,"enable_affine":false},"solver":{"solver_type":"admm","outer_iterations":1,"inner_iterations":1,"splitting_warmup_outer":0,"rho":0.1,"rho_growth":1.1,"prior_type":"tv","tv_variant":"tv3d_corrected","tv_weight":0.005,"soft_tv_weight":0.006,"temporal_tv_weight":0.1,"lr_scene":0.009,"lr_motion":0.15,"motion_warmup_fraction":0.0,"motion_warmup_outer":0,"prior_proximal_iterations":50,"loss_norm":"zscore"}}}},
      "gsdiff_diffusion": {"publication-v1":{"method_config_id":"default","publication_eligible":false,"selection_eligible":false,"promotion_eligible":false,"convergence_status":"convergence-required","execution_ready":false,"execution_blockers":["missing-reproducible-checkpoint-locator","missing-checkpoint-training-provenance"],"semantic_config":{"scene":{"scene_type":"gaussian","gaussian_count":1000,"initialization":"dgi_adaptive","init_scale":1.5,"min_scale":0.0},"motion":{"enable_rotation":true,"polynomial_degree":1,"enable_affine":false},"solver":{"solver_type":"admm","outer_iterations":80,"inner_iterations":50,"splitting_warmup_outer":20,"rho":0.1,"rho_growth":1.1,"prior_type":"diffusion","proximal_weight":0.005,"tv_weight":0.005,"soft_tv_weight":0.006,"temporal_tv_weight":0.05,"lr_scene":0.009,"lr_motion":0.15,"motion_warmup_fraction":0.0,"motion_warmup_outer":0,"loss_norm":"zscore"},"diffusion":{"denoise_steps":1,"clamp_range":[0.0,1.0],"in_channels":1,"base_channels":32,"channel_mults":[1,2,4],"emb_dim":128,"sigma_min":0.002,"sigma_max":0.5,"sigma_start":0.3,"sigma_end":0.05,"renoise":false,"ddim_spacing":"linear"},"compute_cap":{"wall_time_seconds":1800,"peak_vram_bytes":15032385536,"on_exceed":"ineligible-retain-artifacts"}}},"controller-cpu-smoke-v1":{"method_config_id":"smoke-default-v1","publication_eligible":false,"selection_eligible":false,"promotion_eligible":false,"convergence_status":"smoke-only/not-convergence-assessed","execution_ready":true,"execution_blockers":[],"semantic_config":{"scene":{"scene_type":"gaussian","gaussian_count":1000,"initialization":"dgi_adaptive","init_scale":1.5,"min_scale":0.0},"motion":{"enable_rotation":true,"polynomial_degree":1,"enable_affine":false},"solver":{"solver_type":"admm","outer_iterations":1,"inner_iterations":1,"splitting_warmup_outer":0,"rho":0.1,"rho_growth":1.1,"prior_type":"diffusion","proximal_weight":0.005,"tv_weight":0.005,"soft_tv_weight":0.006,"temporal_tv_weight":0.05,"lr_scene":0.009,"lr_motion":0.15,"motion_warmup_fraction":0.0,"motion_warmup_outer":0,"loss_norm":"zscore"},"diffusion":{"denoise_steps":1,"clamp_range":[0.0,1.0],"in_channels":1,"base_channels":32,"channel_mults":[1,2,4],"emb_dim":128,"sigma_min":0.002,"sigma_max":0.5,"sigma_start":0.3,"sigma_end":0.05,"renoise":false,"ddim_spacing":"linear"}}}}
    }
  },
  "blind_contract": {
    "selection_formula_id": "heldout-normalized-l2-v1",
    "selection_formula": "||pred-y||_2/max(||y||_2,1e-12); pred[k]=sum(P[k]*reconstruction[frame_indices[k]])",
    "selection_dtype": "float64",
    "selection_units": "raw physical detector measurement units before ratio",
    "algorithm_seed_domain": "algorithm-seed-v1",
    "child_output_files": [
      "reconstruction.npz",
      "method-info.json"
    ],
    "final_output_files": [
      "reconstruction.npz",
      "metrics.json",
      "method-info.json",
      "stdout.log",
      "stderr.log"
    ],
    "boundary": "procedural-boundary-for-trusted-research-code",
    "os_sandbox": false,
    "native_extensions_covered": false,
    "direct_syscalls_covered": false
  },
  "diffusion_checkpoint": {
    "logical_id": "gsdiff-diffusion-prior-v1",
    "sha256": "667948800911acb9f9a7271e20af5692b0f007007d0fc32a15ac169eba32c5dd",
    "provenance_status": "blocked-missing-training-provenance",
    "publication_execution_ready": false,
    "publication_blockers": [
      "missing-reproducible-checkpoint-locator",
      "missing-checkpoint-training-provenance"
    ]
  },
  "pilot_readiness": {
    "campaign": "configs/protocols/pilot-v1.yaml",
    "execution_ready": false,
    "blockers": [
      "all-eleven-method-clean-clone-subprocess-evidence-missing",
      "diffusion-reproducible-checkpoint-locator-missing",
      "diffusion-checkpoint-training-provenance-missing",
      "required-cuda-preflight-evidence-missing",
      "required-disk-preflight-evidence-missing"
    ]
  },
  "recinr_provenance": {
    "url": "https://github.com/liuqjjin/ReCINR",
    "pinned_commit": "9149d1d228db2e4eb3ae852a004f1d9e95ee0229",
    "pinned_tree": "61df3a42e83f3145892ca8bba0aadfc88dc38c08",
    "remote_sha256": {
      "README.md": "ce118bfc1514ae9a15d688ab5c4333a73600b756370bd666593e140dffefdfa2",
      "pyproject.toml": "30c8d4e440f9e6ba18b345e035cef7b974ed858d3109364ab8eae99b235a8521",
      "src/recinr/model.py": "9c0d85bbc7e634038c9e060e4458f1df5d9d72f21069d5a924915b570a08660a",
      "src/recinr/train.py": "a9a066b454e6be1cbbf67cfe30b2311767560edd5e6d5c96ca842c0b838c399c"
    },
    "local_sha256": {
      "gsdiff/baselines/recinr_model.py": "7cfa1c7f20809634bc71fc14b143f81512358198af946f0039a38ad119c94eb7",
      "gsdiff/baselines/recinr.py": "6bcb79509e1dcfe07e857aa5a5f92cb1cf32211c8017e4b0b692c80a64a6875a",
      "gsdiff/baselines/inr.py": "600c26aec1545c0f7aab36d0cad5f95d4f4005ad0d3b9042e6c09f9ccdeabdf4"
    },
    "source_mapping": {
      "gsdiff/baselines/recinr_model.py": {
        "relation": "earlier-upstream-snapshot-after-removing-641-byte-local-header-and-retaining-opening-triple-quote",
        "ancestor_commit": "847cca7cafded24ffc36522e92bc504090e48ab0",
        "ancestor_path": "src/recinr/model.py",
        "ancestor_sha256": "bf80fc0b2573839ef500c45511e6692c5a669aef5fd4619974ddbb220b396047",
        "ancestor_bytes": 18537,
        "pinned_commit_byte_identical": false
      },
      "gsdiff/baselines/recinr.py": {
        "relation": "local-adapter-no-one-to-one-remote-blob"
      },
      "gsdiff/baselines/inr.py": {
        "relation": "earlier-local-control-not-vendored"
      }
    },
    "github_license_info": null,
    "github_is_archived": false,
    "pinned_tree_entry_count": 285,
    "pinned_tree_truncated": false,
    "license_files_present": [],
    "license_file_names_checked": [
      "LICENSE",
      "LICENSE.md",
      "LICENSE.txt",
      "COPYING",
      "NOTICE"
    ],
    "repository_license_declarations": "README-and-pyproject-declarations-only",
    "confirmed_redistribution_grant": false,
    "archive_status": "blocked-license-copyright-review"
  }
}
```
<!-- method-registry-machine-v1:end -->
