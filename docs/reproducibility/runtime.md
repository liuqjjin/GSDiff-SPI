# Reproducible test runtime

This workstation uses one authoritative interpreter for dependency locking,
verification, and all reported tests:

```text
Interpreter: D:\conda\envs\spi\python.exe
Install: D:\conda\envs\spi\python.exe -m pip install -r requirements-dev.txt
CPU tests: D:\conda\envs\spi\python.exe -m pytest -m "not cuda" -q
CUDA tests: D:\conda\envs\spi\python.exe -m pytest -m cuda -q
```

The portable Python tests compare runtime metadata to `sys.executable`; the
absolute path above is workstation preflight evidence, not a repository-wide
interpreter assertion.

## Observed workstation runtime

Command:

```powershell
D:\conda\envs\spi\python.exe -c "import platform,torch; print(platform.python_version()); print(torch.__version__); print(torch.version.cuda); print(torch.cuda.get_device_name(0))"
```

Observed output on 2026-07-27:

```text
3.12.13
2.8.0+cu128
12.8
NVIDIA GeForce RTX 5060 Ti
```

The canonical environment lock additionally records NVIDIA driver `596.21`,
compute capability `12.0`, and 17,102,864,384 bytes of device memory. Its
fingerprint SHA-256 is
`b5d6922a9f3a9638ee8826b9a74f00998cd3ac81aa25c03de016358e0e435a56`.

## Dependency and environment locks

`requirements.txt` remains the human-maintained minimum runtime. The exact
installed distribution versions from the authoritative interpreter are sorted
in `requirements-lock.txt`. `requirements-dev.txt` adds the test and schema
validation pins. Publication experiments must match
`docs/reproducibility/environment-lock.json`, not only the minimum constraints.

The exact distribution lock was captured after the dev install using:

```powershell
D:\conda\envs\spi\python.exe -m pip list --format=freeze
```

The canonical environment fingerprint contains:

- normalized, sorted installed distribution name/version records;
- Python implementation, version, cache tag, ABI flags, and SOABI;
- operating-system, release, version, platform, and machine;
- PyTorch version, CUDA build, and cuDNN version;
- CUDA availability, driver, device identity, compute capability, and memory;
- only the numerical environment variables listed below.

The numerical environment-variable allowlist is:

```text
CUBLAS_WORKSPACE_CONFIG
CUDA_LAUNCH_BLOCKING
MKL_NUM_THREADS
NVIDIA_TF32_OVERRIDE
NUMEXPR_NUM_THREADS
OMP_NUM_THREADS
OPENBLAS_NUM_THREADS
PYTORCH_CUDA_ALLOC_CONF
TORCH_ALLOW_TF32_CUBLAS_OVERRIDE
VECLIB_MAXIMUM_THREADS
```

Every allowlisted key is recorded with its string value or `null`. No other
environment variable is inspected or serialized, so credentials, tokens,
usernames, paths, and unrelated host configuration are excluded.

Regenerate and verify the lock with:

```powershell
D:\conda\envs\spi\python.exe scripts\reproducibility\verify_environment_lock.py docs\reproducibility\environment-lock.json --write
D:\conda\envs\spi\python.exe scripts\reproducibility\verify_environment_lock.py --strict
```

Strict verification recomputes both the stored payload hash and the complete
current fingerprint. A self-consistent edited payload still fails when any
dependency, Python ABI field, numerical environment value, platform, PyTorch
build, CUDA driver, or GPU device differs.

## Canonical experiment identities

`build_run_identity()` accepts only
`execution_class="blind_method_child"`. Compatibility execution with
child-visible truth, a missing class, and unknown classes fail before identity
construction. Scientific-contract, method, target, motion, metric, asset, and
checkpoint names match ASCII `^[a-z0-9][a-z0-9_-]*$`. Identity-bearing SHA-256
values are exactly 64 lowercase hexadecimal characters, and the Git commit is
exactly 40. Seeds and dirty flags use exact integer and boolean types.

Resolved configurations are recursively copied into ordinary JSON mappings and
arrays before compact, sorted-key UTF-8 serialization. NaN and infinities are
rejected. The run identity stores only those immutable canonical bytes; their
SHA-256 is the full storage identity. `RunIdentity.payload()` decodes a new
recursively read-only view for each manifest consumer.
`canonical_json_bytes(identity.payload())` returns the exact stored canonical
bytes, including when the view contains nested read-only mapping proxies and
tuples. Direct dataclass construction revalidates the complete field set,
canonical bytes, blind execution class, IDs, hashes, dirty/source pairing,
payload hash, and display ID, so it cannot bypass the builder gate.

A clean identity requires `dirty_worktree=false` and a null source-tree hash.
A diagnostic dirty identity requires `dirty_worktree=true` and a valid
source-tree SHA-256. Publication campaigns reject dirty execution, and later
aggregation/promotion code must not treat diagnostic runs as locked evidence.
`git_state(repo, source_roots)` requires the same explicit source roots as
`source_tree_sha256()`. It reports the full commit, symbolic branch (or null
when detached), and evidence baseline
`c03420784bc92b4e9b9eef8330cbd9571ebebc68`. Its dirty flag preserves
whole-repository Git dirtiness for tracked and ordinary untracked files, and
additionally treats ignored untracked regular inputs inside the validated
source roots as dirty. Ignored files outside the roots and exact literal
exclusions inside them do not independently dirty the worktree.

### `source-tree-v1` framing

`source_tree_sha256()` requires one or more lexical source roots inside the
resolved Git worktree root. Those roots are the only allowlist for both tracked
and untracked paths. An ignored untracked file inside a root is included; the
implementation deliberately does not consult `.gitignore`. HEAD paths, index
paths, and scanned working files outside the roots do not enter the hash.
Existing Windows roots, including direct files, are reconciled to a unique Git
prefix only when `samefile` proves the filesystem object is identical; multiple
case-colliding Git candidates fail closed. A missing root that represents
staged deletion uses an exact Git prefix on all platforms. On Windows, a
pure-ASCII case-only alias is additionally accepted only when it maps to one
unique Git prefix. Missing non-ASCII lookalikes fail closed rather than using
Unicode lower- or case-folding. Thus `src` and fully deleted `SRC` share one
deletion identity, while `straße` cannot alias `strasse`, and a case-only
direct-file root cannot discard its staged index view.

Git subprocess output is captured as bytes. Commit and mode fields use strict
ASCII; Git paths and symbolic branch names use strict UTF-8. Worktree-root
validation requires raw `git rev-parse --is-inside-work-tree` to equal `true`
and the raw `--show-prefix` result to be empty. This accepts a linked-worktree
root while rejecting its `.git` metadata directory and bare repositories, and
does not round-trip the absolute root through the host locale. Chinese and
non-BMP repository roots and branch names therefore retain their exact values.

The hash input begins with `source-tree-v1`, a NUL byte, the 40-byte ASCII HEAD
commit, and a NUL byte. Records then follow in bytewise-sorted UTF-8
repo-relative path order. Each record contains:

1. an unsigned 64-bit big-endian path-byte length and the path bytes;
2. an index view with status `P` (present), `D` (HEAD path deleted from the
   index), or `U` (untracked), followed by mode `R`, `X`, or `-`, unsigned
   64-bit content length, and the 32 raw SHA-256 bytes; and
3. a working view with status `P` or `D` and the same mode, length, and digest
   framing.

Absent content uses mode `-`, zero length, and 32 zero bytes. Index content is
read directly from the staged Git blob, while working content is read as
binary bytes. Consequently staged content remains identity-bearing even if the
working file is restored to HEAD, staged deletion remains identity-bearing if
the working file is recreated, and unstaged edits remain independently
identity-bearing. Renames are the old deleted path plus the new added path;
there is no rename or text-diff heuristic. Index modes must be regular
`100644` or executable `100755`. POSIX working executable bits come from the
filesystem; Windows uses the Git index mode. Symlinks, junctions, other
reparse points, submodules, and other non-regular Git modes fail closed before
content traversal.

The exclusion policy is literal, not glob- or substring-based. Any exact path
component in this set excludes that subtree:

```text
.git
artifacts
results
_trash
__pycache__
.pytest_cache
.mypy_cache
.ruff_cache
.cache
.tox
.nox
.hypothesis
.ipynb_checkpoints
.venv
venv
env
ENV
__pypackages__
.pixi
```

Generated paper outputs are excluded only under the exact repo-root prefixes
`paper/figure_data`, `paper/figures`, `paper/tables`, `paper/build`, and
`paper/generated`. A source file such as `gsdiff/data/artifacts.py` is
therefore included.

## Test evidence gates

CPU numerical tests use the deterministic autouse seed fixture. CUDA tests
carry the `cuda` marker and skip only when `torch.cuda.is_available()` is
false. A valid CUDA campaign gate requires at least one executed test and zero
skips.

Fresh JUnit evidence is generated and checked with an explicit UTC boundary:

```powershell
D:\conda\envs\spi\python.exe -m pytest -q --junitxml=<report-path>
D:\conda\envs\spi\python.exe scripts\reproducibility\verify_pytest_junit.py <report-path> --created-after-utc <ISO-8601-UTC>
```

The verifier rejects missing or malformed XML, zero tests, failures, errors,
skips beyond the allowed maximum, inconsistent declared counts, and reports
older than the supplied boundary.
