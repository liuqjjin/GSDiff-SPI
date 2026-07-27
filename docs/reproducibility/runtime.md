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
