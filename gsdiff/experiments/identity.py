from __future__ import annotations

import hashlib
from importlib import metadata as importlib_metadata
import json
import os
import platform
import re
import subprocess
import sys
import sysconfig
from typing import Any

import torch


NUMERICAL_ENV_ALLOWLIST = (
    "CUBLAS_WORKSPACE_CONFIG",
    "CUDA_LAUNCH_BLOCKING",
    "MKL_NUM_THREADS",
    "NVIDIA_TF32_OVERRIDE",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "PYTORCH_CUDA_ALLOC_CONF",
    "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
    "VECLIB_MAXIMUM_THREADS",
)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def collect_runtime_metadata() -> dict[str, object]:
    return {
        "python_executable": os.path.realpath(sys.executable),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "gpu_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "os": platform.platform(),
    }


def _normalize_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _installed_distributions() -> list[dict[str, str]]:
    records = []
    for distribution in importlib_metadata.distributions():
        name = distribution.metadata["Name"] or distribution.name
        records.append(
            {
                "name": _normalize_distribution_name(name),
                "version": str(distribution.version),
            }
        )
    return sorted(records, key=lambda item: (item["name"], item["version"]))


def _gpu_driver_version() -> str | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    versions = sorted({line.strip() for line in result.stdout.splitlines() if line.strip()})
    return ",".join(versions) if versions else None


def _gpu_fingerprint() -> dict[str, object]:
    available = torch.cuda.is_available()
    devices = []
    if available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "compute_capability": [
                        int(properties.major),
                        int(properties.minor),
                    ],
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "total_memory_bytes": int(properties.total_memory),
                }
            )
    return {
        "available": available,
        "device_count": len(devices),
        "devices": devices,
        "driver_version": _gpu_driver_version() if available else None,
    }


def collect_environment_fingerprint() -> dict[str, object]:
    """Return a deterministic, canonical, secret-free environment-lock payload."""
    implementation_version = sys.implementation.version
    return {
        "installed_distributions": _installed_distributions(),
        "numerical_environment": {
            name: os.environ.get(name) for name in NUMERICAL_ENV_ALLOWLIST
        },
        "platform": {
            "machine": platform.machine(),
            "platform": platform.platform(),
            "release": platform.release(),
            "system": platform.system(),
            "version": platform.version(),
        },
        "python": {
            "abi": {
                "abiflags": getattr(sys, "abiflags", ""),
                "cache_tag": sys.implementation.cache_tag,
                "soabi": sysconfig.get_config_var("SOABI"),
            },
            "implementation": platform.python_implementation(),
            "implementation_version": ".".join(
                str(part)
                for part in (
                    implementation_version.major,
                    implementation_version.minor,
                    implementation_version.micro,
                )
            ),
            "version": platform.python_version(),
        },
        "pytorch": {
            "cuda_build": torch.version.cuda,
            "cudnn_version": (
                torch.backends.cudnn.version()
                if torch.backends.cudnn.is_available()
                else None
            ),
            "version": str(torch.__version__),
        },
        "gpu": _gpu_fingerprint(),
    }
