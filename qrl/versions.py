"""Resolved-dependency reporting.

The spec requires the resolved versions to be printed at the top of every run log,
so a reader of any results CSV / log can reconstruct the exact environment. Keep this
import-light: it must not fail if an optional package (e.g. qiskit-aer, only used for
the finite-shot hook) is missing.
"""
from __future__ import annotations

import platform
from importlib import metadata

# Packages whose versions materially affect results / reproducibility.
_TRACKED = [
    "qiskit",
    "qiskit-machine-learning",
    "qiskit-algorithms",
    "qiskit-aer",
    "torch",
    "gymnasium",
    "numpy",
    "scipy",
    "matplotlib",
    "pandas",
    "imageio",
    "pyyaml",
]

def resolved_versions() -> dict[str, str]:
    """Return {package: version} for tracked deps, plus python/platform info."""
    out: dict[str, str] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for name in _TRACKED:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "NOT INSTALLED"
    return out


def format_versions(header: str = "resolved environment") -> str:
    v = resolved_versions()
    width = max(len(k) for k in v)
    lines = [f"=== {header} ==="]
    lines += [f"  {k.ljust(width)} : {val}" for k, val in v.items()]
    return "\n".join(lines)


def print_versions(header: str = "resolved environment") -> None:
    print(format_versions(header), flush=True)


if __name__ == "__main__":
    print_versions()
