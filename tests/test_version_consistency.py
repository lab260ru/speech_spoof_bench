"""Guard: pyproject `version` and `__init__.__version__` must stay identical.

`green_store._BENCH_VERSION` derives from `__version__`, while the pip release
metadata derives from `pyproject.toml` — versioning.md requires them to match so
the two never drift. Regex-parsed (no tomllib) to run on Python 3.10.
"""
from __future__ import annotations

import re
from pathlib import Path

_PKG = Path(__file__).resolve().parent.parent


def _pyproject_version() -> str:
    text = (_PKG / "pyproject.toml").read_text()
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    assert m, "version not found in pyproject.toml"
    return m.group(1)


def _init_version() -> str:
    text = (_PKG / "src" / "speech_spoof_bench" / "__init__.py").read_text()
    m = re.search(r'__version__\s*=\s*"([^"]+)"', text)
    assert m, "__version__ not found in __init__.py"
    return m.group(1)


def test_pyproject_and_init_match():
    pv, iv = _pyproject_version(), _init_version()
    assert pv == iv, (pv, iv)
