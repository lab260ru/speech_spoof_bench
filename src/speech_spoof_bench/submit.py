"""Phase 7b — `submit` command implementation.

Public surface (used by cli.py):
  - load_meta(path) -> dict
  - submit(...)             # added later in Task 8
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any

import yaml
from jsonschema import ValidationError, validate

_META_SCHEMA_PACKAGE = "speech_spoof_bench.data"
_META_SCHEMA_FILENAME = "submission_meta.schema.json"


class MetaValidationError(ValueError):
    """Raised when a submission meta YAML fails schema validation."""


def _load_meta_schema() -> dict[str, Any]:
    with resources.files(_META_SCHEMA_PACKAGE).joinpath(_META_SCHEMA_FILENAME).open("r") as f:
        return json.load(f)


def load_meta(path: Path | str) -> dict[str, Any]:
    """Parse and validate a submission meta YAML.

    Raises:
      FileNotFoundError: path doesn't exist.
      MetaValidationError: YAML parses but fails the schema.
    """
    p = Path(path)
    text = p.read_text()  # raises FileNotFoundError as desired
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise MetaValidationError(f"{p}: not a YAML mapping")
    try:
        validate(instance=data, schema=_load_meta_schema())
    except ValidationError as exc:
        raise MetaValidationError(f"{p}: {exc.message}") from exc
    return data
