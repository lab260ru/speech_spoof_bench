"""load_meta validates meta.yaml against the submission_meta JSON Schema."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from speech_spoof_bench.submit import MetaValidationError, load_meta


_GOOD = {
    "system": {
        "name": "AASIST",
        "slug": "aasist-clovaai-default",
        "description": "Reference AASIST.\n",
        "code": "https://github.com/clovaai/aasist",
        "checkpoint": "https://huggingface.co/owner/repo",
        "paper": {
            "arxiv_id": "2110.01200",
            "url": "https://arxiv.org/abs/2110.01200",
            "bibtex": "@inproceedings{jung2022aasist}",
        },
    },
    "notes": "free-form notes",
}


def _write(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "meta.yaml"
    p.write_text(yaml.safe_dump(data))
    return p


def test_load_meta_accepts_complete(tmp_path: Path):
    out = load_meta(_write(tmp_path, _GOOD))
    assert out["system"]["slug"] == "aasist-clovaai-default"


def test_load_meta_accepts_no_notes(tmp_path: Path):
    data = dict(_GOOD)
    del data["notes"]
    out = load_meta(_write(tmp_path, data))
    assert "notes" not in out


def test_load_meta_rejects_missing_paper(tmp_path: Path):
    data = {"system": dict(_GOOD["system"])}
    del data["system"]["paper"]
    with pytest.raises(MetaValidationError):
        load_meta(_write(tmp_path, data))


def test_load_meta_rejects_bad_slug(tmp_path: Path):
    data = {"system": dict(_GOOD["system"]), "notes": ""}
    data["system"]["slug"] = "Has Spaces"
    with pytest.raises(MetaValidationError):
        load_meta(_write(tmp_path, data))


def test_load_meta_rejects_extra_top_level_key(tmp_path: Path):
    data = dict(_GOOD)
    data["unexpected"] = 1
    with pytest.raises(MetaValidationError):
        load_meta(_write(tmp_path, data))


def test_load_meta_file_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_meta(tmp_path / "nope.yaml")
