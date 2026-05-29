"""Validates the optional `system.params_millions` field across the submission schemas."""

import json
from importlib import resources

import pytest
from jsonschema import ValidationError, validate


def _meta_schema():
    with resources.files("speech_spoof_bench.data").joinpath("submission_meta.schema.json").open() as f:
        return json.load(f)


_BASE_SYSTEM = {
    "name": "AASIST", "slug": "aasist", "description": "x",
    "code": "https://github.com/x/y", "checkpoint": "https://huggingface.co/x/y",
    "paper": {"arxiv_id": "1", "url": "https://arxiv.org/abs/1", "bibtex": "@x{y}"},
}


def test_meta_accepts_params_millions():
    inst = {"system": {**_BASE_SYSTEM, "params_millions": 52.3}}
    validate(inst, _meta_schema())  # must not raise


def test_meta_valid_without_params_millions():
    inst = {"system": dict(_BASE_SYSTEM)}
    validate(inst, _meta_schema())  # optional → still valid


def test_meta_rejects_negative_params():
    inst = {"system": {**_BASE_SYSTEM, "params_millions": -1}}
    with pytest.raises(ValidationError):
        validate(inst, _meta_schema())
