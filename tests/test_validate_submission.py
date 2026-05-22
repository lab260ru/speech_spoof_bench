"""Tests for `speech-spoof-bench validate-submission`."""

from __future__ import annotations

from pathlib import Path

import pytest

from speech_spoof_bench.cli import main

FIX = Path(__file__).parent / "fixtures" / "submissions"


def test_valid(capsys):
    rc = main(["validate-submission", str(FIX / "valid.yaml")])
    assert rc == 0
    assert "OK" in capsys.readouterr().out


@pytest.mark.parametrize(
    "name",
    [
        "invalid_no_reproduction.yaml",
        "invalid_unpinned_url.yaml",
        "invalid_bad_sha.yaml",
        "invalid_bad_slug.yaml",
        "invalid_wrong_schema_version.yaml",
        "invalid_malformed.yaml",
    ],
)
def test_invalid(name, capsys):
    rc = main(["validate-submission", str(FIX / name)])
    assert rc == 1
    assert "FAIL" in capsys.readouterr().err
