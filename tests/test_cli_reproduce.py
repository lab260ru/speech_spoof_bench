"""Tests for the `reproduce` CLI subcommand."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from speech_spoof_bench.cli import main

FIX = Path(__file__).parent / "fixtures"


def test_inference_raises():
    p = FIX / "submissions" / "valid.yaml"
    with pytest.raises(NotImplementedError):
        main(["reproduce", "--inference", str(p)])


def test_scoring_invokes_run_scoring():
    p = FIX / "submissions" / "valid.yaml"
    with patch("speech_spoof_bench.reproduce.run_scoring", return_value=0) as r:
        rc = main(["reproduce", "--scoring", str(p), "--tolerance", "1e-3"])
    assert rc == 0
    r.assert_called_once()
    kwargs = r.call_args.kwargs
    assert kwargs["tolerance"] == 1e-3


def test_scoring_or_inference_required():
    p = FIX / "submissions" / "valid.yaml"
    with pytest.raises(SystemExit):
        main(["reproduce", str(p)])
