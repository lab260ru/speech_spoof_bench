"""CLI integration tests for `validate-dataset`."""

from __future__ import annotations

from speech_spoof_bench.cli import main


def test_cli_validate_dataset_skip_submissions(synth_local_dataset, capsys):
    rc = main(["validate-dataset", str(synth_local_dataset), "--skip-submissions"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "OK" in out


def test_cli_validate_dataset_failure(synth_local_dataset, capsys):
    (synth_local_dataset / "README.md").write_text("no frontmatter")
    rc = main(["validate-dataset", str(synth_local_dataset), "--skip-submissions"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "D6" in out
    assert "failed" in out
