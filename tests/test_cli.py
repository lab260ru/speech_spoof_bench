"""Tests for the CLI entry point."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

import speech_spoof_bench.metrics.eer  # noqa: F401
from speech_spoof_bench.cli import main


def test_cli_help_runs(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


def test_cli_run_against_local(synth_local_dataset: Path, tmp_path: Path):
    out = tmp_path / "results"
    rc = main(
        [
            "run",
            "--model-module",
            "speech_spoof_bench.examples.random_baseline:RandomBaseline",
            "--datasets",
            str(synth_local_dataset),
            "--output-dir",
            str(out),
            "--no-cleanup",
            "--no-skip-existing",
        ]
    )
    assert rc == 0
    result_yaml = out / "SynthDataset_TEST" / "result.yaml"
    assert result_yaml.exists()


def test_cli_validate_dataset_local(synth_local_dataset: Path):
    rc = main(["validate-dataset", str(synth_local_dataset), "--skip-submissions"])
    assert rc == 0


from speech_spoof_bench import manifest as _mf


_FAKE_MANIFEST = {
    "ranking_version": "v1",
    "schema_version": 1,
    "metrics_in_use": ["eer_percent"],
    "tiers": [
        {"name": "gold", "min_coverage": 1.0},
        {"name": "silver", "min_coverage": 0.5},
        {"name": "bronze", "min_coverage": 0.0},
    ],
    "core_set": [
        {"id": "Org/A", "revision": "9b2040e8c57749dcd9a4f16ad61b4f47626b89ec"}
    ],
    "extended": [
        {"id": "Org/B", "revision": "deadbeef"}
    ],
}


def test_cli_manifest_prints_yaml(monkeypatch, capsys, tmp_path):
    """`manifest` prints the raw file contents verbatim."""
    raw = (
        "ranking_version: v1\n"
        "schema_version: 1\n"
        "metrics_in_use:\n  - eer_percent\n"
        "tiers:\n  - {name: gold, min_coverage: 1.0}\n"
        "  - {name: silver, min_coverage: 0.5}\n"
        "  - {name: bronze, min_coverage: 0.0}\n"
        "core_set:\n  - id: Org/A\n    revision: 9b2040e8c57749dcd9a4f16ad61b4f47626b89ec\n"
        "extended: []\n"
    )
    fake = tmp_path / "manifest.yaml"
    fake.write_text(raw)

    def fake_download(**kwargs):
        return str(fake)

    monkeypatch.setattr(_mf, "hf_hub_download", fake_download)
    rc = main(["manifest"])
    assert rc == 0
    out = capsys.readouterr().out
    assert out.rstrip("\n") == raw.rstrip("\n")


def test_cli_list_prints_core_then_extended(monkeypatch, capsys):
    def fake_fetch():
        return _FAKE_MANIFEST

    monkeypatch.setattr(_mf, "fetch_manifest", fake_fetch)
    rc = main(["list"])
    assert rc == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines == ["[core] Org/A", "[ext]  Org/B"]


def test_cli_submit_smoke(monkeypatch, tmp_path):
    """`submit` wires flags into submit_one for each dataset."""
    import speech_spoof_bench.submit as submit_mod

    meta_path = tmp_path / "meta.yaml"
    meta_path.write_text(
        "system:\n"
        "  name: RB\n"
        "  slug: rb\n"
        "  description: d\n"
        "  code: https://example.com/c\n"
        "  checkpoint: https://huggingface.co/Org/rb\n"
        "  paper:\n"
        "    arxiv_id: '1911.01601'\n"
        "    url: https://arxiv.org/abs/1911.01601\n"
        "    bibtex: '@x{}'\n"
    )

    calls = []

    def fake_submit_one(**kwargs):
        calls.append(kwargs["dataset_spec"])
        return f"https://huggingface.co/datasets/{kwargs['dataset_spec']}/discussions/1"

    monkeypatch.setattr(submit_mod, "submit_one", fake_submit_one)

    rc = main([
        "submit",
        "--model-module", "x:Y",
        "--datasets", "Org/A",
        "--datasets", "Org/B",
        "--model-repo", "Org/rb",
        "--submission-meta", str(meta_path),
        "--hf-username", "u",
        "--contact", "c@example.com",
        "--output-dir", str(tmp_path / "results"),
    ])
    assert rc == 0
    assert calls == ["Org/A", "Org/B"]


def test_cli_submit_all_uses_manifest(monkeypatch, tmp_path):
    """`--datasets all` iterates core_set + extended from the manifest."""
    import speech_spoof_bench.submit as submit_mod
    from speech_spoof_bench import manifest as _mf

    meta_path = tmp_path / "meta.yaml"
    meta_path.write_text(
        "system:\n"
        "  name: RB\n  slug: rb\n  description: d\n"
        "  code: https://example.com/c\n"
        "  checkpoint: https://huggingface.co/Org/rb\n"
        "  paper:\n    arxiv_id: '1'\n    url: https://arxiv.org/abs/1\n    bibtex: '@x{}'\n"
    )

    monkeypatch.setattr(_mf, "fetch_manifest", lambda: _FAKE_MANIFEST)

    seen = []
    monkeypatch.setattr(submit_mod, "submit_one", lambda **kw: seen.append(kw["dataset_spec"]) or "url")

    rc = main([
        "submit",
        "--model-module", "x:Y",
        "--datasets", "all",
        "--model-repo", "Org/rb",
        "--submission-meta", str(meta_path),
        "--hf-username", "u", "--contact", "c@example.com",
        "--output-dir", str(tmp_path / "results"),
    ])
    assert rc == 0
    assert seen == ["Org/A", "Org/B"]


def test_cli_scaffold_dataset(tmp_path):
    out = tmp_path / "Y"
    rc = main(["scaffold-dataset", "--name", "Y", "--output-dir", str(out)])
    assert rc == 0
    assert (out / "eval.yaml").is_file()
    assert "name: Y" in (out / "eval.yaml").read_text()


def test_local_set_and_list(monkeypatch, tmp_path, capsys):
    from speech_spoof_bench import cli, local_registry as lr
    monkeypatch.setattr(lr, "_registry_path", lambda: tmp_path / "reg.yaml")
    d = tmp_path / "LA"
    (d / "data").mkdir(parents=True)
    (d / "data" / "test-00000-of-00001.parquet").write_bytes(b"")
    (d / "eval.yaml").write_text("name: x\n")

    assert cli.main(["local", "set", "Org/Foo", str(d)]) == 0
    assert cli.main(["local", "list"]) == 0
    out = capsys.readouterr().out
    assert "Org/Foo" in out and str(d) in out


def test_local_unset(monkeypatch, tmp_path):
    from speech_spoof_bench import cli, local_registry as lr
    monkeypatch.setattr(lr, "_registry_path", lambda: tmp_path / "reg.yaml")
    d = tmp_path / "LA"
    (d / "data").mkdir(parents=True)
    (d / "data" / "test-00000-of-00001.parquet").write_bytes(b"")
    (d / "eval.yaml").write_text("name: x\n")
    lr.set("Org/Foo", d)
    assert cli.main(["local", "unset", "Org/Foo"]) == 0
    assert lr.load() == {}


def test_local_show_mapped_and_unmapped(monkeypatch, tmp_path, capsys):
    from speech_spoof_bench import cli, local_registry as lr
    monkeypatch.setattr(lr, "_registry_path", lambda: tmp_path / "reg.yaml")
    d = tmp_path / "LA"
    (d / "data").mkdir(parents=True)
    (d / "data" / "test-00000-of-00001.parquet").write_bytes(b"")
    (d / "eval.yaml").write_text("name: x\n")

    assert cli.main(["local", "show", "Org/Foo"]) == 0
    assert "remote" in capsys.readouterr().out

    lr.set("Org/Foo", d)
    assert cli.main(["local", "show", "Org/Foo"]) == 0
    out = capsys.readouterr().out
    assert "local" in out and str(d) in out


def test_run_no_local_bypasses_registry(monkeypatch, tmp_path):
    """When --no-local is set, runner must not see the registered local path."""
    from speech_spoof_bench import benchmark as bm, cli, local_registry as lr

    monkeypatch.setattr(lr, "_registry_path", lambda: tmp_path / "reg.yaml")
    # Pretend Org/Foo is registered locally.
    d = tmp_path / "LA"
    (d / "data").mkdir(parents=True)
    (d / "data" / "test-00000-of-00001.parquet").write_bytes(b"")
    (d / "eval.yaml").write_text("name: x\n")
    lr.set("Org/Foo", d)

    seen = {}
    def fake_resolve(spec, *, streaming=True, force_remote=False):
        seen["spec"] = spec
        seen["force_remote"] = force_remote
        # short-circuit so we don't actually load anything
        raise SystemExit(0)
    # benchmark.py uses `from .loader import resolve`, so patch there directly
    monkeypatch.setattr(bm, "resolve", fake_resolve)

    with pytest.raises(SystemExit):
        cli.main([
            "run",
            "--model-module", "speech_spoof_bench.examples.random_baseline:RandomBaseline",
            "--datasets", "Org/Foo",
            "--no-local",
        ])
    assert seen["force_remote"] is True


def test_ci_verify_pr_dispatches(monkeypatch):
    from speech_spoof_bench import cli
    from speech_spoof_bench.ci import verify_pr
    captured = {}
    def fake_run(*, repo, pr, branch, **_kw):
        captured["args"] = (repo, pr, branch)
        return 0
    monkeypatch.setattr(verify_pr, "run", fake_run)
    rc = cli.main(["ci", "verify-pr", "--repo", "Org/Foo", "--pr", "7", "--branch", "refs/pr/7"])
    assert rc == 0
    assert captured["args"] == ("Org/Foo", 7, "refs/pr/7")


def test_ci_preview_manifest_dispatches(monkeypatch, tmp_path):
    from speech_spoof_bench import cli
    from speech_spoof_bench.ci import preview_manifest

    captured = {}

    def fake_run(*, manifest_path=None, repo=None, pr=None, branch=None):
        captured["args"] = (manifest_path, repo, pr, branch)
        return 0

    monkeypatch.setattr(preview_manifest, "run", fake_run)
    local_manifest = tmp_path / "manifest.yaml"
    local_manifest.write_text("schema_version: 1\n")

    rc = cli.main(["ci", "preview-manifest", "--manifest", str(local_manifest)])

    assert rc == 0
    assert captured["args"] == (local_manifest, None, None, None)
