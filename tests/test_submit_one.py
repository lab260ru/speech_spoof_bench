"""submit_one wires hybrid-run + upload + PR with all HF calls mocked."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import yaml

import speech_spoof_bench.submit as submit_mod
from speech_spoof_bench.submit import submit_one


_META = {
    "system": {
        "name": "RB",
        "slug": "rb-phase7b",
        "description": "d",
        "code": "https://example.com/c",
        "checkpoint": "https://huggingface.co/Org/rb",
        "paper": {
            "arxiv_id": "1911.01601",
            "url": "https://arxiv.org/abs/1911.01601",
            "bibtex": "@x{}",
        },
    },
    "notes": "n",
}


def _result_yaml_text(revision: str) -> str:
    payload = {
        "schema_version": 4,
        "system": {"name": "unknown", "slug": None, "description": None,
                   "code": None, "checkpoint": None, "paper": None},
        "dataset": {"id": "Org/A", "revision": revision, "split": "test"},
        "scores": {"eer_percent": 50.0, "n_trials": 10, "n_skipped": 0},
        "artifact": {
            "scores_url": None,
            "scores_sha256": "f" * 64,
            "bench_version": "speech-spoof-bench==0.1.0",
        },
        "reproduction": {}, "submitter": {}, "submitted_at": None, "notes": None,
    }
    return yaml.safe_dump(payload, sort_keys=False)


@pytest.fixture
def fake_hf_api():
    api = MagicMock()
    upload_info = MagicMock()
    upload_info.oid = "a" * 40
    api.upload_file.return_value = upload_info
    commit_info = MagicMock()
    commit_info.pr_url = "https://huggingface.co/datasets/Org/A/discussions/1"
    api.create_commit.return_value = commit_info
    repo_info = MagicMock()
    repo_info.sha = "deadbee"
    api.repo_info.return_value = repo_info
    return api


def _make_existing_result(tmp_path: Path, slug: str, revision: str) -> Path:
    out = tmp_path / "results" / slug
    out.mkdir(parents=True, exist_ok=True)
    (out / "scores.txt").write_text("u 1.0\n")
    sha = hashlib.sha256(b"u 1.0\n").hexdigest()
    text = _result_yaml_text(revision).replace("f" * 64, sha)
    (out / "result.yaml").write_text(text)
    return out


def test_submit_one_reuses_existing_result_when_revision_matches(
    tmp_path: Path, fake_hf_api, monkeypatch
):
    """Hybrid path: result.yaml present + revision matches → no Benchmark.run call."""
    fake_hf_api.repo_info.return_value.sha = "deadbee"
    _make_existing_result(tmp_path, "Org_A", "deadbee")

    bench_called = {"n": 0}

    def fake_bench_run(*args, **kwargs):
        bench_called["n"] += 1

    monkeypatch.setattr(submit_mod, "_run_benchmark", fake_bench_run)
    monkeypatch.setattr(submit_mod, "_resolve_dataset_slug",
                        lambda spec, api, **_: ("Org/A", "Org_A", "deadbee", "test"))

    pr_url = submit_one(
        model_module_spec="x:Y",
        dataset_spec="Org/A",
        output_dir=tmp_path / "results",
        meta=_META,
        model_repo="Org/rb",
        hf_username="u",
        contact="c@example.com",
        submitted_at="2026-05-22",
        api=fake_hf_api,
    )

    assert pr_url.endswith("/discussions/1")
    assert bench_called["n"] == 0
    fake_hf_api.upload_file.assert_called_once()
    fake_hf_api.create_commit.assert_called_once()
    assert fake_hf_api.create_commit.call_args.kwargs["parent_commit"] == "deadbee"


def test_submit_one_runs_benchmark_when_result_missing(
    tmp_path: Path, fake_hf_api, monkeypatch
):
    fake_hf_api.repo_info.return_value.sha = "deadbee"

    def fake_bench_run(**_):
        # Materialize the same files Benchmark.run would have created:
        # output_dir/<slug>/scores.txt + result.yaml. _make_existing_result
        # adds a leading "results/" so pass tmp_path (parent of output_dir).
        _make_existing_result(tmp_path, "Org_A", "deadbee")

    monkeypatch.setattr(submit_mod, "_run_benchmark", fake_bench_run)
    monkeypatch.setattr(submit_mod, "_resolve_dataset_slug",
                        lambda spec, api, **_: ("Org/A", "Org_A", "deadbee", "test"))

    pr_url = submit_one(
        model_module_spec="x:Y",
        dataset_spec="Org/A",
        output_dir=tmp_path / "results",
        meta=_META,
        model_repo="Org/rb",
        hf_username="u", contact="c",
        submitted_at="2026-05-22",
        api=fake_hf_api,
    )
    assert pr_url.endswith("/discussions/1")


def test_submit_one_reruns_when_revision_mismatch(
    tmp_path: Path, fake_hf_api, monkeypatch
):
    fake_hf_api.repo_info.return_value.sha = "newrev1"
    _make_existing_result(tmp_path, "Org_A", "oldrev1")

    bench_called = {"n": 0}

    def fake_bench_run(**_):
        bench_called["n"] += 1
        _make_existing_result(tmp_path, "Org_A", "newrev1")

    monkeypatch.setattr(submit_mod, "_run_benchmark", fake_bench_run)
    monkeypatch.setattr(submit_mod, "_resolve_dataset_slug",
                        lambda spec, api, **_: ("Org/A", "Org_A", "newrev1", "test"))

    submit_one(
        model_module_spec="x:Y",
        dataset_spec="Org/A",
        output_dir=tmp_path / "results",
        meta=_META,
        model_repo="Org/rb",
        hf_username="u", contact="c",
        submitted_at="2026-05-22",
        api=fake_hf_api,
    )
    assert bench_called["n"] == 1


def test_submit_one_yaml_is_schema_valid(
    tmp_path: Path, fake_hf_api, monkeypatch
):
    """The YAML pushed in the PR commit must parse as a valid v4 submission."""
    from speech_spoof_bench import submission

    fake_hf_api.repo_info.return_value.sha = "deadbee"
    _make_existing_result(tmp_path, "Org_A", "deadbee")
    monkeypatch.setattr(submit_mod, "_resolve_dataset_slug",
                        lambda spec, api, **_: ("Org/A", "Org_A", "deadbee", "test"))
    monkeypatch.setattr(submit_mod, "_run_benchmark", lambda **_: None)

    submit_one(
        model_module_spec="x:Y",
        dataset_spec="Org/A",
        output_dir=tmp_path / "results",
        meta=_META,
        model_repo="Org/rb",
        hf_username="u", contact="c@example.com",
        submitted_at="2026-05-22",
        api=fake_hf_api,
    )

    ops = fake_hf_api.create_commit.call_args.kwargs["operations"]
    op = ops[0]
    yaml_text = op.path_or_fileobj.getvalue().decode("utf-8")
    parsed = submission.parse_submission(yaml_text)
    assert parsed["reproduction"] == {}
    assert parsed["system"]["slug"] == "rb-phase7b"
    assert parsed["submitter"] == {"hf_username": "u", "contact": "c@example.com"}
    assert parsed["dataset"]["revision"] == "deadbee"


def test_resolve_dataset_slug_hf_uses_eval_yaml_without_loading_dataset(
    tmp_path: Path, fake_hf_api, monkeypatch
):
    eval_yaml = tmp_path / "eval.yaml"
    eval_yaml.write_text(
        "name: Tiny\n"
        "tasks:\n"
        "  - id: antispoofing_eval\n"
        "    split: test\n"
        "    metrics: [eer_percent]\n"
    )
    fake_hf_api.repo_info.return_value.sha = "abcdef1234"

    monkeypatch.setattr(
        submit_mod,
        "hf_hub_download",
        lambda **kw: str(eval_yaml),
        raising=False,
    )

    def boom(*args, **kwargs):
        raise AssertionError("submit metadata resolution must not load the dataset")

    monkeypatch.setattr("speech_spoof_bench.loader.resolve", boom)

    assert submit_mod._resolve_dataset_slug(
        "Org/Tiny", fake_hf_api, force_remote=True
    ) == ("Org/Tiny", "Tiny", "abcdef1234", "test")


def test_resolve_dataset_slug_existing_local_path_uses_loader(
    tmp_path: Path, fake_hf_api, monkeypatch
):
    local_ds = tmp_path / "data" / "tiny"
    local_ds.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    fake_hf_api.repo_info.return_value.sha = "localsha"

    monkeypatch.setattr(
        submit_mod,
        "hf_hub_download",
        lambda **kw: (_ for _ in ()).throw(
            AssertionError("existing local dataset paths must not hit the Hub")
        ),
        raising=False,
    )

    source = SimpleNamespace(
        canonical_id="Local/Tiny",
        slug="Tiny",
        split="test",
    )
    monkeypatch.setattr(
        "speech_spoof_bench.loader.resolve",
        lambda spec, **kw: (source, None),
    )

    assert submit_mod._resolve_dataset_slug(
        "data/tiny", fake_hf_api, force_remote=False
    ) == ("Local/Tiny", "Tiny", "localsha", "test")
