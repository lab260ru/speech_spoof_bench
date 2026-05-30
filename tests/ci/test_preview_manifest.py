"""Tests for arena-manifest preview CI."""

from __future__ import annotations

from pathlib import Path

from speech_spoof_bench.ci import preview_manifest


def _manifest(*dataset_ids: str) -> dict:
    return {
        "schema_version": 1,
        "ranking_version": "test",
        "metrics_in_use": ["eer_percent"],
        "tiers": [{"name": "gold", "min_coverage": 1.0}],
        "core_set": [{"id": dataset_ids[0], "revision": "abc1234"}],
        "extended": [
            {"id": dataset_id, "revision": "def5678"}
            for dataset_id in dataset_ids[1:]
        ],
    }


def _submission(slug: str = "sys", *, reproduced: bool = True) -> dict:
    return {
        "system": {"slug": slug, "name": slug},
        "scores": {"eer_percent": 1.0},
        "reproduction": {"match": "scoring"} if reproduced else {},
    }


def test_preview_counts_rows_warnings_and_dataset_delta(monkeypatch):
    candidate = _manifest("Org/A", "Org/B")
    base = _manifest("Org/A")

    monkeypatch.setattr(
        preview_manifest.submission,
        "list_submission_files",
        lambda dataset_id: [f"submissions/{dataset_id.split('/')[-1]}.yaml"],
    )
    monkeypatch.setattr(
        preview_manifest.submission,
        "fetch_submission",
        lambda dataset_id, path: _submission(slug=dataset_id.split("/")[-1]),
    )
    monkeypatch.setattr(
        preview_manifest.hf_fetch,
        "list_repo_files",
        lambda dataset_id, repo_type, revision: ["eval.yaml"],
    )

    result = preview_manifest.preview(candidate, base_manifest=base)

    assert result.rows == 2
    assert result.warnings == []
    assert result.added_datasets == ["Org/B"]
    assert result.removed_datasets == []
    assert "Rows: 2" in preview_manifest.format_markdown(result)
    assert "Org/B" in preview_manifest.format_markdown(result)


def test_preview_warns_and_skips_submission_without_reproduction(monkeypatch):
    candidate = _manifest("Org/A")

    monkeypatch.setattr(
        preview_manifest.submission,
        "list_submission_files",
        lambda dataset_id: ["submissions/no-repro.yaml"],
    )
    monkeypatch.setattr(
        preview_manifest.submission,
        "fetch_submission",
        lambda dataset_id, path: _submission(reproduced=False),
    )
    monkeypatch.setattr(
        preview_manifest.hf_fetch,
        "list_repo_files",
        lambda dataset_id, repo_type, revision: ["eval.yaml"],
    )

    result = preview_manifest.preview(candidate)

    assert result.rows == 0
    assert len(result.warnings) == 1
    assert result.warnings[0].dataset_id == "Org/A"
    assert "missing reproduction block" in result.warnings[0].reason


def test_run_downloads_manifest_branch_and_posts_comment(monkeypatch, tmp_path, capsys):
    candidate_path = tmp_path / "candidate.yaml"
    base_path = tmp_path / "base.yaml"
    candidate_path.write_text(
        "schema_version: 1\n"
        "ranking_version: test\n"
        "metrics_in_use: [eer_percent]\n"
        "tiers:\n"
        "  - {name: gold, min_coverage: 1.0}\n"
        "core_set:\n"
        "  - {id: Org/A, revision: abc1234}\n"
        "extended:\n"
        "  - {id: Org/B, revision: def5678}\n"
    )
    base_path.write_text(
        "schema_version: 1\n"
        "ranking_version: test\n"
        "metrics_in_use: [eer_percent]\n"
        "tiers:\n"
        "  - {name: gold, min_coverage: 1.0}\n"
        "core_set:\n"
        "  - {id: Org/A, revision: abc1234}\n"
        "extended: []\n"
    )

    def fake_download(repo_id, filename, repo_type, revision=None):
        assert filename == "manifest.yaml"
        return candidate_path if revision == "refs/pr/7" else base_path

    monkeypatch.setattr(preview_manifest.hf_fetch, "hub_download", fake_download)
    monkeypatch.setattr(
        preview_manifest.submission,
        "list_submission_files",
        lambda dataset_id: ["submissions/a.yaml"],
    )
    monkeypatch.setattr(
        preview_manifest.submission,
        "fetch_submission",
        lambda dataset_id, path: _submission(slug=dataset_id.split("/")[-1]),
    )
    monkeypatch.setattr(
        preview_manifest.hf_fetch,
        "list_repo_files",
        lambda dataset_id, repo_type, revision: ["eval.yaml"],
    )
    monkeypatch.delenv("HF_BOT_TOKEN", raising=False)

    rc = preview_manifest.run(
        repo="SpeechAntiSpoofingBenchmarks/arena-manifest",
        pr=7,
        branch="refs/pr/7",
    )

    out = capsys.readouterr().out
    assert rc == 0
    assert "arena-manifest preview" in out
    assert "Rows: 2" in out
    assert "Org/B" in out


def test_preview_warns_when_manifest_revision_is_not_fetchable(monkeypatch):
    candidate = _manifest("Org/A")

    monkeypatch.setattr(
        preview_manifest.hf_fetch,
        "list_repo_files",
        lambda dataset_id, repo_type, revision: (_ for _ in ()).throw(
            RuntimeError("revision missing")
        ),
    )
    monkeypatch.setattr(
        preview_manifest.submission,
        "list_submission_files",
        lambda dataset_id: [],
    )

    result = preview_manifest.preview(candidate)

    assert result.rows == 0
    assert len(result.warnings) == 1
    assert result.warnings[0].dataset_id == "Org/A"
    assert result.warnings[0].path == "<revision:abc1234>"
    assert "revision missing" in result.warnings[0].reason


def test_preview_warns_when_dataset_has_no_submission_yamls(monkeypatch):
    candidate = _manifest("Org/A")

    monkeypatch.setattr(
        preview_manifest.hf_fetch,
        "list_repo_files",
        lambda dataset_id, repo_type, revision: ["eval.yaml"],
    )
    monkeypatch.setattr(
        preview_manifest.submission,
        "list_submission_files",
        lambda dataset_id: [],
    )

    result = preview_manifest.preview(candidate)

    assert result.rows == 0
    assert len(result.warnings) == 1
    assert result.warnings[0].dataset_id == "Org/A"
    assert result.warnings[0].path == "<submissions>"
    assert "no submission YAMLs" in result.warnings[0].reason


def test_format_markdown_escapes_warning_table_and_links_run():
    result = preview_manifest.PreviewResult(
        dataset_count=1,
        rows=0,
        warnings=[
            preview_manifest.PreviewWarning(
                "Org/A|B",
                "submissions/a.yaml",
                "bad | reason\nnext line",
            )
        ],
    )

    md = preview_manifest.format_markdown(result, gh_run_url="https://gh/run")

    assert "Org/A\\|B" in md
    assert "bad \\| reason<br>next line" in md
    assert "_[view CI run](https://gh/run)_" in md


def test_run_passes_gh_run_url_from_env(monkeypatch, tmp_path, capsys):
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        "schema_version: 1\n"
        "ranking_version: test\n"
        "metrics_in_use: [eer_percent]\n"
        "tiers:\n"
        "  - {name: gold, min_coverage: 1.0}\n"
        "core_set:\n"
        "  - {id: Org/A, revision: abc1234}\n"
        "extended: []\n"
    )
    monkeypatch.setattr(
        preview_manifest.hf_fetch,
        "list_repo_files",
        lambda dataset_id, repo_type, revision: ["eval.yaml"],
    )
    monkeypatch.setattr(
        preview_manifest.submission,
        "list_submission_files",
        lambda dataset_id: ["submissions/a.yaml"],
    )
    monkeypatch.setattr(
        preview_manifest.submission,
        "fetch_submission",
        lambda dataset_id, path: _submission(),
    )
    monkeypatch.setenv("GH_RUN_URL", "https://gh/run/123")

    rc = preview_manifest.run(manifest_path=manifest_path)

    out = capsys.readouterr().out
    assert rc == 0
    assert "https://gh/run/123" in out
