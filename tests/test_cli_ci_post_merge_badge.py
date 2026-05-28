"""CLI smoke: `ci post-merge-badge` dispatches to post_merge_badge.run."""
from __future__ import annotations

from unittest.mock import MagicMock

from speech_spoof_bench import cli


def test_cli_dispatches_with_args(monkeypatch):
    seen = {}
    def fake_run(**kwargs):
        seen.update(kwargs)
        return 0
    from speech_spoof_bench.ci import post_merge_badge
    monkeypatch.setattr(post_merge_badge, "run", fake_run)

    rc = cli.main([
        "ci", "post-merge-badge",
        "--repo", "Org/Foo",
        "--pr", "42",
        "--sha", "deadbeef",
    ])
    assert rc == 0
    assert seen == {"repo": "Org/Foo", "pr": 42, "sha": "deadbeef"}
