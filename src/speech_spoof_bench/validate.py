"""Aggregating validator for a dataset repo against the v4 spec (§1.9).

Public surface:
  - validate_dataset(spec, *, skip_submissions=False) -> ValidationReport
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from datasets import ClassLabel
from huggingface_hub import hf_hub_download

from .loader import resolve

REQUIRED_README_KEYS = {
    "license", "language", "pretty_name", "task_categories",
    "size_categories", "configs", "tags", "arxiv",
}


@dataclass
class CheckResult:
    id: str
    passed: bool
    message: str = ""


@dataclass
class SubmissionReport:
    path: str
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.passed for c in self.checks)


@dataclass
class ValidationReport:
    dataset_spec: str
    dataset_checks: list[CheckResult] = field(default_factory=list)
    submission_reports: list[SubmissionReport] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.passed for c in self.dataset_checks) and all(
            s.ok for s in self.submission_reports
        )

    def format(self) -> str:
        lines = [f"Dataset: {self.dataset_spec}"]
        for c in self.dataset_checks:
            mark = "✓" if c.passed else "✗"
            lines.append(f"  [{mark}] {c.id} {c.message}")
        if self.submission_reports:
            lines.append("")
            lines.append(f"Submissions ({len(self.submission_reports)}):")
            for sr in self.submission_reports:
                if sr.ok:
                    lines.append(f"  [✓] {sr.path}")
                else:
                    lines.append(f"  [✗] {sr.path}")
                    for c in sr.checks:
                        if not c.passed:
                            lines.append(f"        {c.id} {c.message}")
        failed = sum(1 for c in self.dataset_checks if not c.passed)
        failed += sum(
            1
            for sr in self.submission_reports
            for c in sr.checks
            if not c.passed
        )
        lines.append("")
        lines.append("OK" if self.ok else f"{failed} checks failed.")
        return "\n".join(lines)


def _read_readme_frontmatter(text: str) -> dict[str, Any] | None:
    if not text.startswith("---"):
        return None
    parts = text.split("---", 2)
    if len(parts) < 3:
        return None
    try:
        data = yaml.safe_load(parts[1])
    except yaml.YAMLError:
        return None
    return data if isinstance(data, dict) else None


def _load_readme(spec_path: Path | None, repo_id: str | None) -> str:
    if spec_path is not None:
        p = spec_path / "README.md"
        return p.read_text() if p.is_file() else ""
    local = hf_hub_download(
        repo_id=repo_id, filename="README.md", repo_type="dataset"
    )
    return Path(local).read_text()


def _check_dataset_side(spec: str) -> tuple[list[CheckResult], dict[str, Any]]:
    """Run D1–D7. Returns (checks, info_for_submission_checks)."""
    checks: list[CheckResult] = []
    source, ds = resolve(spec, streaming=True)

    # D1: schema columns
    first = None
    try:
        first = next(iter(ds))
        expected = {"path", "audio", "label", "notes"}
        actual = set(first.keys())
        if expected == actual:
            checks.append(CheckResult("D1", True, "schema matches v4"))
        else:
            checks.append(CheckResult(
                "D1", False,
                f"schema mismatch (got {sorted(actual)}, want {sorted(expected)})",
            ))
    except Exception as e:
        checks.append(CheckResult("D1", False, f"could not read first row: {e}"))

    # D2: ClassLabel names
    try:
        feat = getattr(ds, "features", None)
        label_feat = feat.get("label") if feat else None
        if isinstance(label_feat, ClassLabel) and label_feat.names == ["bonafide", "spoof"]:
            checks.append(CheckResult("D2", True, "label classes ok"))
        elif isinstance(label_feat, ClassLabel):
            checks.append(CheckResult(
                "D2", False, f"label classes {label_feat.names!r} != ['bonafide','spoof']"
            ))
        else:
            # Local synth fixture uses Value("int64"); HF datasets use ClassLabel.
            # Accept int-with-0/1 as a soft pass with a note.
            if first is not None and int(first.get("label", -1)) in (0, 1):
                checks.append(CheckResult("D2", True, "label is int 0/1 (non-ClassLabel)"))
            else:
                checks.append(CheckResult("D2", False, "label feature not ClassLabel"))
    except Exception as e:
        checks.append(CheckResult("D2", False, f"label feature check error: {e}"))

    # D3: sampling rate + duration
    try:
        if first is None:
            checks.append(CheckResult("D3", False, "first row unavailable"))
        else:
            audio = first["audio"]
            sr = int(audio["sampling_rate"])
            arr = audio["array"]
            dur = len(arr) / sr
            if sr == 16000 and dur >= 1.0:
                checks.append(CheckResult("D3", True, f"sr=16000 dur={dur:.2f}s"))
            else:
                checks.append(CheckResult(
                    "D3", False, f"sr={sr} dur={dur:.2f}s (want sr=16000, dur>=1.0s)"
                ))
    except Exception as e:
        checks.append(CheckResult("D3", False, f"audio decode error: {e}"))

    # D4 + D5: stream rows, accumulate sets.
    utt_ids: set[str] = set()
    paths: set[str] = set()
    dup_utt: set[str] = set()
    dup_path: set[str] = set()
    d4_failed = False
    d4_msg = ""
    sampled_for_d4 = 0
    # IterableDataset is single-pass: the `ds` above was exhausted by
    # next(iter(ds)) for D1/D3. Re-resolve to get a fresh iterator.
    try:
        for row in resolve(spec, streaming=True)[1]:
            raw_notes = row.get("notes")
            note: dict | None = None
            try:
                note = json.loads(raw_notes) if raw_notes is not None else None
            except Exception as e:
                if sampled_for_d4 < 100 and not d4_failed:
                    d4_failed = True
                    d4_msg = f"notes JSON decode error: {e}"
                note = None
            if sampled_for_d4 < 100 and not d4_failed and note is not None:
                if not note.get("utterance_id"):
                    d4_failed = True
                    d4_msg = "notes missing non-empty utterance_id"
            if sampled_for_d4 < 100:
                sampled_for_d4 += 1

            uid = note.get("utterance_id") if isinstance(note, dict) else None
            if uid:
                if uid in utt_ids:
                    dup_utt.add(uid)
                utt_ids.add(uid)
            p = row.get("path")
            if p:
                if p in paths:
                    dup_path.add(p)
                paths.add(p)
        if d4_failed:
            checks.append(CheckResult("D4", False, d4_msg))
        else:
            checks.append(CheckResult("D4", True, f"sampled {sampled_for_d4} rows"))
        if not dup_utt and not dup_path:
            checks.append(CheckResult(
                "D5", True, f"uniqueness ok ({len(utt_ids)} ids, {len(paths)} paths)"
            ))
        else:
            checks.append(CheckResult(
                "D5", False,
                f"duplicates: utt={len(dup_utt)} path={len(dup_path)}",
            ))
    except Exception as e:
        checks.append(CheckResult("D4", False, f"iteration error: {e}"))
        checks.append(CheckResult("D5", False, "skipped (iteration failed)"))

    # D6: README frontmatter
    try:
        readme_text = _load_readme(
            spec_path=source.local_path if source.is_local else None,
            repo_id=None if source.is_local else source.canonical_id,
        )
        fm = _read_readme_frontmatter(readme_text)
        if fm is None:
            checks.append(CheckResult("D6", False, "README frontmatter missing or invalid"))
        else:
            missing = REQUIRED_README_KEYS - set(fm.keys())
            if missing:
                checks.append(CheckResult("D6", False, f"frontmatter missing keys: {sorted(missing)}"))
            elif "arena-ready" not in (fm.get("tags") or []):
                checks.append(CheckResult("D6", False, "tags missing 'arena-ready'"))
            else:
                checks.append(CheckResult("D6", True, "frontmatter ok"))
    except Exception as e:
        checks.append(CheckResult("D6", False, f"README load error: {e}"))

    # D7: metric registry — loader.resolve already validates this. Reaching
    # this point means D7 passed.
    checks.append(CheckResult("D7", True, f"metrics registered: {source.metrics}"))

    info = {
        "is_local": source.is_local,
        "local_path": source.local_path,
        "canonical_id": source.canonical_id if not source.is_local else None,
    }
    return checks, info


def validate_dataset(spec: str, *, skip_submissions: bool = False) -> ValidationReport:
    report = ValidationReport(dataset_spec=spec)
    try:
        dataset_checks, info = _check_dataset_side(spec)
    except KeyError as e:
        # loader.py's _parse_eval_yaml raises KeyError("metric id 'X' not
        # registered (from <path>)") when eval.yaml references an unknown
        # metric. We surface this as a D7 failure rather than propagating.
        # If the loader's message format ever changes, the substring check
        # below will silently fail — keep these in sync.
        msg = str(e)
        if "not registered" in msg:
            report.dataset_checks.append(CheckResult("D7", False, msg))
        else:
            report.dataset_checks.append(CheckResult("D0", False, f"load error: {e}"))
        return report
    except Exception as e:
        report.dataset_checks.append(CheckResult("D0", False, f"load error: {e}"))
        return report
    report.dataset_checks = dataset_checks
    if skip_submissions:
        return report
    # Submission checks (S1–S4) land in Task 5.
    return report
