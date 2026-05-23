"""scaffold_dataset generates the §1.1 skeleton with {{NAME}} substitutions."""

from __future__ import annotations

from pathlib import Path

import pytest

from speech_spoof_bench.scaffold import scaffold_dataset


def test_scaffold_writes_all_expected_files(tmp_path: Path):
    out = tmp_path / "MyDataset"
    scaffold_dataset(name="MyDataset", output_dir=out)

    expected = {
        "README.md",
        "LICENSE.txt",
        "eval.yaml",
        "build_parquet.py",
        "submissions/README.md",
        "submissions/results_template.yaml",
    }
    actual = {
        str(p.relative_to(out)).replace("\\", "/")
        for p in out.rglob("*") if p.is_file()
    }
    actual = {a for a in actual if not a.endswith("__init__.py")}
    assert expected.issubset(actual)


def test_scaffold_substitutes_name_token(tmp_path: Path):
    out = tmp_path / "InTheWild"
    scaffold_dataset(name="InTheWild", output_dir=out)
    readme = (out / "README.md").read_text()
    eval_yaml = (out / "eval.yaml").read_text()
    assert "{{NAME}}" not in readme
    assert "{{NAME}}" not in eval_yaml
    assert "InTheWild" in readme
    assert "name: InTheWild" in eval_yaml


def test_scaffold_refuses_nonempty_dir_without_force(tmp_path: Path):
    out = tmp_path / "X"
    out.mkdir()
    (out / "stuff.txt").write_text("hi")
    with pytest.raises(FileExistsError):
        scaffold_dataset(name="X", output_dir=out)


def test_scaffold_force_overwrites(tmp_path: Path):
    out = tmp_path / "X"
    out.mkdir()
    (out / "stuff.txt").write_text("hi")
    scaffold_dataset(name="X", output_dir=out, force=True)
    assert (out / "README.md").is_file()


def test_scaffold_empty_dir_ok(tmp_path: Path):
    out = tmp_path / "X"
    out.mkdir()
    scaffold_dataset(name="X", output_dir=out)
    assert (out / "eval.yaml").is_file()
