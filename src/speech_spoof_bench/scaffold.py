"""Phase 7b — `scaffold-dataset` command.

Produces the §1.1 skeleton for a new dataset repo by copying packaged
template files into `output_dir` and substituting `{{NAME}}` in README.md
and eval.yaml.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

_TEMPLATE_PACKAGE = "speech_spoof_bench.data.dataset_skeleton"
_SUBSTITUTE_IN = {"README.md", "eval.yaml"}


def _iter_template_files() -> list[tuple[str, bytes]]:
    """Walk the packaged template and yield (relpath, bytes) pairs."""
    root = resources.files(_TEMPLATE_PACKAGE)
    out: list[tuple[str, bytes]] = []

    def walk(node, rel: str) -> None:
        for child in node.iterdir():
            if child.is_dir():
                walk(child, f"{rel}{child.name}/")
            else:
                name = child.name
                if name == "__init__.py":
                    continue
                out.append((f"{rel}{name}", child.read_bytes()))

    walk(root, "")
    return out


def scaffold_dataset(
    *, name: str, output_dir: Path | str, force: bool = False,
) -> None:
    """Materialize the dataset skeleton under `output_dir`.

    Raises:
      FileExistsError: `output_dir` exists and is non-empty without force.
    """
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()) and not force:
        raise FileExistsError(
            f"{out} already exists and is non-empty (use force=True to overwrite)"
        )
    out.mkdir(parents=True, exist_ok=True)

    for relpath, data in _iter_template_files():
        target = out / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        if Path(relpath).name in _SUBSTITUTE_IN:
            text = data.decode("utf-8").replace("{{NAME}}", name)
            target.write_text(text)
        else:
            target.write_bytes(data)
