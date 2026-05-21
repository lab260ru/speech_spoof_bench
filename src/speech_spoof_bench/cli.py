"""Command-line interface for speech-spoof-bench."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
from typing import Sequence

from .benchmark import Benchmark
from .loader import resolve


def _import_model_class(spec: str):
    if ":" not in spec:
        raise SystemExit(
            f"--model-module must be <module>:<ClassName>, got {spec!r}"
        )
    mod_name, cls_name = spec.split(":", 1)
    module = importlib.import_module(mod_name)
    try:
        return getattr(module, cls_name)
    except AttributeError:
        raise SystemExit(f"class {cls_name!r} not found in module {mod_name!r}")


def _cmd_run(args: argparse.Namespace) -> int:
    cls = _import_model_class(args.model_module)
    model = cls()
    Benchmark.run(
        model,
        datasets=list(args.datasets),
        output_dir=args.output_dir,
        streaming=not args.no_streaming,
        cleanup=not args.no_cleanup,
        skip_existing=not args.no_skip_existing,
    )
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    from . import manifest as mf

    m = mf.fetch_manifest()
    for entry in m["core_set"]:
        print(f"[core] {entry['id']}")
    for entry in m["extended"]:
        print(f"[ext]  {entry['id']}")
    return 0


def _cmd_validate_dataset(args: argparse.Namespace) -> int:
    source, ds = resolve(args.spec, streaming=True)
    first = next(iter(ds))
    expected = {"path", "audio", "label", "notes"}
    actual = set(first.keys())
    if not expected.issubset(actual):
        raise SystemExit(
            f"dataset row missing required columns: {expected - actual} "
            f"(got {sorted(actual)})"
        )
    notes = json.loads(first["notes"])
    if not notes.get("utterance_id"):
        raise SystemExit("first row's notes JSON has no non-empty 'utterance_id'")
    print(f"OK: {source.canonical_id} (display: {source.display_name!r})")
    return 0


def _cmd_manifest(args: argparse.Namespace) -> int:
    """Print the raw manifest.yaml contents verbatim to stdout."""
    from pathlib import Path
    from . import manifest as mf

    # Reuse the repo coordinates from the manifest module, but skip the
    # parse/validate round-trip so the output equals the upstream file byte-for-byte.
    local = mf.hf_hub_download(
        repo_id=mf.MANIFEST_REPO,
        repo_type="dataset",
        filename=mf.MANIFEST_FILENAME,
    )
    sys.stdout.write(Path(local).read_text())
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="speech-spoof-bench")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="run a model against one or more datasets")
    run.add_argument("--model-module", required=True,
                     help="module:ClassName, e.g. mypkg.mymod:MyModel")
    run.add_argument("--datasets", action="append", required=True,
                     help="local dir path OR org/name HF repo id; repeatable")
    run.add_argument("--output-dir", default="./results")
    run.add_argument("--no-streaming", action="store_true")
    run.add_argument("--no-cleanup", action="store_true")
    run.add_argument("--no-skip-existing", action="store_true")
    run.set_defaults(func=_cmd_run)

    lst = sub.add_parser("list", help="list datasets in the arena manifest")
    lst.set_defaults(func=_cmd_list)

    man = sub.add_parser("manifest",
                         help="print the arena-manifest YAML contents")
    man.set_defaults(func=_cmd_manifest)

    vd = sub.add_parser("validate-dataset",
                        help="quick check that a dataset loads with the v4 schema")
    vd.add_argument("spec", help="local dir path or org/name HF repo id")
    vd.set_defaults(func=_cmd_validate_dataset)

    return p


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
