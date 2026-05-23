"""Command-line interface for speech-spoof-bench."""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from pathlib import Path
from typing import Sequence

from .benchmark import Benchmark


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
        force_remote=args.no_local,
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
    from . import validate
    report = validate.validate_dataset(
        args.spec,
        skip_submissions=args.skip_submissions,
        force_remote=args.no_local,
    )
    print(report.format())
    return 0 if report.ok else 1


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


def _cmd_reproduce(args: argparse.Namespace) -> int:
    if args.inference:
        raise NotImplementedError("reproduce --inference lands in Phase 7b/8")
    from . import reproduce
    return reproduce.run_scoring(
        args.path, tolerance=args.tolerance, force_remote=args.no_local,
    )


def _cmd_validate_submission(args: argparse.Namespace) -> int:
    from . import submission
    path = args.path
    try:
        text = open(path).read()
        submission.parse_submission(text)
    except FileNotFoundError as e:
        print(f"FAIL {path}: {e}", file=sys.stderr)
        return 1
    except submission.SubmissionValidationError as e:
        print(f"FAIL {path}: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"FAIL {path}: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    print(f"OK: {path}")
    return 0


def _cmd_scaffold_dataset(args: argparse.Namespace) -> int:
    from . import scaffold

    scaffold.scaffold_dataset(
        name=args.name, output_dir=args.output_dir, force=args.force,
    )
    print(f"scaffolded dataset skeleton at {args.output_dir}")
    return 0


def _cmd_local_set(args: argparse.Namespace) -> int:
    from . import local_registry as lr
    lr.set(args.dataset_id, args.path)
    print(f"registered {args.dataset_id} -> {args.path}")
    return 0


def _cmd_local_unset(args: argparse.Namespace) -> int:
    from . import local_registry as lr
    lr.unset(args.dataset_id)
    print(f"unset {args.dataset_id}")
    return 0


def _cmd_local_list(args: argparse.Namespace) -> int:
    from . import local_registry as lr
    mapping = lr.load()
    if not mapping:
        print("(no local datasets registered)")
        return 0
    for did, p in sorted(mapping.items()):
        print(f"{did}\t{p}")
    return 0


def _cmd_local_show(args: argparse.Namespace) -> int:
    from . import local_registry as lr
    try:
        p = lr.lookup(args.dataset_id)
    except FileNotFoundError as e:
        print(f"{args.dataset_id}\tBROKEN: {e}")
        return 1
    if p is None:
        print(f"{args.dataset_id}\tremote (no local registration)")
    else:
        print(f"{args.dataset_id}\tlocal: {p}")
    return 0


def _cmd_ci_verify_pr(args: argparse.Namespace) -> int:
    from .ci import verify_pr
    return verify_pr.run(repo=args.repo, pr=int(args.pr), branch=args.branch)


def _cmd_ci_nightly(args: argparse.Namespace) -> int:
    from .ci import nightly
    return nightly.run(open_issues=args.open_issues)


def _cmd_submit(args: argparse.Namespace) -> int:
    from . import submit as submit_mod

    results = submit_mod.submit(
        model_module_spec=args.model_module,
        dataset_specs=list(args.datasets),
        output_dir=args.output_dir,
        meta_path=args.submission_meta,
        model_repo=args.model_repo,
        hf_username=args.hf_username,
        contact=args.contact,
        continue_on_error=args.continue_on_error,
        force_remote=args.no_local,
    )
    for spec, url in results.items():
        print(f"{spec}\t{url}")
    return 0 if all(not str(v).startswith("ERROR:") for v in results.values()) else 1


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
    run.add_argument(
        "--no-local",
        action="store_true",
        help="ignore the local-dataset registry; always stream from HF",
    )
    run.set_defaults(func=_cmd_run)

    lst = sub.add_parser("list", help="list datasets in the arena manifest")
    lst.set_defaults(func=_cmd_list)

    man = sub.add_parser("manifest",
                         help="print the arena-manifest YAML contents")
    man.set_defaults(func=_cmd_manifest)

    vd = sub.add_parser(
        "validate-dataset",
        help="full §1.9 dataset + submission validation",
    )
    vd.add_argument("spec", help="local dir path or org/name HF repo id")
    vd.add_argument("--skip-submissions", action="store_true",
                    help="skip per-submission network checks")
    vd.add_argument(
        "--no-local",
        action="store_true",
        help="ignore the local-dataset registry; always stream from HF",
    )
    vd.set_defaults(func=_cmd_validate_dataset)

    vs = sub.add_parser(
        "validate-submission",
        help="schema-check a submission YAML offline (no network)",
    )
    vs.add_argument("path", help="path to a submission YAML file")
    vs.set_defaults(func=_cmd_validate_submission)

    sm = sub.add_parser("submit", help="run model + upload scores + open PR on dataset repo")
    sm.add_argument("--model-module", required=True,
                    help="module:ClassName, e.g. mypkg.mymod:MyModel")
    sm.add_argument("--datasets", action="append", required=True,
                    help="HF dataset id; repeatable; use 'all' for manifest-wide")
    sm.add_argument("--model-repo", required=True,
                    help="HF model repo (owner/name) that owns the scores.txt")
    sm.add_argument("--submission-meta", required=True, type=Path,
                    help="path to meta.yaml describing the system")
    sm.add_argument("--hf-username", required=True)
    sm.add_argument("--contact", required=True)
    sm.add_argument("--output-dir", default="./results", type=Path)
    sm.add_argument("--continue-on-error", action="store_true")
    sm.add_argument(
        "--no-local",
        action="store_true",
        help="ignore the local-dataset registry; always stream from HF",
    )
    sm.set_defaults(func=_cmd_submit)

    sd = sub.add_parser("scaffold-dataset",
                        help="generate the §1.1 dataset-repo skeleton")
    sd.add_argument("--name", required=True)
    sd.add_argument("--output-dir", required=True, type=Path)
    sd.add_argument("--force", action="store_true",
                    help="overwrite if output-dir is non-empty")
    sd.set_defaults(func=_cmd_scaffold_dataset)

    rp = sub.add_parser(
        "reproduce",
        help="reproduce a submission's scores (--scoring) or "
             "full inference (--inference)",
    )
    rp.add_argument("path", help="path to a submission YAML file")
    rp.add_argument("--tolerance", type=float, default=1e-6,
                    help="metric tolerance for --scoring (default 1e-6)")
    mode = rp.add_mutually_exclusive_group(required=True)
    mode.add_argument("--scoring", action="store_true",
                      help="verify scores_url sha + recompute metrics")
    mode.add_argument("--inference", action="store_true",
                      help="full re-inference (Phase 8+; not yet implemented)")
    rp.add_argument(
        "--no-local",
        action="store_true",
        help="ignore the local-dataset registry; always stream from HF",
    )
    rp.set_defaults(func=_cmd_reproduce)

    loc = sub.add_parser(
        "local",
        help="manage the local-dataset registry (speech-spoof-bench/local-datasets.yaml)",
    )
    loc_sub = loc.add_subparsers(dest="local_cmd", required=True)

    ls_set = loc_sub.add_parser("set", help="register dataset_id -> local path")
    ls_set.add_argument("dataset_id")
    ls_set.add_argument("path")
    ls_set.set_defaults(func=_cmd_local_set)

    ls_unset = loc_sub.add_parser("unset", help="remove a registration")
    ls_unset.add_argument("dataset_id")
    ls_unset.set_defaults(func=_cmd_local_unset)

    ls_list = loc_sub.add_parser("list", help="show all registered mappings")
    ls_list.set_defaults(func=_cmd_local_list)

    ls_show = loc_sub.add_parser("show", help="show resolution for one dataset id")
    ls_show.add_argument("dataset_id")
    ls_show.set_defaults(func=_cmd_local_show)

    ci = sub.add_parser("ci", help="CI commands (Phase 8)")
    ci_sub = ci.add_subparsers(dest="ci_cmd", required=True)

    vpr = ci_sub.add_parser("verify-pr",
        help="validate + reproduce changed submissions on an HF dataset PR")
    vpr.add_argument("--repo", required=True, help="dataset id (org/name)")
    vpr.add_argument("--pr", required=True, help="HF PR (discussion) number")
    vpr.add_argument("--branch", required=True,
        help="ref to fetch the PR contents from, e.g. refs/pr/42")
    vpr.set_defaults(func=_cmd_ci_verify_pr)

    nr = ci_sub.add_parser("nightly-revalidate",
        help="walk all merged submissions, open/close stale-submission issues")
    nr.add_argument("--open-issues", action="store_true",
        help="open/comment/close GitHub issues for failures (requires gh + GH_TOKEN)")
    nr.set_defaults(func=_cmd_ci_nightly)

    return p


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
