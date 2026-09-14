"""Command-line interface for dataset validation and causal audits."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from .adapters import HttpTarget
from .models import AuditReport
from .reports import write_reports
from .runner import AuditRunner
from .scoring import METRIC_NAMES
from .validation import DatasetValidationError, load_dataset
from .version import __version__


def _import_target(specification: str) -> Callable[..., Any]:
    if ":" not in specification:
        raise ValueError("target must use the form package.module:callable")
    module_name, attribute_name = specification.split(":", 1)
    if not module_name or not attribute_name:
        raise ValueError("target must use the form package.module:callable")
    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    module = importlib.import_module(module_name)
    target = getattr(module, attribute_name)
    if not callable(target):
        raise TypeError(f"{specification} is not callable")
    return cast(Callable[..., Any], target)


def _headers_from_environment(values: list[str]) -> dict[str, str]:
    headers = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--header-from-env must use HEADER=ENVIRONMENT_VARIABLE")
        header, variable = value.split("=", 1)
        if not header or not variable:
            raise ValueError("--header-from-env must use HEADER=ENVIRONMENT_VARIABLE")
        if variable not in os.environ:
            raise ValueError(f"environment variable is not set: {variable}")
        headers[header] = os.environ[variable]
    return headers


def _thresholds(values: list[str]) -> dict[str, float]:
    thresholds = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--minimum must use METRIC=RATE")
        name, raw_rate = value.split("=", 1)
        if name not in METRIC_NAMES:
            raise ValueError(
                f"unknown threshold metric {name!r}; choose from {', '.join(METRIC_NAMES)}"
            )
        try:
            rate = float(raw_rate)
        except ValueError as exc:
            raise ValueError(f"threshold for {name} is not a number") from exc
        if not 0 <= rate <= 1:
            raise ValueError(f"threshold for {name} must be between 0 and 1")
        thresholds[name] = rate
    return thresholds


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="causal-rag-audit",
        description="Test whether a RAG system behaviorally relies on its cited evidence.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="validate an audit dataset")
    validate.add_argument("dataset", type=Path)

    run = commands.add_parser(
        "run", help="run an audit against a callable or HTTP target"
    )
    run.add_argument("--dataset", type=Path, required=True)
    targets = run.add_mutually_exclusive_group(required=True)
    targets.add_argument("--target", help="Python callable as package.module:attribute")
    targets.add_argument("--http-url", help="JSON endpoint accepting POST requests")
    run.add_argument("--target-name")
    run.add_argument("--output", type=Path, default=Path("audit-results"))
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--timeout", type=float, default=60.0)
    run.add_argument(
        "--header-from-env",
        action="append",
        default=[],
        metavar="HEADER=VARIABLE",
        help="read a sensitive HTTP header value from an environment variable",
    )
    run.add_argument(
        "--minimum",
        action="append",
        default=[],
        metavar="METRIC=RATE",
        help="fail with exit code 2 when an aggregate metric is below RATE",
    )
    run.add_argument(
        "--include-raw",
        action="store_true",
        help="include raw target output in report.json; review for sensitive data",
    )
    return parser


def _print_metrics(report: AuditReport) -> None:
    print(
        f"Audited {len(report.cases)} cases with {report.configuration['calls']} target calls."
    )
    for name, metric in report.metrics.items():
        print(f"  {name}: {metric.count}/{metric.n} ({metric.rate:.1%})")
    print(f"  target_errors: {report.error_count}")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        dataset = load_dataset(args.dataset)
        if args.command == "validate":
            print(
                f"Valid dataset {dataset.name!r}: {len(dataset.cases)} cases, "
                f"{len(dataset.cases) * 3} target calls per audit."
            )
            return 0

        thresholds = _thresholds(args.minimum)
        if args.target:
            target = _import_target(args.target)
            inferred_name = args.target
        else:
            headers = _headers_from_environment(args.header_from_env)
            target = HttpTarget(
                args.http_url,
                timeout_seconds=args.timeout,
                headers=headers,
            )
            inferred_name = args.http_url
        report = AuditRunner(
            target,
            target_name=args.target_name or inferred_name,
            max_workers=args.workers,
            include_raw=args.include_raw,
        ).run(dataset)
        json_path, markdown_path = write_reports(report, args.output)
        _print_metrics(report)
        print(f"Wrote {json_path} and {markdown_path}.")

        failed = [
            (name, report.metrics[name].rate, minimum)
            for name, minimum in thresholds.items()
            if report.metrics[name].rate < minimum
        ]
        for name, actual, minimum in failed:
            print(
                f"Threshold failed: {name}={actual:.3f} is below {minimum:.3f}.",
                file=sys.stderr,
            )
        if report.error_count or failed:
            return 2
        return 0
    except (
        DatasetValidationError,
        ImportError,
        AttributeError,
        TypeError,
        ValueError,
        OSError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
