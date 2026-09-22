"""JSON and Markdown report rendering."""

from __future__ import annotations

import json
from pathlib import Path

from .models import AuditReport


def report_as_markdown(report: AuditReport) -> str:
    """Render a human-readable audit report with per-case diagnostics."""

    if report.report_schema_version == 2:
        return _extended_report_as_markdown(report)

    lines = [
        f"# Causal RAG Audit: {report.dataset_name}",
        "",
        f"- Target: `{report.target_name}`",
        f"- Framework: `{report.framework_version}`",
        f"- Scoring profile: `{report.configuration['scoring_profile']}`",
        f"- Dataset SHA-256: `{report.dataset_fingerprint}`",
        f"- Cases: {len(report.cases)}",
        f"- Response issues: {report.error_count}",
        "",
        "## Metrics",
        "",
        "| Metric | Result | Wilson 95% interval |",
        "|---|---:|---:|",
    ]
    for name, binary_metric in report.metrics.items():
        low, high = binary_metric.wilson95
        lines.append(
            f"| `{name}` | {binary_metric.count}/{binary_metric.n} "
            f"({binary_metric.rate:.1%}) | "
            f"{low:.1%}–{high:.1%} |"
        )
    lines.extend(
        [
            "",
            "## Proof citation metrics",
            "",
            "| Metric | Macro mean | N |",
            "|---|---:|---:|",
        ]
    )
    for name, proof_metric in report.proof_metrics.items():
        lines.append(f"| `{name}` | {proof_metric.mean:.1%} | {proof_metric.n} |")
    lines.extend(
        [
            "",
            "## Cases",
            "",
            "| Case | OBS | CRC | ENA | CES coverage | CES strict | Errors |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for case in report.cases:
        indicators = case.indicators
        errors = sum(response.error is not None for response in case.responses)

        def mark(value: bool) -> str:
            return "PASS" if value else "FAIL"

        lines.append(
            f"| `{case.case_id}` | {mark(indicators['observational_support'])} | "
            f"{mark(indicators['paired_responsiveness'])} | "
            f"{mark(indicators['ablation_abstention'])} | "
            f"{mark(indicators['coverage_causal_evidence_score'])} | "
            f"{mark(indicators['strict_causal_evidence_score'])} | {errors} |"
        )

    failures = [
        case
        for case in report.cases
        if not case.indicators["strict_causal_evidence_score"]
    ]
    if failures:
        lines.extend(["", "## Failure details", ""])
        for case in failures:
            lines.append(f"### `{case.case_id}`")
            lines.append("")
            for response in case.responses:
                lines.append(
                    f"- `{response.condition}`: answer=`{response.answer or '<none>'}`, "
                    f"citations={list(response.citations)}, correct={response.correct}, "
                    f"complete={response.citation_complete}, valid={response.citation_valid}"
                    + (f", error={response.error}" if response.error else "")
                )
            lines.append("")
    lines.extend(
        [
            "## Interpretation boundary",
            "",
            (
                "This report measures behavioral reliance under the declared evidence "
                "interventions. It does not establish source truth, model intent, or "
                "internal neural causality."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _extended_report_as_markdown(report: AuditReport) -> str:
    lines = [
        f"# CEA-Extended audit: {report.dataset_name}",
        "",
        f"- Target: `{report.target_name}`",
        f"- Framework: `{report.framework_version}`",
        f"- Scoring profile: `{report.configuration['scoring_profile']}`",
        f"- Dataset SHA-256: `{report.dataset_fingerprint}`",
        f"- Cases: {len(report.cases)}",
        f"- Target calls: {report.configuration['calls']}",
        f"- Response issues: {report.error_count}",
        "",
        "## Condition coverage",
        "",
        "| Family | Declared conditions |",
        "|---|---:|",
    ]
    for family, count in report.condition_coverage.items():
        lines.append(f"| `{family}` | {count} |")
    lines.extend(
        [
            "",
            "## Joint and family metrics",
            "",
            "Wilson intervals treat cases as independent; shared templates or domains require clustered analysis.",
            "",
            "| Metric | Result | Per-case Wilson 95% interval |",
            "|---|---:|---:|",
        ]
    )
    for name, metric in report.metrics.items():
        low, high = metric.wilson95
        lines.append(
            f"| `{name}` | {metric.count}/{metric.n} ({metric.rate:.1%}) | {low:.1%}–{high:.1%} |"
        )
    lines.extend(
        [
            "",
            "## Proof citation metrics",
            "",
            "| Metric | Macro mean | Cases |",
            "|---|---:|---:|",
        ]
    )
    for name, proof_metric in report.proof_metrics.items():
        lines.append(f"| `{name}` | {proof_metric.mean:.1%} | {proof_metric.n} |")
    lines.extend(
        [
            "",
            "## Condition diagnostics",
            "",
            "| Case | Condition | Family | Answer correct | Citation coverage | Citation exact | Citation valid | Format valid | Error |",
            "|---|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for case in report.cases:
        diagnostics = case.metadata["condition_diagnostics"]
        for response in case.responses:
            diagnostic = diagnostics[response.condition]
            lines.append(
                f"| `{case.case_id}` | `{response.condition}` | `{diagnostic['family']}` | "
                f"{response.correct} | {response.citation_complete} | {response.citation_exact} | "
                f"{response.citation_valid} | {response.format_valid} | {response.error or ''} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            (
                "The structural validator checks declared edits and proof-set relationships, "
                "but cannot prove that document claims entail the oracle answers. "
                "A domain reviewer must verify every rationale and proof set. "
                "The audit tests behavioral reliance under these declared interventions, "
                "not source truth or internal neural causality."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def write_reports(
    report: AuditReport, output_directory: str | Path
) -> tuple[Path, Path]:
    """Write stable ``report.json`` and ``report.md`` files."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "report.json"
    markdown_path = output / "report.md"
    json_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(report_as_markdown(report), encoding="utf-8")
    return json_path, markdown_path
