"""JSON and Markdown report rendering."""

from __future__ import annotations

import json
from pathlib import Path

from .models import AuditReport


def report_as_markdown(report: AuditReport) -> str:
    """Render a human-readable audit report with per-case diagnostics."""

    lines = [
        f"# Causal RAG Audit: {report.dataset_name}",
        "",
        f"- Target: `{report.target_name}`",
        f"- Framework: `{report.framework_version}`",
        f"- Dataset SHA-256: `{report.dataset_fingerprint}`",
        f"- Cases: {len(report.cases)}",
        f"- Target errors: {report.error_count}",
        "",
        "## Metrics",
        "",
        "| Metric | Result | Wilson 95% interval |",
        "|---|---:|---:|",
    ]
    for name, metric in report.metrics.items():
        low, high = metric.wilson95
        lines.append(
            f"| `{name}` | {metric.count}/{metric.n} ({metric.rate:.1%}) | "
            f"{low:.1%}–{high:.1%} |"
        )
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
