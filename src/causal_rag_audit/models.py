"""Public data models for causal RAG audits."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

CONDITIONS = ("world0", "world1", "ablated")


@dataclass(frozen=True)
class Document:
    """One evidence document with a stable identifier."""

    id: str
    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AuditCase:
    """A matched pair of evidence worlds and its necessary-evidence ablation."""

    id: str
    question: str
    world0: tuple[Document, ...]
    world1: tuple[Document, ...]
    answer_world0: str
    answer_world1: str
    proof_world0: tuple[str, ...]
    proof_world1: tuple[str, ...]
    ablate_document_ids: tuple[str, ...]
    causal_change_ids: tuple[str, ...]
    nuisance_change_ids: tuple[str, ...] = ()
    aliases_world0: tuple[str, ...] = ()
    aliases_world1: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def documents_for(self, condition: str) -> tuple[Document, ...]:
        if condition == "world0":
            return self.world0
        if condition == "world1":
            return self.world1
        if condition == "ablated":
            removed = set(self.ablate_document_ids)
            return tuple(doc for doc in self.world0 if doc.id not in removed)
        raise ValueError(f"unknown audit condition: {condition}")

    def expected_answer_for(self, condition: str) -> str | None:
        if condition == "world0":
            return self.answer_world0
        if condition == "world1":
            return self.answer_world1
        if condition == "ablated":
            return None
        raise ValueError(f"unknown audit condition: {condition}")

    def aliases_for(self, condition: str) -> tuple[str, ...]:
        if condition == "world0":
            return self.aliases_world0
        if condition == "world1":
            return self.aliases_world1
        if condition == "ablated":
            return ()
        raise ValueError(f"unknown audit condition: {condition}")

    def proof_for(self, condition: str) -> tuple[str, ...]:
        if condition == "world0":
            return self.proof_world0
        if condition == "world1":
            return self.proof_world1
        if condition == "ablated":
            return ()
        raise ValueError(f"unknown audit condition: {condition}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "world0": [doc.to_dict() for doc in self.world0],
            "world1": [doc.to_dict() for doc in self.world1],
            "answers": {
                "world0": self.answer_world0,
                "world1": self.answer_world1,
            },
            "proofs": {
                "world0": list(self.proof_world0),
                "world1": list(self.proof_world1),
            },
            "ablate_document_ids": list(self.ablate_document_ids),
            "causal_change_ids": list(self.causal_change_ids),
            "nuisance_change_ids": list(self.nuisance_change_ids),
            "answer_aliases": {
                "world0": list(self.aliases_world0),
                "world1": list(self.aliases_world1),
            },
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AuditDataset:
    """Validated collection of causal audit cases."""

    name: str
    cases: tuple[AuditCase, ...]
    description: str = ""
    abstention_answers: tuple[str, ...] = (
        "INSUFFICIENT",
        "not specified",
        "not provided",
        "cannot determine",
        "cannot be determined",
        "unable to determine",
        "indeterminate",
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "description": self.description,
            "abstention_answers": list(self.abstention_answers),
            "metadata": dict(self.metadata),
            "cases": [case.to_dict() for case in self.cases],
        }


@dataclass(frozen=True)
class TargetRequest:
    """Input given to the RAG target; oracle answers are deliberately absent."""

    question: str
    documents: tuple[Document, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "documents": [doc.to_dict() for doc in self.documents],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TargetResponse:
    """Normalized answer and document-level citations returned by a RAG target."""

    answer: str
    citations: tuple[str, ...]
    raw: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    format_valid: bool = True
    format_error: str | None = None


@dataclass(frozen=True)
class ScoredResponse:
    """One target invocation and every score recomputed from it."""

    condition: str
    answer: str
    citations: tuple[str, ...]
    expected_answer: str | None
    gold_citations: tuple[str, ...]
    correct: bool
    abstained: bool
    citation_complete: bool
    citation_exact: bool
    citation_valid: bool
    format_valid: bool
    latency_ms: float
    error: str | None = None
    raw: Any = None
    proof_citation_recall: float | None = None
    proof_citation_precision: float | None = None

    def to_dict(self) -> dict[str, Any]:
        data = {
            "condition": self.condition,
            "answer": self.answer,
            "citations": list(self.citations),
            "expected_answer": self.expected_answer,
            "gold_citations": list(self.gold_citations),
            "correct": self.correct,
            "abstained": self.abstained,
            "citation_complete": self.citation_complete,
            "citation_exact": self.citation_exact,
            "citation_valid": self.citation_valid,
            "proof_citation_recall": self.proof_citation_recall,
            "proof_citation_precision": self.proof_citation_precision,
            "format_valid": self.format_valid,
            "latency_ms": round(self.latency_ms, 3),
            "error": self.error,
        }
        if self.raw is not None:
            data["raw"] = self.raw
        return data


@dataclass(frozen=True)
class CaseResult:
    """Three scored conditions and the resulting hard joint indicators."""

    case_id: str
    metadata: Mapping[str, Any]
    responses: tuple[ScoredResponse, ...]
    indicators: Mapping[str, bool]

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "metadata": dict(self.metadata),
            "responses": [response.to_dict() for response in self.responses],
            "indicators": dict(self.indicators),
        }


@dataclass(frozen=True)
class MetricResult:
    """A binary metric aggregated across audit cases."""

    rate: float
    count: int
    n: int
    wilson95: tuple[float, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "rate": self.rate,
            "count": self.count,
            "n": self.n,
            "wilson95": list(self.wilson95),
        }


@dataclass(frozen=True)
class MeanMetricResult:
    """A fractional metric macro-averaged across audit cases."""

    mean: float
    total: float
    n: int

    @property
    def rate(self) -> float:
        """Expose a rate alias so CLI thresholds work across metric families."""

        return self.mean

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean,
            "total": self.total,
            "n": self.n,
        }


@dataclass(frozen=True)
class AuditReport:
    """Complete machine-readable output of one audit run."""

    generated_at_utc: str
    framework_version: str
    dataset_name: str
    dataset_fingerprint: str
    target_name: str
    configuration: Mapping[str, Any]
    provenance: Mapping[str, Any]
    metrics: Mapping[str, MetricResult]
    proof_metrics: Mapping[str, MeanMetricResult]
    cases: tuple[CaseResult, ...]

    @property
    def error_count(self) -> int:
        return sum(
            response.error is not None
            for case in self.cases
            for response in case.responses
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "generated_at_utc": self.generated_at_utc,
            "framework_version": self.framework_version,
            "dataset": {
                "name": self.dataset_name,
                "fingerprint_sha256": self.dataset_fingerprint,
                "case_count": len(self.cases),
            },
            "target": {"name": self.target_name},
            "configuration": dict(self.configuration),
            "provenance": dict(self.provenance),
            "metrics": {
                name: metric.to_dict() for name, metric in self.metrics.items()
            },
            "proof_metrics": {
                name: metric.to_dict() for name, metric in self.proof_metrics.items()
            },
            "error_count": self.error_count,
            "cases": [case.to_dict() for case in self.cases],
        }
