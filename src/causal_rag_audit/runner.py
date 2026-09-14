"""Target execution and report assembly."""

from __future__ import annotations

import hashlib
import inspect
import json
import platform
import re
import sys
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any

from .models import (
    CONDITIONS,
    AuditCase,
    AuditDataset,
    AuditReport,
    CaseResult,
    ScoredResponse,
    TargetRequest,
    TargetResponse,
)
from .scoring import (
    PAPER_PROFILE,
    AbstentionJudge,
    AnswerJudge,
    evaluate_case,
    score_response,
    summarize,
    summarize_proof_metrics,
    validate_scoring_profile,
)
from .version import __version__

Target = Callable[[TargetRequest], TargetResponse | Mapping[str, Any] | str]


class ResponseFormatError(ValueError):
    """Raised when a target response cannot be normalized safely."""


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return repr(value)


def _invalid_text_response(value: Any, reason: str) -> TargetResponse:
    answer = str(value).strip()
    if "INSUFFICIENT" in answer.upper():
        answer = "INSUFFICIENT"
    return TargetResponse(
        answer=answer,
        citations=(),
        raw=_json_safe(value),
        format_valid=False,
        format_error=reason,
    )


def _paper_response(decoded: Mapping[str, Any], raw: Any) -> TargetResponse:
    answer_value = decoded.get("answer", "")
    answer = str(answer_value).strip()
    citations_value = decoded.get("citations", [])
    if not isinstance(citations_value, list):
        return TargetResponse(
            answer=answer,
            citations=(),
            raw=_json_safe(raw),
            format_valid=False,
            format_error="citations must be a JSON array",
        )
    citation_pattern = re.compile(r"D\d+")
    normalized = tuple(
        sorted(
            {
                str(citation).strip().upper()
                for citation in citations_value
                if citation_pattern.fullmatch(str(citation).strip().upper())
            }
        )
    )
    format_valid = (
        set(decoded) == {"answer", "citations"}
        and isinstance(answer_value, str)
        and all(
            isinstance(citation, str)
            and citation_pattern.fullmatch(citation.strip().upper())
            for citation in citations_value
        )
    )
    return TargetResponse(
        answer=answer,
        citations=normalized,
        raw=_json_safe(raw),
        format_valid=format_valid,
        format_error=None
        if format_valid
        else "response does not match paper-v0.1 JSON schema",
    )


def _strict_response(decoded: Mapping[str, Any], raw: Any) -> TargetResponse:
    issues = []
    allowed = {"answer", "citations", "metadata"}
    if not {"answer", "citations"}.issubset(decoded):
        issues.append("answer and citations are required")
    if set(decoded) - allowed:
        issues.append("unexpected response fields")

    answer_value = decoded.get("answer", "")
    answer = (
        answer_value.strip()
        if isinstance(answer_value, str)
        else str(answer_value).strip()
    )
    if not isinstance(answer_value, str) or not answer:
        issues.append("answer must be a non-empty string")

    citations_value = decoded.get("citations", [])
    if not isinstance(citations_value, (list, tuple)):
        issues.append("citations must be an array")
        citations_value = []
    valid_citations = [
        citation.strip()
        for citation in citations_value
        if isinstance(citation, str) and citation.strip()
    ]
    if len(valid_citations) != len(citations_value):
        issues.append("citations must contain non-empty strings")
    if len(valid_citations) != len(set(valid_citations)):
        issues.append("duplicate citations were deduplicated")
    normalized = tuple(dict.fromkeys(valid_citations))

    metadata_value = decoded.get("metadata", {})
    if not isinstance(metadata_value, Mapping):
        issues.append("metadata must be an object")
        metadata_value = {}
    return TargetResponse(
        answer=answer,
        citations=normalized,
        raw=_json_safe(raw),
        metadata=dict(metadata_value),
        format_valid=not issues,
        format_error="; ".join(dict.fromkeys(issues)) or None,
    )


def coerce_response(
    value: TargetResponse | Mapping[str, Any] | str,
    *,
    scoring_profile: str = PAPER_PROFILE,
) -> TargetResponse:
    """Normalize output while preserving scoreable fields from malformed responses."""

    profile = validate_scoring_profile(scoring_profile)
    if isinstance(value, TargetResponse):
        decoded: Mapping[str, Any] = {
            "answer": value.answer,
            "citations": list(value.citations),
        }
        if value.metadata:
            decoded = {**decoded, "metadata": dict(value.metadata)}
        response = (
            _paper_response(decoded, value.raw)
            if profile == PAPER_PROFILE
            else _strict_response(decoded, value.raw)
        )
        if not value.format_valid:
            return TargetResponse(
                answer=response.answer,
                citations=response.citations,
                raw=response.raw,
                metadata=response.metadata,
                format_valid=False,
                format_error=value.format_error
                or response.format_error
                or "invalid response",
            )
        return response

    raw: Any = value
    if isinstance(value, str):
        try:
            decoded_value = json.loads(value.strip())
        except json.JSONDecodeError as exc:
            return _invalid_text_response(value, f"invalid JSON text: {exc.msg}")
    else:
        decoded_value = value
    if not isinstance(decoded_value, Mapping):
        return _invalid_text_response(
            value,
            "target response must be an object or JSON object",
        )
    if profile == PAPER_PROFILE:
        return _paper_response(decoded_value, raw)
    return _strict_response(decoded_value, raw)


def dataset_fingerprint(dataset: AuditDataset) -> str:
    """Hash the canonical validated dataset representation."""

    payload = json.dumps(
        dataset.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class AuditRunner:
    """Execute a validated dataset against any synchronous RAG target."""

    def __init__(
        self,
        target: Target,
        *,
        target_name: str | None = None,
        max_workers: int = 1,
        include_raw: bool = False,
        target_metadata: Mapping[str, Any] | None = None,
        scoring_profile: str = PAPER_PROFILE,
        answer_judge: AnswerJudge | None = None,
        abstention_judge: AbstentionJudge | None = None,
    ) -> None:
        if not callable(target):
            raise TypeError("target must be callable")
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        self.target = target
        inferred_name = getattr(target, "__name__", target.__class__.__name__)
        self.target_name = str(target_name or inferred_name)
        self.max_workers = max_workers
        self.include_raw = include_raw
        self.target_metadata = dict(target_metadata or {})
        self.scoring_profile = validate_scoring_profile(scoring_profile)
        self.answer_judge = answer_judge
        self.abstention_judge = abstention_judge

    def _invoke(
        self,
        case: AuditCase,
        condition: str,
        abstention_answers: tuple[str, ...],
    ) -> ScoredResponse:
        request = TargetRequest(
            question=case.question,
            documents=case.documents_for(condition),
            metadata=dict(case.metadata),
        )
        started = time.perf_counter()
        response = None
        error = None
        try:
            returned = self.target(request)
            if inspect.isawaitable(returned):
                close = getattr(returned, "close", None)
                if callable(close):
                    close()
                raise TypeError(
                    "async targets are not supported directly; provide a synchronous adapter"
                )
            response = coerce_response(returned, scoring_profile=self.scoring_profile)
            if not response.format_valid:
                error = f"ResponseFormatError: {response.format_error}"
        # The target is third-party application code. Every ordinary target failure
        # is retained as audit evidence so one bad call cannot erase the full run.
        except Exception as exc:  # noqa: BLE001
            error = f"{exc.__class__.__name__}: {exc}"
        latency_ms = (time.perf_counter() - started) * 1000
        return score_response(
            case,
            condition,
            response,
            abstention_answers,
            latency_ms,
            error=error,
            include_raw=self.include_raw,
            scoring_profile=self.scoring_profile,
            answer_judge=self.answer_judge,
            abstention_judge=self.abstention_judge,
        )

    def run(self, dataset: AuditDataset) -> AuditReport:
        """Run three evidence conditions per case and return an immutable report."""

        jobs = [
            (case_index, condition_index, case, condition)
            for case_index, case in enumerate(dataset.cases)
            for condition_index, condition in enumerate(CONDITIONS)
        ]
        scored: dict[tuple[int, int], Any] = {}
        if self.max_workers == 1:
            for case_index, condition_index, case, condition in jobs:
                scored[(case_index, condition_index)] = self._invoke(
                    case, condition, dataset.abstention_answers
                )
        else:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(
                        self._invoke, case, condition, dataset.abstention_answers
                    ): (case_index, condition_index)
                    for case_index, condition_index, case, condition in jobs
                }
                for future in as_completed(futures):
                    scored[futures[future]] = future.result()

        case_results: list[CaseResult] = []
        for case_index, case in enumerate(dataset.cases):
            responses = tuple(
                scored[(case_index, condition_index)]
                for condition_index in range(len(CONDITIONS))
            )
            case_results.append(evaluate_case(case, responses))
        finalized = tuple(case_results)
        return AuditReport(
            generated_at_utc=datetime.now(timezone.utc).isoformat(),
            framework_version=__version__,
            dataset_name=dataset.name,
            dataset_fingerprint=dataset_fingerprint(dataset),
            target_name=self.target_name,
            configuration={
                "conditions": list(CONDITIONS),
                "calls": len(jobs),
                "max_workers": self.max_workers,
                "include_raw": self.include_raw,
                "abstention_answers": list(dataset.abstention_answers),
                "scoring_profile": self.scoring_profile,
                "answer_judge": getattr(self.answer_judge, "__name__", None),
                "abstention_judge": getattr(self.abstention_judge, "__name__", None),
            },
            provenance={
                "python": platform.python_version(),
                "platform": platform.platform(),
                "machine": platform.machine(),
                "target_metadata": self.target_metadata,
                "executable": sys.executable,
            },
            metrics=summarize(finalized),
            proof_metrics=summarize_proof_metrics(finalized),
            cases=finalized,
        )
