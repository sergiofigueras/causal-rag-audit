"""Target execution and report assembly."""

from __future__ import annotations

import hashlib
import inspect
import json
import platform
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
from .scoring import evaluate_case, score_response, summarize
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


def coerce_response(value: TargetResponse | Mapping[str, Any] | str) -> TargetResponse:
    """Normalize a dataclass, mapping, or JSON string into a target response."""

    raw: Any = value
    if isinstance(value, TargetResponse):
        response = value
    else:
        if isinstance(value, str):
            try:
                decoded = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ResponseFormatError(
                    f"target returned invalid JSON text: {exc.msg}"
                ) from exc
        else:
            decoded = value
        if not isinstance(decoded, Mapping):
            raise ResponseFormatError(
                "target response must be an object or JSON object"
            )
        answer = decoded.get("answer")
        citations = decoded.get("citations")
        if not isinstance(answer, str) or not answer.strip():
            raise ResponseFormatError(
                "target response answer must be a non-empty string"
            )
        if not isinstance(citations, (list, tuple)) or any(
            not isinstance(citation, str) or not citation.strip()
            for citation in citations
        ):
            raise ResponseFormatError(
                "target response citations must be an array of strings"
            )
        normalized_citations = tuple(citation.strip() for citation in citations)
        if len(normalized_citations) != len(set(normalized_citations)):
            raise ResponseFormatError(
                "target response citations must not contain duplicates"
            )
        metadata = decoded.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ResponseFormatError("target response metadata must be an object")
        response = TargetResponse(
            answer=answer.strip(),
            citations=normalized_citations,
            raw=raw,
            metadata=dict(metadata),
        )
    if not response.answer.strip():
        raise ResponseFormatError("target response answer must be a non-empty string")
    if any(
        not isinstance(citation, str) or not citation.strip()
        for citation in response.citations
    ):
        raise ResponseFormatError("target response citations must be non-empty strings")
    if len(response.citations) != len(set(response.citations)):
        raise ResponseFormatError(
            "target response citations must not contain duplicates"
        )
    return TargetResponse(
        answer=response.answer.strip(),
        citations=tuple(citation.strip() for citation in response.citations),
        raw=_json_safe(response.raw),
        metadata=dict(response.metadata),
    )


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
            response = coerce_response(returned)
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
            },
            provenance={
                "python": platform.python_version(),
                "platform": platform.platform(),
                "machine": platform.machine(),
                "target_metadata": self.target_metadata,
                "executable": sys.executable,
            },
            metrics=summarize(finalized),
            cases=finalized,
        )
