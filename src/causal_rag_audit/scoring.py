"""Deterministic scoring for causal evidence audits."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Mapping
from typing import Any

from .models import (
    AuditCase,
    CaseResult,
    MeanMetricResult,
    MetricResult,
    ScoredResponse,
    TargetResponse,
)

PAPER_PROFILE = "paper-v0.1"
STRICT_PROFILE = "strict-exact"
SCORING_PROFILES = (PAPER_PROFILE, STRICT_PROFILE)

AnswerJudge = Callable[
    [str, str, tuple[str, ...], tuple[str, ...]],
    bool,
]
AbstentionJudge = Callable[[str, tuple[str, ...]], bool]

METRIC_NAMES = (
    "world0_accuracy",
    "world1_accuracy",
    "paired_responsiveness",
    "ablation_abstention",
    "ablation_citation_validity",
    "observational_support",
    "coverage_causal_evidence_score",
    "strict_causal_evidence_score",
    "format_validity",
)

PROOF_METRIC_NAMES = (
    "world0_proof_citation_recall",
    "world0_proof_citation_precision",
    "world1_proof_citation_recall",
    "world1_proof_citation_precision",
)

ALL_METRIC_NAMES = (*METRIC_NAMES, *PROOF_METRIC_NAMES)


def canonical_answer(value: str) -> str:
    """Normalize an answer for conservative exact matching."""

    normalized = unicodedata.normalize("NFKC", value).casefold().strip()
    normalized = re.sub(r"[^\w]+", " ", normalized, flags=re.UNICODE)
    return " ".join(normalized.split())


def paper_canonical_answer(value: str) -> str:
    """Match the alphanumeric normalization used by the paper's pilot scorer."""

    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def answer_matches(actual: str, expected: str, aliases: tuple[str, ...] = ()) -> bool:
    """Return whether an answer exactly matches the oracle or a declared alias."""

    candidate = canonical_answer(actual)
    accepted = {
        canonical_answer(expected),
        *(canonical_answer(alias) for alias in aliases),
    }
    return bool(candidate) and candidate in accepted


def paper_answer_matches(
    actual: str,
    expected: str,
    aliases: tuple[str, ...] = (),
    alternatives: tuple[str, ...] = (),
) -> bool:
    """Apply the paper's fixed candidate-in-wrapper matching rule."""

    answer_key = paper_canonical_answer(actual)
    accepted = tuple(
        key
        for key in (
            paper_canonical_answer(expected),
            *(paper_canonical_answer(alias) for alias in aliases),
        )
        if key
    )
    if not any(candidate in answer_key for candidate in accepted):
        return False
    alternative_keys = {
        paper_canonical_answer(alternative)
        for alternative in alternatives
        if paper_canonical_answer(alternative) not in accepted
    }
    return not any(key and key in answer_key for key in alternative_keys)


def is_abstention(actual: str, abstention_answers: tuple[str, ...]) -> bool:
    """Return whether the answer matches a predeclared abstention form."""

    candidate = canonical_answer(actual)
    return candidate in {canonical_answer(answer) for answer in abstention_answers}


def paper_is_abstention(actual: str, abstention_answers: tuple[str, ...]) -> bool:
    """Recognize a predeclared paper-style abstention marker inside a response."""

    candidate = paper_canonical_answer(actual)
    markers = {paper_canonical_answer(answer) for answer in abstention_answers}
    return any(marker and marker in candidate for marker in markers)


def validate_scoring_profile(scoring_profile: str) -> str:
    """Return a known scoring profile or raise a configuration error."""

    if scoring_profile not in SCORING_PROFILES:
        raise ValueError(
            f"unknown scoring profile {scoring_profile!r}; "
            f"choose from {', '.join(SCORING_PROFILES)}"
        )
    return scoring_profile


def score_response(
    case: AuditCase,
    condition: str,
    response: TargetResponse | None,
    abstention_answers: tuple[str, ...],
    latency_ms: float,
    *,
    error: str | None = None,
    include_raw: bool = False,
    scoring_profile: str = PAPER_PROFILE,
    answer_judge: AnswerJudge | None = None,
    abstention_judge: AbstentionJudge | None = None,
) -> ScoredResponse:
    """Recompute every row-level score from one normalized target response."""

    documents = case.documents_for(condition)
    available_ids = {document.id for document in documents}
    expected = case.expected_answer_for(condition)
    proof = case.proof_for(condition)
    if response is None:
        answer = ""
        citations: tuple[str, ...] = ()
        raw: Any = None
        format_valid = False
    else:
        answer = response.answer
        citations = response.citations
        raw = response.raw if include_raw else None
        format_valid = response.format_valid

    profile = validate_scoring_profile(scoring_profile)
    abstained = (
        abstention_judge(answer, abstention_answers)
        if abstention_judge
        else (
            paper_is_abstention(answer, abstention_answers)
            if profile == PAPER_PROFILE
            else is_abstention(answer, abstention_answers)
        )
    )
    if condition == "world0":
        alternatives = (case.answer_world1, *case.aliases_world1)
    elif condition == "world1":
        alternatives = (case.answer_world0, *case.aliases_world0)
    else:
        alternatives = ()
    if condition == "ablated":
        correct = abstained
    elif answer_judge:
        correct = answer_judge(
            answer,
            expected or "",
            case.aliases_for(condition),
            alternatives,
        )
    elif profile == PAPER_PROFILE:
        correct = paper_answer_matches(
            answer,
            expected or "",
            case.aliases_for(condition),
            alternatives,
        )
    else:
        correct = answer_matches(answer, expected or "", case.aliases_for(condition))
    citation_set = set(citations)
    proof_set = set(proof)
    if condition == "ablated":
        proof_recall = None
        proof_precision = None
    else:
        intersection = len(citation_set & proof_set)
        proof_recall = intersection / len(proof_set)
        proof_precision = intersection / len(citation_set) if citation_set else 0.0
    return ScoredResponse(
        condition=condition,
        answer=answer,
        citations=citations,
        expected_answer=expected,
        gold_citations=proof,
        correct=correct,
        abstained=abstained,
        citation_complete=proof_set.issubset(citation_set),
        citation_exact=citation_set == proof_set,
        citation_valid=citation_set.issubset(available_ids),
        format_valid=format_valid,
        latency_ms=latency_ms,
        error=error,
        raw=raw,
        proof_citation_recall=proof_recall,
        proof_citation_precision=proof_precision,
    )


def evaluate_case(case: AuditCase, responses: tuple[ScoredResponse, ...]) -> CaseResult:
    """Calculate observational and hard causal pass indicators for one case."""

    by_condition = {response.condition: response for response in responses}
    if set(by_condition) != {"world0", "world1", "ablated"}:
        raise ValueError("a case result requires world0, world1, and ablated responses")
    world0 = by_condition["world0"]
    world1 = by_condition["world1"]
    ablated = by_condition["ablated"]
    paired = (
        world0.correct
        and world1.correct
        and canonical_answer(world0.answer) != canonical_answer(world1.answer)
    )
    observational = world0.correct and world0.citation_complete
    coverage = (
        paired
        and ablated.abstained
        and ablated.citation_valid
        and world0.citation_complete
        and world1.citation_complete
    )
    strict = (
        paired
        and ablated.abstained
        and not ablated.citations
        and world0.citation_exact
        and world1.citation_exact
    )
    indicators = {
        "world0_accuracy": world0.correct,
        "world1_accuracy": world1.correct,
        "paired_responsiveness": paired,
        "ablation_abstention": ablated.abstained,
        "ablation_citation_validity": ablated.citation_valid,
        "observational_support": observational,
        "coverage_causal_evidence_score": coverage,
        "strict_causal_evidence_score": strict,
        "format_validity": all(response.format_valid for response in responses),
    }
    return CaseResult(
        case_id=case.id,
        metadata=dict(case.metadata),
        responses=responses,
        indicators=indicators,
    )


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""

    if n <= 0:
        raise ValueError("n must be positive")
    proportion = successes / n
    denominator = 1 + z * z / n
    center = (proportion + z * z / (2 * n)) / denominator
    margin = (
        z
        * ((proportion * (1 - proportion) / n + z * z / (4 * n * n)) ** 0.5)
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def summarize(cases: tuple[CaseResult, ...]) -> Mapping[str, MetricResult]:
    """Macro-average every binary metric over audit cases."""

    if not cases:
        raise ValueError("cannot summarize an empty audit")
    metrics: dict[str, MetricResult] = {}
    for name in METRIC_NAMES:
        count = sum(int(case.indicators[name]) for case in cases)
        metrics[name] = MetricResult(
            rate=count / len(cases),
            count=count,
            n=len(cases),
            wilson95=wilson_interval(count, len(cases)),
        )
    return metrics


def summarize_proof_metrics(
    cases: tuple[CaseResult, ...],
) -> Mapping[str, MeanMetricResult]:
    """Macro-average paper-defined proof citation recall and precision by world."""

    if not cases:
        raise ValueError("cannot summarize an empty audit")
    metrics: dict[str, MeanMetricResult] = {}
    for condition in ("world0", "world1"):
        responses = [
            next(
                response
                for response in case.responses
                if response.condition == condition
            )
            for case in cases
        ]
        for suffix, attribute in (
            ("recall", "proof_citation_recall"),
            ("precision", "proof_citation_precision"),
        ):
            values = [getattr(response, attribute) for response in responses]
            if any(value is None for value in values):
                raise ValueError(
                    f"{condition} responses require proof citation {suffix}"
                )
            total = sum(float(value) for value in values)
            metrics[f"{condition}_proof_citation_{suffix}"] = MeanMetricResult(
                mean=total / len(values),
                total=total,
                n=len(values),
            )
    return metrics
