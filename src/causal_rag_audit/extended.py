"""CEA-Extended dataset schema v2 and scoring.

Structural validation is deliberately separate from human review of whether the
declared answers and proof sets actually follow from the evidence.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .models import CaseResult, Document, ScoredResponse, TargetResponse
from .scoring import (
    PAPER_PROFILE,
    AbstentionJudge,
    AnswerJudge,
    answer_matches,
    canonical_answer,
    is_abstention,
    paper_answer_matches,
    paper_is_abstention,
    validate_scoring_profile,
)
from .validation import (
    DatasetValidationError,
    _documents,
    _known_keys,
    _mapping,
    _string,
    _string_tuple,
)

FAMILIES = (
    "baseline",
    "counterfactual",
    "nuisance",
    "ablation",
    "redundancy",
    "conflict",
)
REQUIRED_FAMILIES = frozenset(FAMILIES)
V2_METRIC_NAMES = (
    "extended_coverage_causal_evidence_score",
    "extended_strict_causal_evidence_score",
    "extended_format_validity",
    *(f"{family}_coverage_pass" for family in FAMILIES),
    *(f"{family}_strict_pass" for family in FAMILIES),
)
V2_PROOF_METRIC_NAMES = (
    "extended_answerable_proof_citation_recall",
    "extended_answerable_proof_citation_precision",
    *(
        f"{family}_proof_citation_{measure}"
        for family in FAMILIES
        for measure in ("recall", "precision")
    ),
)
V2_ROOT_KEYS = {
    "schema_version",
    "name",
    "description",
    "abstention_answers",
    "metadata",
    "cases",
}
V2_CASE_KEYS = {"id", "question", "conditions", "metadata"}
V2_CONDITION_KEYS = {
    "id",
    "family",
    "documents",
    "expected_answer",
    "answer_aliases",
    "proof_sets",
    "changed_document_ids",
    "removed_document_ids",
    "rationale",
    "resolution_policy",
    "authority_document_ids",
}


@dataclass(frozen=True)
class ExtendedCondition:
    id: str
    family: str
    documents: tuple[Document, ...]
    expected_answer: str | None
    proof_sets: tuple[tuple[str, ...], ...]
    changed_document_ids: tuple[str, ...] = ()
    removed_document_ids: tuple[str, ...] = ()
    rationale: str = ""
    answer_aliases: tuple[str, ...] = ()
    resolution_policy: str | None = None
    authority_document_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.id,
            "family": self.family,
            "documents": [document.to_dict() for document in self.documents],
            "expected_answer": self.expected_answer,
            "answer_aliases": list(self.answer_aliases),
            "proof_sets": [list(proof) for proof in self.proof_sets],
            "changed_document_ids": list(self.changed_document_ids),
            "removed_document_ids": list(self.removed_document_ids),
            "rationale": self.rationale,
        }
        if self.resolution_policy is not None:
            data["resolution_policy"] = self.resolution_policy
            data["authority_document_ids"] = list(self.authority_document_ids)
        return data


@dataclass(frozen=True)
class ExtendedCase:
    id: str
    question: str
    conditions: tuple[ExtendedCondition, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def condition(self, condition_id: str) -> ExtendedCondition:
        return next(
            condition for condition in self.conditions if condition.id == condition_id
        )

    def documents_for(self, condition_id: str) -> tuple[Document, ...]:
        return self.condition(condition_id).documents

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "conditions": [condition.to_dict() for condition in self.conditions],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ExtendedDataset:
    name: str
    cases: tuple[ExtendedCase, ...]
    description: str = ""
    abstention_answers: tuple[str, ...] = ("INSUFFICIENT",)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "name": self.name,
            "description": self.description,
            "abstention_answers": list(self.abstention_answers),
            "metadata": dict(self.metadata),
            "cases": [case.to_dict() for case in self.cases],
        }


def _proof_sets(
    raw: Any, path: str, available_ids: set[str], abstain: bool
) -> tuple[tuple[str, ...], ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise DatasetValidationError(f"{path} must be an array of proof arrays")
    proofs = tuple(
        _string_tuple(item, f"{path}[{index}]", allow_empty=False)
        for index, item in enumerate(raw)
    )
    if bool(proofs) == abstain:
        raise DatasetValidationError(
            f"{path} must be empty exactly when the oracle abstains"
        )
    sets = [set(proof) for proof in proofs]
    if any(not proof.issubset(available_ids) for proof in sets):
        raise DatasetValidationError(f"{path} references an unavailable document")
    if any(
        left <= right
        for i, left in enumerate(sets)
        for j, right in enumerate(sets)
        if i != j
    ):
        raise DatasetValidationError(
            f"{path} must be a minimal antichain without duplicates"
        )
    return proofs


def _parse_condition(raw: Any, path: str) -> ExtendedCondition:
    value = _mapping(raw, path)
    _known_keys(value, V2_CONDITION_KEYS, path)
    family = _string(value.get("family"), f"{path}.family")
    if family not in REQUIRED_FAMILIES:
        raise DatasetValidationError(f"{path}.family must be one of {FAMILIES}")
    documents = _documents(value.get("documents"), f"{path}.documents")
    ids = {document.id for document in documents}
    answer = value.get("expected_answer")
    if answer is not None:
        answer = _string(answer, f"{path}.expected_answer")
    aliases = _string_tuple(
        value.get("answer_aliases", []), f"{path}.answer_aliases", allow_empty=True
    )
    if answer is None and aliases:
        raise DatasetValidationError(f"{path} abstention cannot have answer aliases")
    if answer is not None and canonical_answer(answer) in {
        canonical_answer(a) for a in aliases
    }:
        raise DatasetValidationError(f"{path}.answer_aliases duplicates the answer")
    proofs = _proof_sets(
        value.get("proof_sets"), f"{path}.proof_sets", ids, answer is None
    )
    changed = _string_tuple(
        value.get("changed_document_ids", []),
        f"{path}.changed_document_ids",
        allow_empty=True,
    )
    removed = _string_tuple(
        value.get("removed_document_ids", []),
        f"{path}.removed_document_ids",
        allow_empty=True,
    )
    policy = value.get("resolution_policy")
    authority = _string_tuple(
        value.get("authority_document_ids", []),
        f"{path}.authority_document_ids",
        allow_empty=True,
    )
    if family == "conflict":
        if policy not in ("authority", "abstain"):
            raise DatasetValidationError(
                f"{path}.resolution_policy must be authority or abstain"
            )
        if (policy == "authority") != (answer is not None):
            raise DatasetValidationError(
                f"{path} conflict answer must agree with resolution_policy"
            )
        if policy == "authority" and (
            not authority or not set(authority).issubset(ids)
        ):
            raise DatasetValidationError(
                f"{path}.authority_document_ids must identify available documents"
            )
        if policy == "authority" and not any(
            set(authority).issubset(set(p)) for p in proofs
        ):
            raise DatasetValidationError(
                f"{path} an authoritative proof must cite the authority IDs"
            )
        if policy == "abstain" and authority:
            raise DatasetValidationError(
                f"{path} unresolved conflict must not declare authority IDs"
            )
    elif policy is not None or authority:
        raise DatasetValidationError(
            f"{path} resolution policy is only valid for conflict"
        )
    return ExtendedCondition(
        id=_string(value.get("id"), f"{path}.id"),
        family=family,
        documents=documents,
        expected_answer=answer,
        proof_sets=proofs,
        changed_document_ids=changed,
        removed_document_ids=removed,
        rationale=_string(value.get("rationale"), f"{path}.rationale"),
        answer_aliases=aliases,
        resolution_policy=policy,
        authority_document_ids=authority,
    )


def _parse_case(raw: Any, index: int) -> ExtendedCase:
    path = f"cases[{index}]"
    value = _mapping(raw, path)
    _known_keys(value, V2_CASE_KEYS, path)
    raw_conditions = value.get("conditions")
    if not isinstance(raw_conditions, Sequence) or isinstance(
        raw_conditions, (str, bytes)
    ):
        raise DatasetValidationError(f"{path}.conditions must be an array")
    conditions = tuple(
        _parse_condition(item, f"{path}.conditions[{i}]")
        for i, item in enumerate(raw_conditions)
    )
    ids = [condition.id for condition in conditions]
    if len(ids) != len(set(ids)):
        raise DatasetValidationError(f"{path} condition IDs must be unique")
    family_counts = Counter(condition.family for condition in conditions)
    if (
        family_counts["baseline"] != 1
        or family_counts["counterfactual"] < 2
        or any(
            family_counts[family] < 1
            for family in ("nuisance", "ablation", "redundancy", "conflict")
        )
    ):
        raise DatasetValidationError(
            f"{path} requires one baseline, at least two counterfactuals, and each other intervention family"
        )
    baseline = next(
        condition for condition in conditions if condition.family == "baseline"
    )
    if baseline.id != "baseline" or baseline.expected_answer is None:
        raise DatasetValidationError(
            f"{path} baseline must have id 'baseline' and an answer"
        )
    if len(baseline.proof_sets) < 2:
        raise DatasetValidationError(
            f"{path} redundancy requires at least two baseline proof sets"
        )
    baseline_by_id = {document.id: document for document in baseline.documents}
    baseline_ids = set(baseline_by_id)
    if baseline.changed_document_ids or baseline.removed_document_ids:
        raise DatasetValidationError(f"{path} baseline must not declare edits")
    counterfactual_worlds: set[str] = set()
    baseline_answers = {
        canonical_answer(baseline.expected_answer),
        *(canonical_answer(alias) for alias in baseline.answer_aliases),
    }
    for condition in conditions:
        if condition is baseline:
            continue
        condition_path = f"{path}.conditions[{condition.id}]"
        by_id = {document.id: document for document in condition.documents}
        available = set(by_id)
        if not available.issubset(baseline_ids):
            raise DatasetValidationError(
                f"{condition_path} adds document IDs; IDs must be stable"
            )
        actual_removed = baseline_ids - available
        actual_changed = {
            document_id
            for document_id in available
            if by_id[document_id] != baseline_by_id[document_id]
        }
        if actual_removed != set(
            condition.removed_document_ids
        ) or actual_changed != set(condition.changed_document_ids):
            raise DatasetValidationError(
                f"{condition_path} declared edits do not match baseline diff"
            )
        if (
            condition.family in ("counterfactual", "nuisance", "conflict")
            and actual_removed
        ):
            raise DatasetValidationError(
                f"{condition_path} family cannot remove documents"
            )
        if condition.family in ("ablation", "redundancy") and (
            actual_changed or not actual_removed
        ):
            raise DatasetValidationError(
                f"{condition_path} requires removal-only intervention"
            )
        if condition.family == "counterfactual":
            if not actual_changed or condition.expected_answer is None:
                raise DatasetValidationError(
                    f"{condition_path} requires an edited, answerable world"
                )
            answer_forms = {
                canonical_answer(condition.expected_answer),
                *(canonical_answer(alias) for alias in condition.answer_aliases),
            }
            if answer_forms & baseline_answers:
                raise DatasetValidationError(
                    f"{condition_path} must change the baseline answer"
                )
            world_signature = json.dumps(
                [doc.to_dict() for doc in condition.documents],
                sort_keys=True,
                ensure_ascii=False,
            )
            if world_signature in counterfactual_worlds:
                raise DatasetValidationError(
                    f"{condition_path} duplicates another counterfactual world"
                )
            counterfactual_worlds.add(world_signature)
            if not any(
                set(condition.changed_document_ids) & set(proof)
                for proof in (*baseline.proof_sets, *condition.proof_sets)
            ):
                raise DatasetValidationError(
                    f"{condition_path} edited IDs must participate in a proof"
                )
        elif condition.family == "nuisance":
            if canonical_answer(condition.expected_answer or "") != canonical_answer(
                baseline.expected_answer
            ):
                raise DatasetValidationError(
                    f"{condition_path} nuisance must preserve the answer"
                )
            if {frozenset(p) for p in condition.proof_sets} != {
                frozenset(p) for p in baseline.proof_sets
            }:
                raise DatasetValidationError(
                    f"{condition_path} nuisance must preserve all proof sets"
                )
            if not actual_changed and tuple(by_id) == tuple(baseline_by_id):
                raise DatasetValidationError(
                    f"{condition_path} nuisance requires a metadata, text, or order change"
                )
        elif condition.family == "ablation":
            if condition.expected_answer is not None or not all(
                set(proof) & actual_removed for proof in baseline.proof_sets
            ):
                raise DatasetValidationError(
                    f"{condition_path} ablation must hit every baseline proof and require abstention"
                )
        elif condition.family == "redundancy":
            remaining = {
                frozenset(proof)
                for proof in baseline.proof_sets
                if set(proof).issubset(available)
            }
            if (
                not remaining
                or len(remaining) == len(baseline.proof_sets)
                or {frozenset(proof) for proof in condition.proof_sets} != remaining
                or canonical_answer(condition.expected_answer or "")
                != canonical_answer(baseline.expected_answer)
            ):
                raise DatasetValidationError(
                    f"{condition_path} redundancy must remove one proof path and retain all surviving paths"
                )
        elif condition.family == "conflict" and not actual_changed:
            raise DatasetValidationError(
                f"{condition_path} conflict requires an edited document"
            )
    metadata = _mapping(value.get("metadata", {}), f"{path}.metadata")
    return ExtendedCase(
        id=_string(value.get("id"), f"{path}.id"),
        question=_string(value.get("question"), f"{path}.question"),
        conditions=conditions,
        metadata=dict(metadata),
    )


def dataset_from_mapping(raw: Mapping[str, Any]) -> ExtendedDataset:
    value = _mapping(raw, "dataset")
    _known_keys(value, V2_ROOT_KEYS, "dataset")
    if value.get("schema_version") != 2:
        raise DatasetValidationError("dataset.schema_version must equal 2")
    raw_cases = value.get("cases")
    if (
        not isinstance(raw_cases, Sequence)
        or isinstance(raw_cases, (str, bytes))
        or not raw_cases
    ):
        raise DatasetValidationError("dataset.cases must be a non-empty array")
    cases = tuple(_parse_case(item, index) for index, item in enumerate(raw_cases))
    if len({case.id for case in cases}) != len(cases):
        raise DatasetValidationError("dataset case IDs must be unique")
    abstentions = _string_tuple(
        value.get("abstention_answers", ["INSUFFICIENT"]),
        "dataset.abstention_answers",
        allow_empty=False,
    )
    if len({canonical_answer(answer) for answer in abstentions}) != len(abstentions):
        raise DatasetValidationError(
            "dataset.abstention_answers must be unique after normalization"
        )
    if any(
        condition.expected_answer is not None
        and canonical_answer(condition.expected_answer)
        in {canonical_answer(a) for a in abstentions}
        for case in cases
        for condition in case.conditions
    ):
        raise DatasetValidationError(
            "oracle answers must not equal an abstention marker"
        )
    metadata = _mapping(value.get("metadata", {}), "dataset.metadata")
    description = value.get("description", "")
    if not isinstance(description, str):
        raise DatasetValidationError("dataset.description must be a string")
    return ExtendedDataset(
        name=_string(value.get("name"), "dataset.name"),
        cases=cases,
        description=description.strip(),
        abstention_answers=abstentions,
        metadata=dict(metadata),
    )


def score_response(
    case: ExtendedCase,
    condition_id: str,
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
    condition = case.condition(condition_id)
    profile = validate_scoring_profile(scoring_profile)
    answer = response.answer if response else ""
    citations = response.citations if response else ()
    abstained = (
        abstention_judge(answer, abstention_answers)
        if abstention_judge
        else paper_is_abstention(answer, abstention_answers)
        if profile == PAPER_PROFILE
        else is_abstention(answer, abstention_answers)
    )
    if condition.expected_answer is None:
        correct = abstained
    elif answer_judge:
        accepted = {
            canonical_answer(condition.expected_answer),
            *(canonical_answer(alias) for alias in condition.answer_aliases),
        }
        alternatives = tuple(
            other.expected_answer
            for other in case.conditions
            if other.expected_answer is not None
            and other.id != condition.id
            and canonical_answer(other.expected_answer) not in accepted
        )
        correct = answer_judge(
            answer, condition.expected_answer, condition.answer_aliases, alternatives
        )
    elif profile == PAPER_PROFILE:
        accepted = {
            canonical_answer(condition.expected_answer),
            *(canonical_answer(alias) for alias in condition.answer_aliases),
        }
        alternatives = tuple(
            other.expected_answer
            for other in case.conditions
            if other.expected_answer is not None
            and other.id != condition.id
            and canonical_answer(other.expected_answer) not in accepted
        )
        correct = paper_answer_matches(
            answer, condition.expected_answer, condition.answer_aliases, alternatives
        )
    else:
        correct = answer_matches(
            answer, condition.expected_answer, condition.answer_aliases
        )
    citation_set = set(citations)
    available_ids = {document.id for document in condition.documents}
    proof_sets = tuple(set(proof) for proof in condition.proof_sets)
    recall: float | None
    precision: float | None
    if proof_sets:
        best = max(
            proof_sets,
            key=lambda proof: (
                len(proof & citation_set) / len(proof),
                len(proof & citation_set) / len(citation_set) if citation_set else 0,
            ),
        )
        intersection = len(best & citation_set)
        recall = intersection / len(best)
        precision = intersection / len(citation_set) if citation_set else 0.0
    else:
        best = set()
        recall = precision = None
    return ScoredResponse(
        condition=condition_id,
        answer=answer,
        citations=citations,
        expected_answer=condition.expected_answer,
        gold_citations=tuple(sorted(best)),
        correct=correct,
        abstained=abstained,
        citation_complete=any(proof.issubset(citation_set) for proof in proof_sets)
        if proof_sets
        else True,
        citation_exact=any(proof == citation_set for proof in proof_sets)
        if proof_sets
        else not citations,
        citation_valid=citation_set.issubset(available_ids),
        format_valid=response.format_valid if response else False,
        latency_ms=latency_ms,
        error=error,
        raw=response.raw if response and include_raw else None,
        proof_citation_recall=recall,
        proof_citation_precision=precision,
    )


def evaluate_case(
    case: ExtendedCase, responses: tuple[ScoredResponse, ...]
) -> CaseResult:
    by_id = {response.condition: response for response in responses}
    if len(responses) != len(case.conditions) or set(by_id) != {
        condition.id for condition in case.conditions
    }:
        raise ValueError(
            "extended case requires exactly one response for each declared condition"
        )
    per_condition: dict[str, dict[str, Any]] = {}
    indicators: dict[str, bool] = {}
    for condition in case.conditions:
        response = by_id[condition.id]
        common = (
            response.correct
            and response.citation_valid
            and response.format_valid
            and response.error is None
        )
        coverage = common and response.citation_complete
        strict = common and response.citation_exact
        citation_set = set(response.citations)
        matched = next(
            (
                list(proof)
                for proof in condition.proof_sets
                if (
                    set(proof) == citation_set
                    if strict
                    else set(proof).issubset(citation_set)
                )
            ),
            None,
        )
        per_condition[condition.id] = {
            "family": condition.family,
            "coverage": coverage,
            "strict": strict,
            "proof_sets": [list(proof) for proof in condition.proof_sets],
            "matched_proof_set": matched,
        }
    indicators["extended_coverage_causal_evidence_score"] = all(
        bool(row["coverage"]) for row in per_condition.values()
    )
    indicators["extended_strict_causal_evidence_score"] = all(
        bool(row["strict"]) for row in per_condition.values()
    )
    indicators["extended_format_validity"] = all(
        response.format_valid for response in responses
    )
    for family in FAMILIES:
        rows = [row for row in per_condition.values() if row["family"] == family]
        indicators[f"{family}_coverage_pass"] = all(
            bool(row["coverage"]) for row in rows
        )
        indicators[f"{family}_strict_pass"] = all(bool(row["strict"]) for row in rows)
    metadata = dict(case.metadata)
    metadata["condition_diagnostics"] = per_condition
    metadata["condition_coverage"] = dict(
        Counter(condition.family for condition in case.conditions)
    )
    return CaseResult(
        case_id=case.id, metadata=metadata, responses=responses, indicators=indicators
    )
