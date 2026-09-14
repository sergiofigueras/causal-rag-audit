"""Load and validate versioned causal-audit datasets."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .models import AuditCase, AuditDataset, Document
from .scoring import canonical_answer

ROOT_KEYS = {
    "schema_version",
    "name",
    "description",
    "abstention_answers",
    "metadata",
    "cases",
}
CASE_KEYS = {
    "id",
    "question",
    "world0",
    "world1",
    "answers",
    "proofs",
    "ablate_document_ids",
    "causal_change_ids",
    "nuisance_change_ids",
    "answer_aliases",
    "metadata",
}
DOCUMENT_KEYS = {"id", "text", "metadata"}


class DatasetValidationError(ValueError):
    """Raised when an audit dataset violates the versioned schema or protocol."""


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DatasetValidationError(f"{path} must be an object")
    return value


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetValidationError(f"{path} must be a non-empty string")
    return value.strip()


def _string_tuple(value: Any, path: str, *, allow_empty: bool) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise DatasetValidationError(f"{path} must be an array of strings")
    values = tuple(
        _string(item, f"{path}[{index}]") for index, item in enumerate(value)
    )
    if not allow_empty and not values:
        raise DatasetValidationError(f"{path} must not be empty")
    if len(values) != len(set(values)):
        raise DatasetValidationError(f"{path} must not contain duplicates")
    return values


def _known_keys(value: Mapping[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise DatasetValidationError(f"{path} has unknown fields: {', '.join(unknown)}")


def _documents(value: Any, path: str) -> tuple[Document, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise DatasetValidationError(f"{path} must be a non-empty document array")
    documents = []
    for index, raw in enumerate(value):
        item_path = f"{path}[{index}]"
        document = _mapping(raw, item_path)
        _known_keys(document, DOCUMENT_KEYS, item_path)
        metadata = _mapping(document.get("metadata", {}), f"{item_path}.metadata")
        documents.append(
            Document(
                id=_string(document.get("id"), f"{item_path}.id"),
                text=_string(document.get("text"), f"{item_path}.text"),
                metadata=dict(metadata),
            )
        )
    ids = [document.id for document in documents]
    if len(ids) != len(set(ids)):
        raise DatasetValidationError(f"{path} document IDs must be unique")
    return tuple(documents)


def _parse_case(raw: Any, index: int) -> AuditCase:
    path = f"cases[{index}]"
    value = _mapping(raw, path)
    _known_keys(value, CASE_KEYS, path)
    world0 = _documents(value.get("world0"), f"{path}.world0")
    world1 = _documents(value.get("world1"), f"{path}.world1")
    ids0 = {document.id for document in world0}
    ids1 = {document.id for document in world1}
    if ids0 != ids1:
        raise DatasetValidationError(
            f"{path} worlds must contain the same document IDs; use text interventions"
        )

    answers = _mapping(value.get("answers"), f"{path}.answers")
    _known_keys(answers, {"world0", "world1"}, f"{path}.answers")
    answer0 = _string(answers.get("world0"), f"{path}.answers.world0")
    answer1 = _string(answers.get("world1"), f"{path}.answers.world1")
    if canonical_answer(answer0) == canonical_answer(answer1):
        raise DatasetValidationError(f"{path} oracle answers must differ")

    proofs = _mapping(value.get("proofs"), f"{path}.proofs")
    _known_keys(proofs, {"world0", "world1"}, f"{path}.proofs")
    proof0 = _string_tuple(
        proofs.get("world0"), f"{path}.proofs.world0", allow_empty=False
    )
    proof1 = _string_tuple(
        proofs.get("world1"), f"{path}.proofs.world1", allow_empty=False
    )
    if not set(proof0).issubset(ids0):
        raise DatasetValidationError(
            f"{path}.proofs.world0 references an unknown document"
        )
    if not set(proof1).issubset(ids1):
        raise DatasetValidationError(
            f"{path}.proofs.world1 references an unknown document"
        )

    ablate = _string_tuple(
        value.get("ablate_document_ids"),
        f"{path}.ablate_document_ids",
        allow_empty=False,
    )
    if not set(ablate).issubset(set(proof0)):
        raise DatasetValidationError(
            f"{path}.ablate_document_ids must remove world0 proof documents"
        )

    causal = _string_tuple(
        value.get("causal_change_ids"),
        f"{path}.causal_change_ids",
        allow_empty=False,
    )
    nuisance = _string_tuple(
        value.get("nuisance_change_ids", []),
        f"{path}.nuisance_change_ids",
        allow_empty=True,
    )
    if set(causal) & set(nuisance):
        raise DatasetValidationError(
            f"{path} causal and nuisance changes must be disjoint"
        )
    if not set(causal).issubset(ids0) or not set(nuisance).issubset(ids0):
        raise DatasetValidationError(
            f"{path} change declarations reference unknown documents"
        )

    by_id0 = {document.id: document for document in world0}
    by_id1 = {document.id: document for document in world1}
    changed = {
        document_id
        for document_id in ids0
        if by_id0[document_id].text != by_id1[document_id].text
        or dict(by_id0[document_id].metadata) != dict(by_id1[document_id].metadata)
    }
    declared = set(causal) | set(nuisance)
    if changed != declared:
        raise DatasetValidationError(
            f"{path} changed documents {sorted(changed)} do not match declarations "
            f"{sorted(declared)}"
        )
    if not set(causal).issubset(set(proof0) | set(proof1)):
        raise DatasetValidationError(
            f"{path}.causal_change_ids must participate in at least one oracle proof"
        )
    if set(nuisance) & (set(proof0) | set(proof1)):
        raise DatasetValidationError(
            f"{path}.nuisance_change_ids cannot participate in an oracle proof"
        )

    aliases = _mapping(value.get("answer_aliases", {}), f"{path}.answer_aliases")
    _known_keys(aliases, {"world0", "world1"}, f"{path}.answer_aliases")
    aliases0 = _string_tuple(
        aliases.get("world0", []), f"{path}.answer_aliases.world0", allow_empty=True
    )
    aliases1 = _string_tuple(
        aliases.get("world1", []), f"{path}.answer_aliases.world1", allow_empty=True
    )
    normalized0 = {
        canonical_answer(answer0),
        *(canonical_answer(alias) for alias in aliases0),
    }
    normalized1 = {
        canonical_answer(answer1),
        *(canonical_answer(alias) for alias in aliases1),
    }
    if normalized0 & normalized1:
        raise DatasetValidationError(f"{path} answer aliases overlap across worlds")

    metadata = _mapping(value.get("metadata", {}), f"{path}.metadata")
    return AuditCase(
        id=_string(value.get("id"), f"{path}.id"),
        question=_string(value.get("question"), f"{path}.question"),
        world0=world0,
        world1=world1,
        answer_world0=answer0,
        answer_world1=answer1,
        proof_world0=proof0,
        proof_world1=proof1,
        ablate_document_ids=ablate,
        causal_change_ids=causal,
        nuisance_change_ids=nuisance,
        aliases_world0=aliases0,
        aliases_world1=aliases1,
        metadata=dict(metadata),
    )


def dataset_from_mapping(raw: Mapping[str, Any]) -> AuditDataset:
    """Validate and construct an :class:`AuditDataset` from decoded JSON."""

    value = _mapping(raw, "dataset")
    _known_keys(value, ROOT_KEYS, "dataset")
    if value.get("schema_version") != 1:
        raise DatasetValidationError("dataset.schema_version must equal 1")
    cases_raw = value.get("cases")
    if not isinstance(cases_raw, Sequence) or isinstance(cases_raw, (str, bytes)):
        raise DatasetValidationError("dataset.cases must be an array")
    if not cases_raw:
        raise DatasetValidationError("dataset.cases must not be empty")
    cases = tuple(_parse_case(case, index) for index, case in enumerate(cases_raw))
    case_ids = [case.id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise DatasetValidationError("dataset case IDs must be unique")

    abstentions = _string_tuple(
        value.get(
            "abstention_answers",
            ["INSUFFICIENT", "UNKNOWN", "NOT ENOUGH INFORMATION"],
        ),
        "dataset.abstention_answers",
        allow_empty=False,
    )
    normalized_abstentions = [canonical_answer(answer) for answer in abstentions]
    if len(normalized_abstentions) != len(set(normalized_abstentions)):
        raise DatasetValidationError(
            "dataset.abstention_answers must be unique after normalization"
        )
    metadata = _mapping(value.get("metadata", {}), "dataset.metadata")
    description = value.get("description", "")
    if not isinstance(description, str):
        raise DatasetValidationError("dataset.description must be a string")
    return AuditDataset(
        name=_string(value.get("name"), "dataset.name"),
        description=description.strip(),
        cases=cases,
        abstention_answers=abstentions,
        metadata=dict(metadata),
    )


def load_dataset(path: str | Path) -> AuditDataset:
    """Load a UTF-8 JSON audit dataset and validate the full protocol contract."""

    source = Path(path)
    try:
        with source.open(encoding="utf-8") as stream:
            raw = json.load(stream)
    except FileNotFoundError as exc:
        raise DatasetValidationError(f"dataset file not found: {source}") from exc
    except UnicodeDecodeError as exc:
        raise DatasetValidationError(f"dataset is not valid UTF-8: {source}") from exc
    except json.JSONDecodeError as exc:
        raise DatasetValidationError(
            f"invalid JSON in {source}:{exc.lineno}:{exc.colno}: {exc.msg}"
        ) from exc
    except OSError as exc:
        raise DatasetValidationError(f"could not read dataset {source}: {exc}") from exc
    return dataset_from_mapping(_mapping(raw, "dataset"))
