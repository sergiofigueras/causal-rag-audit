"""Convenience API for embedding the framework in Python applications."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .models import AuditDataset, AuditReport
from .runner import AuditRunner, Target
from .scoring import PAPER_PROFILE, AbstentionJudge, AnswerJudge
from .validation import load_dataset


def audit(
    target: Target,
    dataset: AuditDataset | str | Path,
    *,
    target_name: str | None = None,
    max_workers: int = 1,
    include_raw: bool = False,
    target_metadata: Mapping[str, Any] | None = None,
    scoring_profile: str = PAPER_PROFILE,
    answer_judge: AnswerJudge | None = None,
    abstention_judge: AbstentionJudge | None = None,
) -> AuditReport:
    """Run a causal audit from a validated object or JSON dataset path."""

    validated = load_dataset(dataset) if isinstance(dataset, (str, Path)) else dataset
    return AuditRunner(
        target,
        target_name=target_name,
        max_workers=max_workers,
        include_raw=include_raw,
        target_metadata=target_metadata,
        scoring_profile=scoring_profile,
        answer_judge=answer_judge,
        abstention_judge=abstention_judge,
    ).run(validated)
