"""Causal RAG Audit public API."""

from .adapters import HttpTarget, KeywordTarget
from .api import audit
from .models import (
    AuditCase,
    AuditDataset,
    AuditReport,
    Document,
    MeanMetricResult,
    TargetRequest,
    TargetResponse,
)
from .runner import AuditRunner, ResponseFormatError
from .scoring import (
    PAPER_PROFILE,
    SCORING_PROFILES,
    STRICT_PROFILE,
    AbstentionJudge,
    AnswerJudge,
)
from .validation import DatasetValidationError, dataset_from_mapping, load_dataset
from .version import __version__

__all__ = [
    "PAPER_PROFILE",
    "SCORING_PROFILES",
    "STRICT_PROFILE",
    "AbstentionJudge",
    "AnswerJudge",
    "AuditCase",
    "AuditDataset",
    "AuditReport",
    "AuditRunner",
    "DatasetValidationError",
    "Document",
    "HttpTarget",
    "KeywordTarget",
    "MeanMetricResult",
    "ResponseFormatError",
    "TargetRequest",
    "TargetResponse",
    "__version__",
    "audit",
    "dataset_from_mapping",
    "load_dataset",
]
