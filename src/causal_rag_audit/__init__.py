"""Causal RAG Audit public API."""

from .adapters import HttpTarget, KeywordTarget
from .api import audit
from .models import (
    AuditCase,
    AuditDataset,
    AuditReport,
    Document,
    TargetRequest,
    TargetResponse,
)
from .runner import AuditRunner, ResponseFormatError
from .validation import DatasetValidationError, dataset_from_mapping, load_dataset
from .version import __version__

__all__ = [
    "AuditCase",
    "AuditDataset",
    "AuditReport",
    "AuditRunner",
    "DatasetValidationError",
    "Document",
    "HttpTarget",
    "KeywordTarget",
    "ResponseFormatError",
    "TargetRequest",
    "TargetResponse",
    "__version__",
    "audit",
    "dataset_from_mapping",
    "load_dataset",
]
