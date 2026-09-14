"""Tiny deterministic targets that make the audit behavior easy to inspect.

These are protocol demonstrations, not production RAG implementations.
"""

from __future__ import annotations

import re

from causal_rag_audit import TargetRequest, TargetResponse


def grounded_rag(request: TargetRequest) -> TargetResponse:
    """Answer only when the supplied documents contain a complete proof."""

    if "coolant" in request.question.casefold():
        for document in request.documents:
            match = re.fullmatch(
                r"Reactor Vega uses the coolant ([A-Za-z]+-[A-Za-z]+)\.",
                document.text,
            )
            if match:
                return TargetResponse(match.group(1), (document.id,))
        return TargetResponse("INSUFFICIENT", ())

    destination = None
    destination_document = None
    for document in request.documents:
        match = re.fullmatch(
            r"Courier Vela delivers to the depot named ([A-Za-z]+)\.",
            document.text,
        )
        if match:
            destination = match.group(1)
            destination_document = document
            break
    if destination and destination_document:
        for document in request.documents:
            match = re.fullmatch(
                rf"The {re.escape(destination)} depot is in the region ([A-Za-z]+)\.",
                document.text,
            )
            if match:
                return TargetResponse(
                    match.group(1),
                    (destination_document.id, document.id),
                )
    return TargetResponse("INSUFFICIENT", ())


def shortcut_rag(request: TargetRequest) -> TargetResponse:
    """Ignore interventions while emitting plausible citations—a failure example."""

    if "coolant" in request.question.casefold():
        return TargetResponse("Argon-X", ("D1",))
    return TargetResponse("Galdite", ("D1", "D2"))
