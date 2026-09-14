"""Adapters for common synchronous RAG application boundaries."""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable, Mapping
from typing import Any

from .models import TargetRequest


class KeywordTarget:
    """Adapt a function accepting ``question``, ``documents``, and ``metadata`` keywords."""

    def __init__(self, function: Callable[..., Any]) -> None:
        if not callable(function):
            raise TypeError("function must be callable")
        self.function = function

    def __call__(self, request: TargetRequest) -> Any:
        return self.function(
            question=request.question,
            documents=[document.to_dict() for document in request.documents],
            metadata=dict(request.metadata),
        )


class HttpTarget:
    """POST audit requests to a JSON HTTP endpoint using the standard library."""

    def __init__(
        self,
        url: str,
        *,
        timeout_seconds: float = 60.0,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        if not url.startswith(("http://", "https://")):
            raise ValueError("HTTP target URL must start with http:// or https://")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.url = url
        self.timeout_seconds = timeout_seconds
        self.headers = {"Content-Type": "application/json", **dict(headers or {})}

    def __call__(self, request: TargetRequest) -> Any:
        payload = json.dumps(request.to_dict(), ensure_ascii=False).encode("utf-8")
        http_request = urllib.request.Request(
            self.url,
            data=payload,
            headers=self.headers,
            method="POST",
        )
        with urllib.request.urlopen(
            http_request, timeout=self.timeout_seconds
        ) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            return json.loads(response.read().decode(charset))
