"""Shared models and state constants for the release pipeline."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


NON_TERMINAL_JOB_STATES = {"QUEUED", "RUNNING"}
TERMINAL_JOB_STATES = {"ACTIVE", "FAILED", "NOOP", "REJECTED"}
ROLLBACKABLE_RELEASE_STATES = {"ACTIVE", "ACTIVE_WITH_WARNING", "SUPERSEDED"}
ALLOWED_SOURCE_EXTENSIONS = {".pdf", ".md", ".txt", ".markdown", ".docx"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class UploadItem:
    """One uploaded file that belongs to a release request."""

    source_path: str
    document_name: str
    content_hash: str
    size_bytes: int = 0

    @property
    def doc_id(self) -> str:
        return self.document_name

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "document_name": self.document_name,
            "content_hash": self.content_hash,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "UploadItem":
        return cls(
            source_path=str(value["source_path"]),
            document_name=str(value["document_name"]),
            content_hash=str(value["content_hash"]),
            size_bytes=int(value.get("size_bytes", 0)),
        )
