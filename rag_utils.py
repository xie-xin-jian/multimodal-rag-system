"""Small, dependency-free helpers shared by the RAG modules."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Iterable

_CJK_SEQUENCE_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
_LATIN_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_UPLOAD_PREFIX_RE = re.compile(r"^[0-9a-f]{8,64}_")


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "").lower()


def tokenize_for_bm25(text: str) -> list[str]:
    """Tokenize mixed Chinese/English text without an external segmenter.

    English tokens are kept whole. Chinese text is represented by characters,
    bigrams and trigrams so exact technical terms still receive useful matches.
    """
    normalized = normalize_text(text)
    tokens = _LATIN_TOKEN_RE.findall(normalized)

    for sequence in _CJK_SEQUENCE_RE.findall(normalized):
        tokens.extend(sequence)
        tokens.extend(sequence[index : index + 2] for index in range(len(sequence) - 1))
        tokens.extend(sequence[index : index + 3] for index in range(len(sequence) - 2))

    return tokens or [normalized]


def sha256_file(file_obj) -> str:
    digest = hashlib.sha256()
    while chunk := file_obj.read(1024 * 1024):
        digest.update(chunk)
    file_obj.seek(0)
    return digest.hexdigest()


def safe_upload_filename(filename: str) -> str:
    """Return a printable basename without characters unsafe on Windows."""
    name = Path(filename.replace("\\", "/")).name
    name = unicodedata.normalize("NFKC", name)
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name)
    name = re.sub(r"\s+", " ", name)
    name = re.sub(r"\.{2,}", ".", name)
    return name.strip(" .")


def display_filename(filename: str) -> str:
    return _UPLOAD_PREFIX_RE.sub("", Path(filename).name, count=1)


def resolve_within(base_dir: str | Path, filename: str) -> Path:
    """Resolve a single filename inside base_dir or raise ValueError."""
    if not filename or Path(filename).name != filename:
        raise ValueError("invalid filename")

    base = Path(base_dir).resolve()
    candidate = (base / filename).resolve()
    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise ValueError("filename escapes the data directory") from exc
    return candidate


def retrieval_metrics(
    retrieved_ids: Iterable[str | None],
    expected_ids: Iterable[str],
    k: int | None = None,
) -> dict[str, float | list[str]]:
    """Compute conventional Hit Rate@K, Recall@K, Precision@K and MRR."""
    retrieved = list(retrieved_ids)
    if k is not None:
        retrieved = retrieved[:k]

    expected = set(expected_ids)
    if not expected:
        raise ValueError("expected_ids must not be empty")

    hits = len(set(retrieved) & expected)
    reciprocal_rank = 0.0
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in expected:
            reciprocal_rank = 1.0 / rank
            break

    return {
        "hit_rate": 1.0 if hits else 0.0,
        "recall_at_k": hits / len(expected),
        "precision_at_k": hits / len(retrieved) if retrieved else 0.0,
        "mrr": reciprocal_rank,
        "retrieved": retrieved,
    }
