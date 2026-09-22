"""Versioned knowledge-base release pipeline.

The manager owns public request APIs. Persistence lives in release_registry.py,
while background processing lives in release_worker.py.
"""

from __future__ import annotations

import json
import queue
import sqlite3
import threading
import uuid
from pathlib import Path
from typing import Any

from release_models import (
    ROLLBACKABLE_RELEASE_STATES,
    UploadItem,
    _hash_text,
    _utc_now,
)
from release_registry import ReleaseRegistryBase
from release_worker import ReleaseWorkerBase


class KnowledgeReleaseManager(ReleaseRegistryBase, ReleaseWorkerBase):
    """Coordinate durable ingest jobs and atomic index releases."""

    def __init__(
        self,
        rag,
        registry_path: str | Path,
        data_directory: str | Path | None = None,
        *,
        parser_version: str = "parser-v1",
        chunking_version: str = "chunk-500-50-v1",
        embedding_version: str | None = None,
        max_attempts: int = 3,
        retain_releases: int = 5,
        reconcile_interval_seconds: int = 0,
        start_worker: bool = True,
    ) -> None:
        self.rag = rag
        self.registry_path = Path(registry_path).resolve()
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.data_directory = Path(
            data_directory or getattr(rag, "data_directory", "data")
        ).resolve()
        self.parser_version = parser_version
        self.chunking_version = chunking_version
        self.embedding_version = (
            embedding_version
            or getattr(rag, "embed_model", None)
            or "embedding-default"
        )
        self.max_attempts = max(1, int(max_attempts))
        self.retain_releases = max(2, int(retain_releases))
        self.reconcile_interval_seconds = max(0, int(reconcile_interval_seconds))

        self._db_lock = threading.RLock()
        self._publish_lock = threading.RLock()
        self._worker_guard = threading.Lock()
        self._reconcile_guard = threading.Lock()
        self._queue: queue.Queue[str] = queue.Queue()
        self._stop_event = threading.Event()
        self._reconcile_stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None
        self._reconcile_thread: threading.Thread | None = None
        self._worker_enabled = bool(start_worker)

        self._conn = sqlite3.connect(
            str(self.registry_path),
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._init_registry()
        self._recover_jobs()
        self._bootstrap_existing_release()
        self._restore_active_release()
        self._enqueue_queued_jobs()
        if self._worker_enabled:
            self.start_worker()
        if self.reconcile_interval_seconds > 0:
            self.start_reconciliation()

    def enqueue_upload(self, items: list[UploadItem]) -> dict[str, Any]:
        if not items:
            raise ValueError("upload items must not be empty")
        payload = {"items": [item.to_dict() for item in items]}
        request_key = self._upload_request_key(items)
        return self._create_job("UPLOAD", request_key, payload)

    def enqueue_delete(
        self,
        filename: str,
        doc_id: str | None = None,
    ) -> dict[str, Any]:
        active_id = self.active_release_id
        if active_id is None:
            raise RuntimeError("knowledge base is empty")
        payload = {"filename": filename, "doc_id": doc_id or filename}
        request_key = _hash_text(
            json.dumps(
                {
                    "operation": "DELETE",
                    "active_release_id": active_id,
                    "payload": payload,
                },
                sort_keys=True,
            )
        )
        return self._create_job("DELETE", request_key, payload)

    def enqueue_rollback(self, target_release_id: int) -> dict[str, Any]:
        active_id = self.active_release_id
        if active_id is None:
            raise RuntimeError("knowledge base is empty")
        target = self._fetch_release(target_release_id)
        if target is None:
            raise ValueError("target release does not exist")
        if target["state"] not in ROLLBACKABLE_RELEASE_STATES:
            raise ValueError("target release is not rollbackable")
        payload = {"target_release_id": int(target_release_id)}
        request_key = _hash_text(
            json.dumps(
                {
                    "operation": "ROLLBACK",
                    "active_release_id": active_id,
                    "target_release_id": int(target_release_id),
                },
                sort_keys=True,
            )
        )
        return self._create_job(
            "ROLLBACK",
            request_key,
            payload,
            release_id=int(target_release_id),
        )

    def retry_job(self, job_id: str) -> dict[str, Any]:
        with self._transaction() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                raise ValueError("job does not exist")
            if row["state"] not in {"FAILED", "REJECTED"}:
                raise ValueError("only failed or rejected jobs can be retried")
            conn.execute(
                """
                UPDATE jobs
                SET state = 'QUEUED', error = NULL, finished_at = NULL
                WHERE id = ?
                """,
                (job_id,),
            )
            if row["operation"] != "ROLLBACK":
                conn.execute(
                    "UPDATE releases SET state = 'PENDING' WHERE id = ?",
                    (row["release_id"],),
                )
        self._queue.put(job_id)
        self.ensure_worker()
        return self.get_job(job_id)

    def _create_job(
        self,
        operation: str,
        request_key: str,
        payload: dict[str, Any],
        *,
        release_id: int | None = None,
    ) -> dict[str, Any]:
        with self._transaction() as conn:
            existing = conn.execute(
                """
                SELECT * FROM jobs
                WHERE request_key = ?
                  AND state IN ('QUEUED', 'RUNNING', 'ACTIVE', 'NOOP')
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (request_key,),
            ).fetchone()
            if existing is not None:
                return self._job_to_dict(existing)

            if release_id is None:
                parent_id = self.active_release_id
                cursor = conn.execute(
                    """
                    INSERT INTO releases (parent_release_id, state, created_at)
                    VALUES (?, 'PENDING', ?)
                    """,
                    (parent_id, _utc_now()),
                )
                release_id = int(cursor.lastrowid)

            job_id = uuid.uuid4().hex
            conn.execute(
                """
                INSERT INTO jobs (
                    id, release_id, operation, request_key, state,
                    payload_json, created_at
                ) VALUES (?, ?, ?, ?, 'QUEUED', ?, ?)
                """,
                (
                    job_id,
                    release_id,
                    operation,
                    request_key,
                    json.dumps(payload, ensure_ascii=False, sort_keys=True),
                    _utc_now(),
                ),
            )
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            result = self._job_to_dict(row)
        self._queue.put(job_id)
        self.ensure_worker()
        return result
