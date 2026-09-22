"""Background release worker, reconciliation, and publication."""

from __future__ import annotations

import json
import queue
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from rag_utils import display_filename, sha256_file
from release_models import (
    ALLOWED_SOURCE_EXTENSIONS,
    ROLLBACKABLE_RELEASE_STATES,
    UploadItem,
    _hash_text,
    _utc_now,
)
from retriever import HybridRetriever


class ReleaseWorkerBase:
    """Queued release processing and active-release publication."""

    def start_worker(self) -> None:
        self._worker_enabled = True
        with self._worker_guard:
            if self._worker_thread and self._worker_thread.is_alive():
                return
            self._stop_event.clear()
            self._worker_thread = threading.Thread(
                target=self._worker_loop,
                name="knowledge-release-worker",
                daemon=True,
            )
            self._worker_thread.start()

    def stop_worker(self) -> None:
        self._worker_enabled = False
        self._stop_event.set()
        thread = self._worker_thread
        if thread and thread.is_alive():
            thread.join(timeout=5)
        self._worker_thread = None

    def start_reconciliation(self, interval_seconds: int | None = None) -> None:
        if interval_seconds is not None:
            self.reconcile_interval_seconds = max(0, int(interval_seconds))
        if self.reconcile_interval_seconds <= 0:
            return
        with self._reconcile_guard:
            if self._reconcile_thread and self._reconcile_thread.is_alive():
                return
            self._reconcile_stop_event.clear()
            self._reconcile_thread = threading.Thread(
                target=self._reconciliation_loop,
                name="knowledge-reconciliation",
                daemon=True,
            )
            self._reconcile_thread.start()

    def stop_reconciliation(self) -> None:
        self._reconcile_stop_event.set()
        thread = self._reconcile_thread
        if thread and thread.is_alive():
            thread.join(timeout=5)
        self._reconcile_thread = None

    def _reconciliation_loop(self) -> None:
        while not self._reconcile_stop_event.wait(self.reconcile_interval_seconds):
            try:
                self.run_reconciliation()
            except Exception as exc:
                print(f"[release] reconciliation failed: {exc}")

    def run_reconciliation(self) -> dict[str, Any] | None:
        """Detect source files that were changed outside the Web API."""
        active_id = self.active_release_id
        if active_id is None:
            self._set_reconciliation_state("NO_ACTIVE_RELEASE", None)
            return None

        active_documents = self._release_documents(active_id)
        active_by_name = {
            document["display_name"]: document for document in active_documents
        }
        latest_files: dict[str, Path] = {}
        if self.data_directory.exists():
            for path in self.data_directory.iterdir():
                if not path.is_file():
                    continue
                if path.suffix.lower() not in ALLOWED_SOURCE_EXTENSIONS:
                    continue
                display_name = display_filename(path.name)
                current = latest_files.get(display_name)
                if current is None or path.stat().st_mtime > current.stat().st_mtime:
                    latest_files[display_name] = path

        changed: list[UploadItem] = []
        for display_name, path in sorted(latest_files.items()):
            with path.open("rb") as file_obj:
                content_hash = sha256_file(file_obj)
            current = active_by_name.get(display_name)
            if current and current["content_hash"] == content_hash:
                continue
            changed.append(
                UploadItem(
                    source_path=str(path.resolve()),
                    document_name=display_name,
                    content_hash=content_hash,
                    size_bytes=path.stat().st_size,
                )
            )

        if not changed:
            self._set_reconciliation_state("NO_CHANGES", None)
            return None

        job = self.enqueue_upload(changed)
        self._set_reconciliation_state("QUEUED", job["id"])
        return job

    def _set_reconciliation_state(
        self,
        state: str,
        job_id: str | None,
    ) -> None:
        with self._transaction() as conn:
            values = {
                "last_reconciliation_at": _utc_now(),
                "last_reconciliation_state": state,
                "last_reconciliation_job_id": job_id or "",
            }
            conn.executemany(
                """
                INSERT INTO system_state(key, value)
                VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                values.items(),
            )

    def ensure_worker(self) -> None:
        if self._worker_enabled:
            self.start_worker()

    def process_next(self, timeout: float = 10.0) -> dict[str, Any] | None:
        try:
            job_id = self._queue.get(timeout=timeout)
        except queue.Empty:
            job_id = self._next_queued_job_id()
            if job_id is None:
                return None
        try:
            self._process_job(job_id)
            return self.get_job(job_id)
        finally:
            self._queue.task_done()

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                job_id = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._process_job(job_id)
            finally:
                self._queue.task_done()

    def _enqueue_queued_jobs(self) -> None:
        with self._db_lock:
            rows = self._conn.execute(
                "SELECT id FROM jobs WHERE state = 'QUEUED' ORDER BY created_at"
            ).fetchall()
        for row in rows:
            self._queue.put(str(row["id"]))

    def _next_queued_job_id(self) -> str | None:
        with self._db_lock:
            row = self._conn.execute(
                """
                SELECT id FROM jobs
                WHERE state = 'QUEUED'
                ORDER BY created_at
                LIMIT 1
                """
            ).fetchone()
        return str(row["id"]) if row else None

    def _process_job(self, job_id: str) -> None:
        job = self.get_job(job_id)
        if not job or job["state"] != "QUEUED":
            return

        attempts = int(job["attempts"]) + 1
        with self._transaction() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET state = 'RUNNING', attempts = ?, started_at = ?, error = NULL
                WHERE id = ?
                """,
                (attempts, _utc_now(), job_id),
            )
            if job["operation"] != "ROLLBACK":
                conn.execute(
                    "UPDATE releases SET state = 'STAGING', error = NULL WHERE id = ?",
                    (job["release_id"],),
                )

        stage: HybridRetriever | None = None
        try:
            payload = json.loads(job["payload_json"])
            if job["operation"] == "UPLOAD":
                stage = self._process_upload(job, payload)
            elif job["operation"] == "DELETE":
                stage = self._process_delete(job, payload)
            elif job["operation"] == "ROLLBACK":
                stage = self._process_rollback(job, payload)
            else:
                raise ValueError(f"unsupported operation: {job['operation']}")
            if stage is not None:
                self._activate_release(
                    release_id=int(job["release_id"]),
                    stage=stage,
                    job_id=job_id,
                )
        except Exception as exc:
            if stage is not None:
                stage.dispose()
            self._handle_job_failure(job, job_id, attempts, str(exc))

    def _process_upload(
        self,
        job: dict[str, Any],
        payload: dict[str, Any],
    ) -> HybridRetriever | None:
        items = [UploadItem.from_dict(value) for value in payload["items"]]
        release_id = int(job["release_id"])
        active_id = self.active_release_id
        active_documents = self._release_documents(active_id) if active_id else []
        active_by_id = {document["doc_id"]: document for document in active_documents}
        release_mode = self._detect_release_mode(active_documents)
        self._set_release_mode(release_id, release_mode)

        if active_id is not None and all(
            self._same_document(item, active_by_id.get(item.doc_id)) for item in items
        ):
            self._mark_noop(job, "No content or indexing-version change detected")
            return None

        affected_ids = {item.doc_id for item in items}
        if release_mode == "INDEX_UPGRADE":
            records, documents = self._build_upgrade_snapshot(
                release_id,
                items,
                active_documents,
            )
        else:
            records, documents = self._build_content_update_snapshot(
                release_id,
                items,
                active_documents,
                affected_ids,
            )
        documents.sort(key=lambda document: document["display_name"])
        self._stage_release(release_id, records, documents)
        return self._build_stage(release_id, records)

    def _build_content_update_snapshot(
        self,
        release_id: int,
        items: list[UploadItem],
        active_documents: list[dict[str, Any]],
        affected_ids: set[str],
    ) -> tuple[list[tuple[Document, list[float]]], list[dict[str, Any]]]:
        records = [
            (document, vector)
            for document, vector in self._active_records()
            if self._record_doc_id(document) not in affected_ids
        ]
        new_chunks: list[Document] = []
        new_documents: list[dict[str, Any]] = []
        for item in items:
            chunks = self._load_item_chunks(item, release_id)
            if not chunks:
                raise RuntimeError(f"no chunks produced for {item.document_name}")
            new_chunks.extend(chunks)
            new_documents.append(self._document_manifest(item, len(chunks)))

        if new_chunks:
            vectors = self.rag.embeddings.embed_documents(
                [chunk.page_content for chunk in new_chunks]
            )
            records.extend(zip(new_chunks, vectors, strict=True))

        documents = [
            document
            for document in active_documents
            if document["doc_id"] not in affected_ids
        ]
        documents.extend(new_documents)
        return records, documents

    def _build_upgrade_snapshot(
        self,
        release_id: int,
        items: list[UploadItem],
        active_documents: list[dict[str, Any]],
    ) -> tuple[list[tuple[Document, list[float]]], list[dict[str, Any]]]:
        replacements = {item.doc_id: item for item in items}
        rebuild_items = [
            UploadItem(
                source_path=document["source_path"],
                document_name=document["display_name"],
                content_hash=document["content_hash"],
                size_bytes=document["size_bytes"],
            )
            for document in active_documents
            if document["doc_id"] not in replacements
        ]
        rebuild_items.extend(items)

        all_chunks: list[Document] = []
        documents: list[dict[str, Any]] = []
        for item in rebuild_items:
            chunks = self._load_item_chunks(item, release_id)
            if not chunks:
                raise RuntimeError(f"no chunks produced for {item.document_name}")
            all_chunks.extend(chunks)
            documents.append(self._document_manifest(item, len(chunks)))

        vectors = self.rag.embeddings.embed_documents(
            [chunk.page_content for chunk in all_chunks]
        )
        return list(zip(all_chunks, vectors, strict=True)), documents

    def _document_manifest(
        self,
        item: UploadItem,
        chunk_count: int,
    ) -> dict[str, Any]:
        return {
            "doc_id": item.doc_id,
            "display_name": item.document_name,
            "source_path": item.source_path,
            "content_hash": item.content_hash,
            "parser_version": self.parser_version,
            "chunking_version": self.chunking_version,
            "embedding_version": self.embedding_version,
            "chunk_count": chunk_count,
            "size_bytes": item.size_bytes,
        }

    def _detect_release_mode(
        self,
        active_documents: list[dict[str, Any]],
    ) -> str:
        for document in active_documents:
            if (
                document["parser_version"] != self.parser_version
                or document["chunking_version"] != self.chunking_version
                or document["embedding_version"] != self.embedding_version
            ):
                return "INDEX_UPGRADE"
        return "CONTENT_UPDATE"

    def _set_release_mode(self, release_id: int, mode: str) -> None:
        with self._transaction() as conn:
            conn.execute(
                "UPDATE releases SET release_mode = ? WHERE id = ?",
                (mode, release_id),
            )

    def _process_delete(
        self,
        job: dict[str, Any],
        payload: dict[str, Any],
    ) -> HybridRetriever | None:
        active_id = self.active_release_id
        if active_id is None:
            self._mark_noop(job, "Knowledge base is already empty")
            return None

        target_doc_id = str(payload["doc_id"])
        documents = self._release_documents(active_id)
        target = next(
            (
                document
                for document in documents
                if document["doc_id"] == target_doc_id
                or document["display_name"] == target_doc_id
                or document["source_path"].endswith(target_doc_id)
            ),
            None,
        )
        if target is None:
            self._mark_noop(job, "Document is not present in the active release")
            return None

        target_doc_id = target["doc_id"]
        records = [
            (document, vector)
            for document, vector in self._active_records()
            if self._record_doc_id(document) != target_doc_id
        ]
        documents = [
            document for document in documents if document["doc_id"] != target_doc_id
        ]
        release_id = int(job["release_id"])
        self._stage_release(release_id, records, documents)
        return self._build_stage(release_id, records)

    def _process_rollback(
        self,
        job: dict[str, Any],
        payload: dict[str, Any],
    ) -> HybridRetriever | None:
        target_release_id = int(payload["target_release_id"])
        if target_release_id == self.active_release_id:
            self._mark_noop(job, "Requested release is already active")
            return None
        release = self._fetch_release(target_release_id)
        if release is None or release["state"] not in ROLLBACKABLE_RELEASE_STATES:
            raise ValueError("target release is not rollbackable")
        return self._load_release_retriever(release)

    def _stage_release(
        self,
        release_id: int,
        records: list[tuple[Document, list[float]]],
        documents: list[dict[str, Any]],
    ) -> None:
        state = "STAGED"
        with self._transaction() as conn:
            self._write_manifest(conn, release_id, documents)
            conn.execute(
                """
                UPDATE releases
                SET state = ?, collection_name = ?, chunk_count = ?,
                    manifest_hash = ?, error = NULL
                WHERE id = ?
                """,
                (
                    state,
                    f"rag_release_{release_id}",
                    len(records),
                    self._manifest_hash(documents),
                    release_id,
                ),
            )

    def _build_stage(
        self,
        release_id: int,
        records: list[tuple[Document, list[float]]],
    ) -> HybridRetriever:
        stage = HybridRetriever(
            persist_directory=self.rag.persist_directory,
            collection_name=f"rag_release_{release_id}",
        )
        try:
            stage.build_index_from_records(records, self.rag.embeddings)
            if stage.chunk_count != len(records):
                raise RuntimeError(
                    f"release validation failed: expected {len(records)} chunks, "
                    f"got {stage.chunk_count}"
                )
            return stage
        except Exception:
            stage.dispose()
            raise

    def _activate_release(
        self,
        release_id: int,
        stage: HybridRetriever,
        job_id: str,
    ) -> None:
        with self._publish_lock:
            with self._transaction() as conn:
                current_row = conn.execute(
                    "SELECT value FROM system_state WHERE key = 'active_release_id'"
                ).fetchone()
                current_id = int(current_row["value"]) if current_row else None
                if current_id is not None and current_id != release_id:
                    conn.execute(
                        "UPDATE releases SET state = 'SUPERSEDED' WHERE id = ?",
                        (current_id,),
                    )
                conn.execute(
                    """
                    UPDATE releases
                    SET state = 'ACTIVE', activated_at = ?, error = NULL,
                        force_activated = 0,
                        force_activated_at = NULL,
                        force_reason = NULL
                    WHERE id = ?
                    """,
                    (
                        _utc_now(),
                        release_id,
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO system_state(key, value)
                    VALUES ('active_release_id', ?)
                    ON CONFLICT(key) DO UPDATE SET value = excluded.value
                    """,
                    (str(release_id),),
                )
                conn.execute(
                    """
                    UPDATE jobs
                    SET state = 'ACTIVE', error = NULL, finished_at = ?
                    WHERE id = ?
                    """,
                    (_utc_now(), job_id),
                )
            self.rag.replace_retriever(stage)
        self._cleanup_old_releases()

    def _cleanup_old_releases(self) -> None:
        active_id = self.active_release_id
        with self._db_lock:
            rows = self._conn.execute(
                "SELECT * FROM releases ORDER BY id DESC"
            ).fetchall()

        protected_ids: set[int] = set()
        if active_id is not None:
            protected_ids.add(active_id)

        for row in rows:
            release_id = int(row["id"])
            if release_id in protected_ids:
                continue
            protected_ids.add(release_id)
            if len(protected_ids) >= self.retain_releases:
                break

        candidates = [
            row
            for row in rows
            if int(row["id"]) not in protected_ids
            and row["state"] in {"SUPERSEDED", "FAILED", "REJECTED", "NOOP"}
        ]
        if not candidates:
            return

        protected_paths: set[str] = set()
        candidate_paths: set[str] = set()
        with self._db_lock:
            for release_id in protected_ids:
                protected_paths.update(
                    str(row["source_path"])
                    for row in self._conn.execute(
                        """
                        SELECT source_path FROM release_documents
                        WHERE release_id = ? AND source_path != ''
                        """,
                        (release_id,),
                    ).fetchall()
                )
            for row in candidates:
                candidate_paths.update(
                    str(document["source_path"])
                    for document in self._release_documents(int(row["id"]))
                    if document["source_path"]
                )

        with self._transaction() as conn:
            for row in candidates:
                release_id = int(row["id"])
                conn.execute("DELETE FROM jobs WHERE release_id = ?", (release_id,))
                conn.execute(
                    "DELETE FROM release_documents WHERE release_id = ?",
                    (release_id,),
                )
                conn.execute("DELETE FROM releases WHERE id = ?", (release_id,))

        for row in candidates:
            collection_name = row["collection_name"]
            if not collection_name or int(row["chunk_count"]) == 0:
                continue
            retriever = HybridRetriever(
                persist_directory=self.rag.persist_directory,
                collection_name=str(collection_name),
            )
            try:
                retriever.dispose()
            except Exception as exc:
                print(f"[release] failed to remove collection {collection_name}: {exc}")

        for source_path in candidate_paths - protected_paths:
            path = Path(source_path)
            try:
                if path.is_file():
                    path.unlink()
            except OSError as exc:
                print(f"[release] failed to remove old source {path}: {exc}")

    def _handle_job_failure(
        self,
        job: dict[str, Any],
        job_id: str,
        attempts: int,
        error: str,
    ) -> None:
        retryable = attempts < self.max_attempts
        state = "QUEUED" if retryable else "FAILED"
        with self._transaction() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET state = ?, error = ?, finished_at = ?
                WHERE id = ?
                """,
                (
                    state,
                    error,
                    None if retryable else _utc_now(),
                    job_id,
                ),
            )
            if job["operation"] != "ROLLBACK":
                conn.execute(
                    """
                    UPDATE releases
                    SET state = ?, error = ?
                    WHERE id = ?
                    """,
                    ("PENDING" if retryable else "FAILED", error, job["release_id"]),
                )
        if retryable:
            time.sleep(min(2 ** max(0, attempts - 1), 5))
            self._queue.put(job_id)

    def _mark_noop(self, job: dict[str, Any], reason: str) -> None:
        with self._transaction() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET state = 'NOOP', error = NULL, finished_at = ?
                WHERE id = ?
                """,
                (_utc_now(), job["id"]),
            )
            if job["operation"] != "ROLLBACK":
                conn.execute(
                    """
                    UPDATE releases
                    SET state = 'NOOP', error = ?
                    WHERE id = ?
                    """,
                    (reason, job["release_id"]),
                )

    def _active_records(self) -> list[tuple[Document, list[float]]]:
        with self._publish_lock:
            release_id = self.active_release_id
            if release_id is None:
                return []
            release = self._fetch_release(release_id)
            if release is None or int(release["chunk_count"]) == 0:
                return []
            return self.rag.retriever.snapshot()

    def _load_release_retriever(self, release: sqlite3.Row) -> HybridRetriever:
        retriever = HybridRetriever(
            persist_directory=self.rag.persist_directory,
            collection_name=str(release["collection_name"]),
        )
        if int(release["chunk_count"]) == 0:
            retriever.clear_index()
            return retriever
        if not retriever.load_index(self.rag.embeddings):
            raise RuntimeError(
                f"release {release['id']} collection could not be loaded"
            )
        return retriever

    def _load_item_chunks(
        self,
        item: UploadItem,
        release_id: int,
    ) -> list[Document]:
        path = Path(item.source_path).resolve()
        try:
            path.relative_to(self.data_directory)
        except ValueError as exc:
            raise ValueError("upload path escapes the data directory") from exc
        if not path.exists():
            raise FileNotFoundError(path)

        chunks = self.rag._load_chunks(str(path), item.document_name)
        for chunk in chunks:
            chunk.metadata.update(
                {
                    "doc_id": item.doc_id,
                    "document_name": item.document_name,
                    "document_hash": item.content_hash,
                    "parser_version": self.parser_version,
                    "chunking_version": self.chunking_version,
                    "embedding_version": self.embedding_version,
                    "release_id": str(release_id),
                }
            )
        return chunks

    def _same_document(
        self,
        item: UploadItem,
        current: dict[str, Any] | None,
    ) -> bool:
        if current is None:
            return False
        return (
            current["content_hash"] == item.content_hash
            and current["parser_version"] == self.parser_version
            and current["chunking_version"] == self.chunking_version
            and current["embedding_version"] == self.embedding_version
        )

    def _upload_request_key(self, items: list[UploadItem]) -> str:
        payload = {
            "operation": "UPLOAD",
            "active_release_id": self.active_release_id,
            "items": sorted(
                (
                    {
                        "doc_id": item.doc_id,
                        "content_hash": item.content_hash,
                    }
                    for item in items
                ),
                key=lambda item: item["doc_id"],
            ),
            "parser_version": self.parser_version,
            "chunking_version": self.chunking_version,
            "embedding_version": self.embedding_version,
        }
        return _hash_text(json.dumps(payload, sort_keys=True))
