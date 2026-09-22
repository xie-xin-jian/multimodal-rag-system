"""SQLite-backed release registry and state queries."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from langchain_core.documents import Document

from rag_utils import sha256_file
from release_models import _hash_text, _utc_now


class ReleaseRegistryBase:
    """Persistence and read/write helpers shared by the release manager."""

    def close(self) -> None:
        self.stop_worker()
        self.stop_reconciliation()
        with self._db_lock:
            self._conn.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._db_lock:
            try:
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def _init_registry(self) -> None:
        with self._transaction() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS system_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS releases (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    parent_release_id INTEGER,
                    state TEXT NOT NULL,
                    collection_name TEXT,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    manifest_hash TEXT,
                    created_at TEXT NOT NULL,
                    activated_at TEXT,
                    error TEXT
                );

                CREATE TABLE IF NOT EXISTS release_documents (
                    release_id INTEGER NOT NULL,
                    doc_id TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    source_path TEXT,
                    content_hash TEXT NOT NULL,
                    parser_version TEXT NOT NULL,
                    chunking_version TEXT NOT NULL,
                    embedding_version TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    size_bytes INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (release_id, doc_id),
                    FOREIGN KEY (release_id) REFERENCES releases(id)
                );

                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    release_id INTEGER NOT NULL,
                    operation TEXT NOT NULL,
                    request_key TEXT NOT NULL,
                    state TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    FOREIGN KEY (release_id) REFERENCES releases(id)
                );

                CREATE INDEX IF NOT EXISTS idx_jobs_state_created
                    ON jobs(state, created_at);
                CREATE INDEX IF NOT EXISTS idx_jobs_request_key
                    ON jobs(request_key, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_releases_created
                    ON releases(created_at DESC);
                """
            )
            self._ensure_column(
                conn,
                "releases",
                "quality_state",
                "TEXT NOT NULL DEFAULT 'NOT_RUN'",
            )
            self._ensure_column(
                conn,
                "releases",
                "quality_report_json",
                "TEXT",
            )
            self._ensure_column(
                conn,
                "releases",
                "quality_gate_version",
                "TEXT",
            )
            self._ensure_column(
                conn,
                "releases",
                "force_activated",
                "INTEGER NOT NULL DEFAULT 0",
            )
            self._ensure_column(
                conn,
                "releases",
                "force_activated_at",
                "TEXT",
            )
            self._ensure_column(
                conn,
                "releases",
                "force_reason",
                "TEXT",
            )
            self._ensure_column(
                conn,
                "releases",
                "release_mode",
                "TEXT NOT NULL DEFAULT 'CONTENT_UPDATE'",
            )

    @staticmethod
    def _ensure_column(
        conn: sqlite3.Connection,
        table: str,
        column: str,
        definition: str,
    ) -> None:
        columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column not in columns:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def _recover_jobs(self) -> None:
        now = _utc_now()
        with self._transaction() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET state = 'QUEUED', started_at = NULL
                WHERE state = 'RUNNING'
                """
            )
            conn.execute(
                """
                UPDATE releases
                SET state = 'PENDING', error = 'Recovered after interrupted worker'
                WHERE state = 'STAGING'
                """
            )
            conn.execute(
                """
                UPDATE jobs
                SET state = 'FAILED', error = ?, finished_at = ?
                WHERE state = 'QUEUED'
                  AND release_id IN (
                      SELECT id FROM releases WHERE state = 'FAILED'
                  )
                """,
                ("Release failed during previous shutdown", now),
            )

    def _bootstrap_existing_release(self) -> None:
        with self._transaction() as conn:
            active = conn.execute(
                "SELECT value FROM system_state WHERE key = 'active_release_id'"
            ).fetchone()
            if active is not None:
                return

            retriever = getattr(self.rag, "retriever", None)
            if not getattr(self.rag, "_loaded", False) or not retriever:
                return
            chunks = list(getattr(retriever, "_all_chunks", []))
            if not chunks:
                return

            now = _utc_now()
            collection_name = str(getattr(retriever, "collection_name", "langchain"))
            cursor = conn.execute(
                """
                INSERT INTO releases (
                    state, collection_name, chunk_count, created_at, activated_at
                ) VALUES ('ACTIVE', ?, ?, ?, ?)
                """,
                (collection_name, len(chunks), now, now),
            )
            release_id = int(cursor.lastrowid)
            documents = self._documents_from_chunks(chunks)
            self._write_manifest(conn, release_id, documents)
            conn.execute(
                """
                UPDATE releases
                SET manifest_hash = ?
                WHERE id = ?
                """,
                (self._manifest_hash(documents), release_id),
            )
            conn.execute(
                """
                INSERT INTO system_state(key, value)
                VALUES ('active_release_id', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (str(release_id),),
            )

    def _restore_active_release(self) -> None:
        release_id = self.active_release_id
        if release_id is None:
            return
        release = self._fetch_release(release_id)
        if release is None:
            return

        current = getattr(self.rag, "retriever", None)
        current_name = getattr(current, "collection_name", None) if current else None
        if current_name == release["collection_name"] and getattr(
            self.rag, "_loaded", False
        ) == bool(release["chunk_count"]):
            return

        restored = self._load_release_retriever(release)
        self.rag.replace_retriever(restored)

    def _documents_from_chunks(self, chunks: list[Document]) -> list[dict[str, Any]]:
        grouped: dict[str, list[Document]] = {}
        for chunk in chunks:
            doc_id = self._record_doc_id(chunk)
            grouped.setdefault(doc_id, []).append(chunk)

        documents = []
        for doc_id, group in grouped.items():
            first = group[0]
            source_path = str(first.metadata.get("source") or "")
            content_hash = self._document_hash(group, source_path)
            documents.append(
                {
                    "doc_id": doc_id,
                    "display_name": str(first.metadata.get("document_name") or doc_id),
                    "source_path": source_path,
                    "content_hash": content_hash,
                    "parser_version": str(
                        first.metadata.get("parser_version", self.parser_version)
                    ),
                    "chunking_version": str(
                        first.metadata.get("chunking_version", self.chunking_version)
                    ),
                    "embedding_version": str(
                        first.metadata.get("embedding_version", self.embedding_version)
                    ),
                    "chunk_count": len(group),
                    "size_bytes": self._file_size(source_path),
                }
            )
        return sorted(documents, key=lambda item: item["display_name"])

    @property
    def active_release_id(self) -> int | None:
        with self._db_lock:
            row = self._conn.execute(
                "SELECT value FROM system_state WHERE key = 'active_release_id'"
            ).fetchone()
        return int(row["value"]) if row else None

    def active_release(self) -> dict[str, Any] | None:
        release_id = self.active_release_id
        if release_id is None:
            return None
        row = self._fetch_release(release_id)
        return self._release_to_dict(row) if row else None

    def _manifest_hash(self, documents: list[dict[str, Any]]) -> str:
        manifest = [
            {
                "doc_id": document["doc_id"],
                "content_hash": document["content_hash"],
                "parser_version": document["parser_version"],
                "chunking_version": document["chunking_version"],
                "embedding_version": document["embedding_version"],
            }
            for document in sorted(documents, key=lambda item: item["doc_id"])
        ]
        return _hash_text(json.dumps(manifest, sort_keys=True))

    def _write_manifest(
        self,
        conn: sqlite3.Connection,
        release_id: int,
        documents: list[dict[str, Any]],
    ) -> None:
        conn.execute(
            "DELETE FROM release_documents WHERE release_id = ?",
            (release_id,),
        )
        conn.executemany(
            """
            INSERT INTO release_documents (
                release_id, doc_id, display_name, source_path, content_hash,
                parser_version, chunking_version, embedding_version,
                chunk_count, size_bytes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    release_id,
                    document["doc_id"],
                    document["display_name"],
                    document.get("source_path", ""),
                    document["content_hash"],
                    document["parser_version"],
                    document["chunking_version"],
                    document["embedding_version"],
                    int(document["chunk_count"]),
                    int(document.get("size_bytes", 0)),
                )
                for document in documents
            ],
        )

    def _document_hash(self, chunks: list[Document], source_path: str) -> str:
        if source_path and os.path.exists(source_path):
            with open(source_path, "rb") as file_obj:
                return sha256_file(file_obj)
        digest = hashlib.sha256()
        for chunk in chunks:
            digest.update(chunk.page_content.encode("utf-8"))
        return digest.hexdigest()

    def _file_size(self, source_path: str) -> int:
        return (
            os.path.getsize(source_path)
            if source_path and os.path.exists(source_path)
            else 0
        )

    @staticmethod
    def _record_doc_id(document: Document) -> str:
        metadata = document.metadata
        return str(
            metadata.get("doc_id")
            or metadata.get("document_name")
            or Path(str(metadata.get("source", "unknown"))).name
        )

    def _fetch_release(self, release_id: int) -> sqlite3.Row | None:
        with self._db_lock:
            return self._conn.execute(
                "SELECT * FROM releases WHERE id = ?",
                (release_id,),
            ).fetchone()

    def _release_documents(self, release_id: int | None) -> list[dict[str, Any]]:
        if release_id is None:
            return []
        with self._db_lock:
            rows = self._conn.execute(
                """
                SELECT * FROM release_documents
                WHERE release_id = ?
                ORDER BY display_name
                """,
                (release_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_active_documents(self) -> list[dict[str, Any]]:
        release_id = self.active_release_id
        documents = self._release_documents(release_id)
        return [
            {
                "name": document["display_name"],
                "file": Path(document["source_path"]).name,
                "doc_id": document["doc_id"],
                "size": document["size_bytes"],
                "chunk_count": document["chunk_count"],
                "content_hash": document["content_hash"],
                "release_id": release_id,
            }
            for document in documents
        ]

    def list_releases(self, limit: int = 10) -> list[dict[str, Any]]:
        with self._db_lock:
            rows = self._conn.execute(
                """
                SELECT * FROM releases
                ORDER BY id DESC
                LIMIT ?
                """,
                (max(1, int(limit)),),
            ).fetchall()
        releases = []
        for row in rows:
            release = self._release_to_dict(row)
            release["documents"] = len(self._release_documents(int(row["id"])))
            releases.append(release)
        return releases

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._db_lock:
            row = self._conn.execute(
                "SELECT * FROM jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
        return self._job_to_dict(row) if row else None

    def get_status(self) -> dict[str, Any]:
        with self._publish_lock:
            active = self.active_release()
            documents = self.list_active_documents()
            with self._db_lock:
                job_row = self._conn.execute(
                    """
                    SELECT * FROM jobs
                    ORDER BY created_at DESC
                    LIMIT 1
                    """
                ).fetchone()
                queued = int(
                    self._conn.execute(
                        """
                        SELECT COUNT(*) FROM jobs
                        WHERE state IN ('QUEUED', 'RUNNING')
                        """
                    ).fetchone()[0]
                )
                reconcile_state = {
                    row["key"]: row["value"]
                    for row in self._conn.execute(
                        """
                        SELECT key, value FROM system_state
                        WHERE key IN (
                            'last_reconciliation_at',
                            'last_reconciliation_state',
                            'last_reconciliation_job_id'
                        )
                        """
                    ).fetchall()
                }
            latest_job = self._job_to_dict(job_row) if job_row else None
            return {
                "loaded": bool(getattr(self.rag, "_loaded", False)),
                "chunk_count": (
                    self.rag.retriever.chunk_count
                    if getattr(self.rag, "_loaded", False)
                    else 0
                ),
                "documents": [document["name"] for document in documents],
                "embed_model": getattr(self.rag, "embed_model", ""),
                "llm_model": getattr(self.rag, "llm_model", ""),
                "active_release": active,
                "active_release_id": active["id"] if active else None,
                "pending_jobs": queued,
                "latest_job": latest_job,
                "reconciliation": {
                    "last_run_at": reconcile_state.get("last_reconciliation_at"),
                    "state": reconcile_state.get("last_reconciliation_state"),
                    "job_id": reconcile_state.get("last_reconciliation_job_id"),
                    "interval_seconds": self.reconcile_interval_seconds,
                },
            }

    def _release_to_dict(self, row: sqlite3.Row | None) -> dict[str, Any]:
        if row is None:
            return {}
        return {
            "id": int(row["id"]),
            "parent_release_id": row["parent_release_id"],
            "state": row["state"],
            "collection_name": row["collection_name"],
            "chunk_count": int(row["chunk_count"]),
            "manifest_hash": row["manifest_hash"],
            "created_at": row["created_at"],
            "activated_at": row["activated_at"],
            "error": row["error"],
            "force_activated": bool(row["force_activated"]),
            "force_activated_at": row["force_activated_at"],
            "force_reason": row["force_reason"],
            "release_mode": row["release_mode"],
        }

    def _job_to_dict(self, row: sqlite3.Row | None) -> dict[str, Any]:
        if row is None:
            return {}
        return {
            "id": row["id"],
            "release_id": int(row["release_id"]),
            "operation": row["operation"],
            "request_key": row["request_key"],
            "state": row["state"],
            "payload_json": row["payload_json"],
            "attempts": int(row["attempts"]),
            "error": row["error"],
            "created_at": row["created_at"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
        }
