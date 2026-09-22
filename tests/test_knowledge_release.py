from __future__ import annotations

import hashlib
import os
import tempfile
import unittest
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from knowledge_release import KnowledgeReleaseManager, UploadItem
from retriever import HybridRetriever


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    @staticmethod
    def _embed(text: str) -> list[float]:
        return [
            ((sum(ord(char) for char in text) + offset) % 31) / 31
            for offset in range(16)
        ]


class TestRAG:
    embed_model = "fake-embed-v1"
    llm_model = "fake-llm"

    def __init__(self, root: Path) -> None:
        self.data_directory = root / "data"
        self.data_directory.mkdir(parents=True, exist_ok=True)
        self.persist_directory = str(root / "chroma")
        self.embeddings = FakeEmbeddings()
        self.retriever = HybridRetriever(
            persist_directory=self.persist_directory,
            collection_name="langchain",
        )
        self._loaded = False
        self.fail_next_load = False

    def _load_chunks(self, path: str, document_name: str | None) -> list[Document]:
        if self.fail_next_load:
            self.fail_next_load = False
            raise RuntimeError("simulated parser failure")

        content = Path(path).read_text(encoding="utf-8")
        name = document_name or Path(path).name
        chunk_id = f"{name}#0"
        return [
            Document(
                id=chunk_id,
                page_content=content,
                metadata={
                    "source": path,
                    "document_name": name,
                    "chunk_id": chunk_id,
                    "chunk_index": 0,
                    "content_hash": hashlib.md5(content.encode("utf-8")).hexdigest(),
                },
            )
        ]

    def replace_retriever(self, retriever: HybridRetriever) -> None:
        self.retriever = retriever
        self._loaded = retriever.has_index


class KnowledgeReleaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self.temp_dir.name)
        self.rag = TestRAG(self.root)
        self.manager = KnowledgeReleaseManager(
            self.rag,
            registry_path=self.root / "registry.sqlite3",
            data_directory=self.rag.data_directory,
            embedding_version=self.rag.embed_model,
            max_attempts=1,
            start_worker=False,
        )

    def tearDown(self) -> None:
        self.manager.close()
        self.rag.retriever.dispose()
        self.temp_dir.cleanup()

    def _upload(self, name: str, content: str) -> UploadItem:
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        path = self.rag.data_directory / f"{digest[:16]}_{name}"
        path.write_text(content, encoding="utf-8")
        return UploadItem(
            source_path=str(path),
            document_name=name,
            content_hash=digest,
            size_bytes=path.stat().st_size,
        )

    def _process(self):
        result = self.manager.process_next(timeout=0.1)
        self.assertIsNotNone(result)
        return result

    def test_duplicate_request_reuses_job(self) -> None:
        item = self._upload("guide.txt", "version one")

        first = self.manager.enqueue_upload([item])
        second = self.manager.enqueue_upload([item])

        self.assertEqual(first["id"], second["id"])
        result = self._process()
        self.assertEqual(result["state"], "ACTIVE")

    def test_update_noop_rollback_and_delete_last_document(self) -> None:
        first_item = self._upload("guide.txt", "version one")
        self.manager.enqueue_upload([first_item])
        first_job = self._process()
        first_release = first_job["release_id"]
        self.assertEqual(self.manager.active_release_id, first_release)
        self.assertTrue(self.rag._loaded)
        self.assertEqual(
            self.manager.get_status()["documents"],
            ["guide.txt"],
        )

        self.manager.enqueue_upload([first_item])
        noop_result = self._process()
        self.assertEqual(noop_result["state"], "NOOP")
        self.assertEqual(self.manager.active_release_id, first_release)

        second_item = self._upload("guide.txt", "version two")
        self.manager.enqueue_upload([second_item])
        second_job = self._process()
        second_release = second_job["release_id"]
        self.assertNotEqual(first_release, second_release)
        self.assertIn(
            "version two",
            self.rag.retriever._all_chunks[0].page_content,
        )

        self.manager.enqueue_rollback(first_release)
        rollback_result = self._process()
        self.assertEqual(rollback_result["state"], "ACTIVE")
        self.assertEqual(self.manager.active_release_id, first_release)
        self.assertIn(
            "version one",
            self.rag.retriever._all_chunks[0].page_content,
        )

        self.manager.enqueue_delete(first_item.source_path, first_item.document_name)
        delete_result = self._process()
        self.assertEqual(delete_result["state"], "ACTIVE")
        self.assertFalse(self.rag._loaded)
        self.assertEqual(self.rag.retriever.chunk_count, 0)
        self.assertEqual(self.manager.list_active_documents(), [])
        self.assertEqual(self.manager.get_status()["documents"], [])

    def test_failed_release_keeps_previous_version_active(self) -> None:
        item = self._upload("guide.txt", "stable version")
        self.manager.enqueue_upload([item])
        self._process()
        active_release = self.manager.active_release_id

        bad_item = UploadItem(
            source_path=str(self.rag.data_directory / "missing.txt"),
            document_name="missing.txt",
            content_hash="missing",
        )
        self.manager.enqueue_upload([bad_item])
        result = self._process()

        self.assertEqual(result["state"], "FAILED")
        self.assertEqual(self.manager.active_release_id, active_release)
        self.assertTrue(self.rag._loaded)
        self.assertEqual(
            self.rag.retriever._all_chunks[0].page_content,
            "stable version",
        )

    def test_queued_job_recovers_after_manager_restart(self) -> None:
        item = self._upload("guide.txt", "recover me")
        self.manager.enqueue_upload([item])
        self.manager.close()

        restarted = KnowledgeReleaseManager(
            self.rag,
            registry_path=self.root / "registry.sqlite3",
            data_directory=self.rag.data_directory,
            embedding_version=self.rag.embed_model,
            max_attempts=1,
            start_worker=False,
        )
        try:
            result = restarted.process_next(timeout=0.1)
            self.assertIsNotNone(result)
            self.assertEqual(result["state"], "ACTIVE")
            self.assertTrue(self.rag._loaded)
        finally:
            restarted.close()

    def test_active_release_restores_after_restart(self) -> None:
        item = self._upload("guide.txt", "persisted release")
        self.manager.enqueue_upload([item])
        active_release = self._process()["release_id"]
        self.manager.close()

        self.rag.retriever = HybridRetriever(
            persist_directory=self.rag.persist_directory,
            collection_name="langchain",
        )
        self.rag._loaded = False

        restarted = KnowledgeReleaseManager(
            self.rag,
            registry_path=self.root / "registry.sqlite3",
            data_directory=self.rag.data_directory,
            embedding_version=self.rag.embed_model,
            max_attempts=1,
            start_worker=False,
        )
        try:
            self.assertEqual(restarted.active_release_id, active_release)
            self.assertTrue(self.rag._loaded)
            self.assertEqual(
                self.rag.retriever._all_chunks[0].page_content,
                "persisted release",
            )
        finally:
            restarted.close()

    def test_failed_job_can_be_retried(self) -> None:
        item = self._upload("guide.txt", "eventual success")
        self.rag.fail_next_load = True
        job = self.manager.enqueue_upload([item])
        failed = self._process()
        self.assertEqual(failed["state"], "FAILED")

        retry = self.manager.retry_job(job["id"])
        self.assertEqual(retry["state"], "QUEUED")
        result = self._process()

        self.assertEqual(result["state"], "ACTIVE")
        self.assertTrue(self.rag._loaded)

    def test_reconciliation_detects_external_file_change(self) -> None:
        first_item = self._upload("guide.txt", "version one")
        self.manager.enqueue_upload([first_item])
        first_release = self._process()["release_id"]

        external_item = self._upload("guide.txt", "version two")
        future = os.path.getmtime(external_item.source_path) + 10
        os.utime(external_item.source_path, (future, future))

        reconcile_job = self.manager.run_reconciliation()
        self.assertIsNotNone(reconcile_job)
        result = self._process()

        self.assertEqual(result["state"], "ACTIVE")
        self.assertNotEqual(self.manager.active_release_id, first_release)
        self.assertEqual(
            self.rag.retriever._all_chunks[0].page_content,
            "version two",
        )

    def test_cleanup_removes_old_release_and_unreferenced_source(self) -> None:
        self.manager.retain_releases = 2
        first_item = self._upload("guide.txt", "version one")
        self.manager.enqueue_upload([first_item])
        self._process()

        second_item = self._upload("guide.txt", "version two")
        self.manager.enqueue_upload([second_item])
        self._process()

        third_item = self._upload("guide.txt", "version three")
        self.manager.enqueue_upload([third_item])
        self._process()

        release_ids = [release["id"] for release in self.manager.list_releases()]
        self.assertEqual(len(release_ids), 2)
        self.assertFalse(Path(first_item.source_path).exists())
        self.assertTrue(Path(third_item.source_path).exists())


if __name__ == "__main__":
    unittest.main()
