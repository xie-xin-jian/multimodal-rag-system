from __future__ import annotations

import gc
import os
import tempfile
import unittest
from unittest.mock import patch

from chromadb.api.shared_system_client import SharedSystemClient
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

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


def make_document(name: str, content: str, chunk_index: int = 0) -> Document:
    chunk_id = f"{name}#{chunk_index}"
    return Document(
        id=chunk_id,
        page_content=content,
        metadata={
            "source": f"/tmp/{name}",
            "document_name": name,
            "chunk_id": chunk_id,
            "chunk_index": chunk_index,
            "content_hash": str(hash(content)),
        },
    )


class IndexLifecycleTests(unittest.TestCase):
    def test_build_replace_delete_and_reload(self) -> None:
        embeddings = FakeEmbeddings()
        # Chroma can keep an HNSW file handle on Windows until process exit.
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as persist_dir:
            try:
                retriever = HybridRetriever(persist_directory=persist_dir)
                retriever.build_index(
                    [
                        make_document("alpha.md", "alpha old"),
                        make_document("beta.md", "beta"),
                    ],
                    embeddings,
                )
                self.assertEqual(retriever.chunk_count, 2)

                retriever.add_documents(
                    [make_document("alpha.md", "alpha new")],
                    embeddings,
                )
                self.assertEqual(retriever.chunk_count, 2)
                self.assertEqual(
                    retriever._vector_store.get_by_ids(["alpha.md#0"])[0].page_content,
                    "alpha new",
                )

                self.assertEqual(retriever.remove_document("beta.md"), 1)
                self.assertTrue(retriever.has_index)

                self.assertEqual(retriever.remove_document("alpha.md"), 1)
                self.assertEqual(retriever.chunk_count, 0)
                self.assertFalse(retriever.has_index)

                retriever.build_index(
                    [make_document("gamma.md", "gamma")],
                    embeddings,
                )
                reloaded = HybridRetriever(persist_directory=persist_dir)
                self.assertTrue(reloaded.load_index(embeddings))
                self.assertEqual(reloaded.chunk_count, 1)
                self.assertEqual(reloaded.document_names, ["gamma.md"])
            finally:
                if "retriever" in locals():
                    retriever._vector_store._client.close()
                    del retriever
                if "reloaded" in locals():
                    reloaded._vector_store._client.close()
                    del reloaded
                gc.collect()
                SharedSystemClient.clear_system_cache()


class RerankerFallbackTests(unittest.TestCase):
    @patch("retriever.CrossEncoder")
    def test_uses_cached_fallback_when_primary_model_is_unavailable(
        self, cross_encoder
    ) -> None:
        cross_encoder.side_effect = [RuntimeError("not cached"), "fallback-model"]
        retriever = HybridRetriever(
            reranker_model="primary-model",
            reranker_fallback_model="fallback-model",
        )

        with patch.dict(os.environ, {"RAG_RERANKER_ALLOW_DOWNLOAD": "0"}):
            self.assertEqual(retriever.reranker, "fallback-model")

        self.assertTrue(cross_encoder.call_args_list[0].kwargs["local_files_only"])
        self.assertTrue(cross_encoder.call_args_list[1].kwargs["local_files_only"])


if __name__ == "__main__":
    unittest.main()
