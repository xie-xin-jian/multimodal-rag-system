"""混合检索器：BM25 + Dense 召回，CrossEncoder 精排。

继承 BaseRetriever 以便直接接入 LangChain 的 RetrievalQA / LCEL 链路，
不必在 main 里再绕一层。
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from pydantic import PrivateAttr
from sentence_transformers import CrossEncoder

from rag_utils import tokenize_for_bm25

_REWRITE_PROMPT = """将下面的问题改写成 {n} 个语义等价但表达不同的检索子问题，每行一条，不要编号、不要解释。
原问题：{query}
子问题："""


class HybridRetriever(BaseRetriever):
    persist_directory: str = "chroma_db"
    bm25_weight: float = 0.5
    dense_weight: float = 0.5
    candidate_k: int = 10
    final_top_k: int = 3
    collection_name: str = "langchain"
    reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    reranker_fallback_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # BaseRetriever 基于 pydantic，运行期对象走 PrivateAttr
    _vector_store: Optional[Chroma] = PrivateAttr(default=None)
    _bm25_retriever: Optional[BM25Retriever] = PrivateAttr(default=None)
    _ensemble_retriever: Optional[EnsembleRetriever] = PrivateAttr(default=None)
    _reranker: Optional[CrossEncoder] = PrivateAttr(default=None)
    _all_chunks: list[Document] = PrivateAttr(default_factory=list)

    # ---- 索引构建 / 加载 / 增量更新 ----

    def build_index(self, chunks: list[Document], embeddings) -> None:
        if not chunks:
            raise ValueError("empty chunk list")

        self._all_chunks = self._deduplicate_documents(chunks)
        self._vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=embeddings,
            persist_directory=self.persist_directory,
        )
        self._vector_store.reset_collection()
        self._add_to_vector_store(self._all_chunks)
        self._init_bm25_and_ensemble()
        print(
            f"[retriever] index built: {len(self._all_chunks)} chunks "
            f"-> {self.persist_directory}"
        )

    def load_index(self, embeddings) -> bool:
        if not os.path.exists(self.persist_directory):
            return False
        try:
            self._vector_store = Chroma(
                collection_name=self.collection_name,
                persist_directory=self.persist_directory,
                embedding_function=embeddings,
            )
            raw = self._vector_store.get(include=["documents", "metadatas"])
            chunks = []
            for doc_id, page_content, metadata in zip(
                raw["ids"], raw["documents"], raw["metadatas"]
            ):
                chunks.append(
                    Document(
                        id=doc_id,
                        page_content=page_content,
                        metadata=metadata,
                    )
                )
            if not chunks:
                return False
            self._all_chunks = self._deduplicate_documents(chunks)
            self._init_bm25_and_ensemble()
            print(
                f"[retriever] index loaded: {len(chunks)} chunks <- {self.persist_directory}"
            )
            return True
        except Exception as e:
            print(f"[retriever] failed to load index: {e}")
            return False

    def _init_bm25_and_ensemble(self) -> None:
        if not self._all_chunks:
            self._bm25_retriever = None
            self._ensemble_retriever = None
            return

        self._bm25_retriever = BM25Retriever.from_documents(
            self._all_chunks,
            preprocess_func=tokenize_for_bm25,
        )
        self._bm25_retriever.k = self.candidate_k
        self._rebuild_ensemble()

    def add_documents(self, new_chunks: list[Document], embeddings) -> None:
        if not self._vector_store:
            raise RuntimeError("index not built, call build_index first")
        deduplicated = self._deduplicate_documents(new_chunks)
        if not deduplicated:
            return

        # 同一逻辑文档再次上传时先删除旧 chunk，避免改短文档后留下残片。
        replacement_keys = set()
        for chunk in deduplicated:
            replacement_keys.update(self._document_keys(chunk))

        stale_chunks = [
            chunk
            for chunk in self._all_chunks
            if replacement_keys.intersection(self._document_keys(chunk))
        ]
        if stale_chunks:
            self._vector_store.delete(
                ids=[self._document_id(doc) for doc in stale_chunks]
            )
            stale_ids = {self._document_id(doc) for doc in stale_chunks}
            self._all_chunks = [
                chunk
                for chunk in self._all_chunks
                if self._document_id(chunk) not in stale_ids
            ]

        self._all_chunks.extend(deduplicated)
        self._add_to_vector_store(deduplicated)
        self._init_bm25_and_ensemble()
        print(
            f"[retriever] incremental update: +{len(deduplicated)} "
            f"(total={len(self._all_chunks)})"
        )

    def remove_document(self, filename: str) -> int:
        """从知识库中删除指定文件的所有 chunk，返回删除数量。"""
        if not self._vector_store:
            raise RuntimeError("index not built")

        to_remove = [
            c
            for c in self._all_chunks
            if (
                filename in self._document_keys(c)
                or os.path.basename(c.metadata.get("source", "")) == filename
            )
        ]

        if not to_remove:
            return 0

        self._vector_store.delete(ids=[self._document_id(doc) for doc in to_remove])

        self._all_chunks = [
            c
            for c in self._all_chunks
            if self._document_id(c) not in {self._document_id(doc) for doc in to_remove}
        ]
        self._init_bm25_and_ensemble()

        print(
            f"[retriever] removed {filename}: -{len(to_remove)} chunks (total={len(self._all_chunks)})"
        )
        return len(to_remove)

    def _rebuild_ensemble(self) -> None:
        if not self._vector_store or not self._bm25_retriever:
            self._ensemble_retriever = None
            return

        dense = self._vector_store.as_retriever(search_kwargs={"k": self.candidate_k})
        self._ensemble_retriever = EnsembleRetriever(
            retrievers=[self._bm25_retriever, dense],
            weights=[self.bm25_weight, self.dense_weight],
        )

    # ---- 检索入口 ----

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        """BaseRetriever 接口实现，供 RetrievalQA 调用。"""
        return self.retrieve_with_rerank(query, top_k=self.final_top_k)

    def retrieve_with_rerank(
        self, query: str, top_k: Optional[int] = None
    ) -> list[Document]:
        if not self._ensemble_retriever:
            return []
        top_k = top_k or self.final_top_k

        candidates = self._ensemble_retriever.invoke(query)
        return self._rerank(query, candidates, top_k)

    def _rerank(
        self,
        query: str,
        candidates: list[Document],
        top_k: int,
    ) -> list[Document]:
        unique_candidates = self._deduplicate_documents(candidates)
        if not unique_candidates:
            return []

        pairs = [[query, d.page_content] for d in unique_candidates]
        scores = self.reranker.predict(pairs)
        ranked = sorted(
            zip(unique_candidates, scores),
            key=lambda item: item[1],
            reverse=True,
        )
        return [doc for doc, _ in ranked[:top_k]]

    def query_expansion_retrieval(
        self, original_query: str, llm, subqueries: int = 3
    ) -> list[Document]:
        """查询改写：让 LLM 生成多个子查询，多路召回后去重合并。"""
        queries = self._rewrite_query(original_query, llm, subqueries)
        queries = [original_query] + queries

        seen: set[str] = set()
        merged: list[Document] = []
        for q in queries:
            if not self._ensemble_retriever:
                break
            for doc in self._ensemble_retriever.invoke(q):
                cid = self._document_id(doc)
                if cid not in seen:
                    seen.add(cid)
                    merged.append(doc)
        return self._rerank(
            original_query,
            merged,
            top_k=max(self.final_top_k, 5),
        )

    # ---- 工具方法 ----

    @property
    def reranker(self) -> CrossEncoder:
        if self._reranker is None:
            allow_download = os.environ.get("RAG_RERANKER_ALLOW_DOWNLOAD") == "1"
            try:
                self._reranker = CrossEncoder(
                    self.reranker_model,
                    local_files_only=not allow_download,
                )
            except Exception as exc:
                if (
                    not self.reranker_fallback_model
                    or self.reranker_fallback_model == self.reranker_model
                ):
                    raise
                print(
                    f"[retriever] failed to load {self.reranker_model}: {exc}; "
                    f"falling back to {self.reranker_fallback_model}"
                )
                self._reranker = CrossEncoder(
                    self.reranker_fallback_model,
                    local_files_only=True,
                )
        return self._reranker

    def _rewrite_query(self, query: str, llm, n: int) -> list[str]:
        try:
            prompt = _REWRITE_PROMPT.format(query=query, n=n)
            resp = llm.invoke(prompt).strip()
            rewritten = []
            for line in resp.splitlines():
                cleaned = re.sub(r"^\s*\d+\s*[.)、:：-]?\s*", "", line).strip()
                if cleaned and cleaned != query and cleaned not in rewritten:
                    rewritten.append(cleaned)
            return rewritten[:n]
        except Exception as e:
            print(f"[retriever] query rewrite failed, fallback to original: {e}")
            return []

    def _add_to_vector_store(self, documents: list[Document]) -> None:
        if not self._vector_store:
            raise RuntimeError("vector store not initialized")
        self._vector_store.add_documents(
            documents=documents,
            ids=[self._document_id(doc) for doc in documents],
        )

    @staticmethod
    def _document_id(document: Document) -> str:
        if document.id:
            return document.id
        metadata_id = document.metadata.get("chunk_id")
        if metadata_id:
            return str(metadata_id)
        source = document.metadata.get("document_name") or document.metadata.get(
            "source", "unknown"
        )
        digest = hashlib.sha1(
            f"{source}|{document.page_content}".encode("utf-8")
        ).hexdigest()
        return f"{source}#{digest}"

    @staticmethod
    def _document_keys(document: Document) -> set[str]:
        keys = set()
        document_name = document.metadata.get("document_name")
        source = document.metadata.get("source")
        if document_name:
            keys.add(str(document_name))
        if source:
            normalized = str(source).replace("\\", "/")
            keys.add(normalized)
            keys.add(os.path.basename(normalized))
        return keys

    def _deduplicate_documents(self, documents: list[Document]) -> list[Document]:
        unique: dict[str, Document] = {}
        for document in documents:
            unique[self._document_id(document)] = document
        return list(unique.values())

    @property
    def has_index(self) -> bool:
        return bool(self._all_chunks and self._ensemble_retriever)

    @property
    def chunk_count(self) -> int:
        return len(self._all_chunks)

    @property
    def document_names(self) -> list[str]:
        names = set()
        for chunk in self._all_chunks:
            document_name = chunk.metadata.get("document_name")
            source = chunk.metadata.get("source")
            if document_name:
                names.add(str(document_name))
            elif source:
                names.add(os.path.basename(str(source)))
        return sorted(names)
