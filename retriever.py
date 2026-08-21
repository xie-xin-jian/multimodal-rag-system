"""混合检索器：BM25 + Dense 召回，CrossEncoder 精排。

继承 BaseRetriever 以便直接接入 LangChain 的 RetrievalQA / LCEL 链路，
不必在 main 里再绕一层。
"""
from __future__ import annotations

import os
from typing import Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from pydantic import PrivateAttr
from sentence_transformers import CrossEncoder

_REWRITE_PROMPT = """将下面的问题改写成 {n} 个语义等价但表达不同的检索子问题，每行一条，不要编号、不要解释。
原问题：{query}
子问题："""


class HybridRetriever(BaseRetriever):
    persist_directory: str = "chroma_db"
    bm25_weight: float = 0.5
    dense_weight: float = 0.5
    candidate_k: int = 10
    final_top_k: int = 3

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

        self._all_chunks = list(chunks)
        self._vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=self.persist_directory,
        )
        self._init_bm25_and_ensemble()
        print(f"[retriever] index built: {len(chunks)} chunks -> {self.persist_directory}")

    def load_index(self, embeddings) -> bool:
        if not os.path.exists(self.persist_directory):
            return False
        try:
            self._vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=embeddings,
            )
            raw = self._vector_store.get(include=["documents", "metadatas"])
            chunks = []
            for page_content, metadata in zip(raw["documents"], raw["metadatas"]):
                chunks.append(Document(page_content=page_content, metadata=metadata))
            if not chunks:
                return False
            self._all_chunks = chunks
            self._init_bm25_and_ensemble()
            print(f"[retriever] index loaded: {len(chunks)} chunks <- {self.persist_directory}")
            return True
        except Exception as e:
            print(f"[retriever] failed to load index: {e}")
            return False

    def _init_bm25_and_ensemble(self) -> None:
        self._bm25_retriever = BM25Retriever.from_documents(self._all_chunks)
        self._bm25_retriever.k = self.candidate_k
        self._rebuild_ensemble()

    def add_documents(self, new_chunks: list[Document], embeddings) -> None:
        if not self._vector_store:
            raise RuntimeError("index not built, call build_index first")
        if not new_chunks:
            return

        # 向量库支持增量；BM25 索引重建代价低，直接整体重建保持一致
        self._vector_store.add_documents(new_chunks)
        self._all_chunks.extend(new_chunks)
        self._bm25_retriever = BM25Retriever.from_documents(self._all_chunks)
        self._bm25_retriever.k = self.candidate_k
        self._rebuild_ensemble()
        print(f"[retriever] incremental update: +{len(new_chunks)} (total={len(self._all_chunks)})")

    def remove_document(self, filename: str) -> int:
        """从知识库中删除指定文件的所有 chunk，返回删除数量。"""
        if not self._vector_store:
            raise RuntimeError("index not built")

        to_remove = [
            c for c in self._all_chunks
            if os.path.basename(c.metadata.get("source", "")) == filename
        ]

        abs_path = os.path.join(
            os.path.dirname(self.persist_directory), "data", filename
        )

        if not to_remove:
            if os.path.exists(abs_path):
                os.remove(abs_path)
            return 0

        source_paths = list({c.metadata.get("source", "") for c in to_remove})
        for sp in source_paths:
            for variant in {sp, sp.replace("\\", "/")}:
                try:
                    matched = self._vector_store._collection.get(where={"source": variant})
                    if matched["ids"]:
                        self._vector_store._collection.delete(ids=matched["ids"])
                        break
                except Exception:
                    continue

        self._all_chunks = [
            c for c in self._all_chunks
            if os.path.basename(c.metadata.get("source", "")) != filename
        ]
        self._init_bm25_and_ensemble()

        if os.path.exists(abs_path):
            os.remove(abs_path)

        print(f"[retriever] removed {filename}: -{len(to_remove)} chunks (total={len(self._all_chunks)})")
        return len(to_remove)

    def _rebuild_ensemble(self) -> None:
        dense = self._vector_store.as_retriever(
            search_kwargs={"k": self.candidate_k}
        )
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

    def retrieve_with_rerank(self, query: str, top_k: Optional[int] = None) -> list[Document]:
        if not self._ensemble_retriever:
            raise RuntimeError("retriever not initialized")
        top_k = top_k or self.final_top_k

        candidates = self._ensemble_retriever.invoke(query)
        if not candidates:
            return []

        # TODO: hybrid 权重目前固定，后续可基于评估集学习 bm25_weight/dense_weight
        pairs = [[query, d.page_content] for d in candidates]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
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
            for doc in self.retrieve_with_rerank(q):
                cid = doc.metadata.get("chunk_id", id(doc))
                if cid not in seen:
                    seen.add(cid)
                    merged.append(doc)
        return merged[: max(self.final_top_k, 5)]

    # ---- 工具方法 ----

    @property
    def reranker(self) -> CrossEncoder:
        if self._reranker is None:
            self._reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        return self._reranker

    def _rewrite_query(self, query: str, llm, n: int) -> list[str]:
        try:
            prompt = _REWRITE_PROMPT.format(query=query, n=n)
            resp = llm.invoke(prompt).strip()
            return [line.strip() for line in resp.splitlines() if line.strip()][:n]
        except Exception as e:
            print(f"[retriever] query rewrite failed, fallback to original: {e}")
            return []
