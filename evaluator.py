"""RAG 系统评估：检索层用 Hit Rate / MRR，生成层用 RAGAS 四件套。"""

from __future__ import annotations

import importlib
import sys
import types
from typing import Optional

import pandas as pd
from datasets import Dataset

from rag_utils import retrieval_metrics


def _ensure_ragas_import_compat() -> None:
    """Bridge a missing legacy import used by ragas 0.4.x."""
    module_name = "langchain_community.chat_models.vertexai"
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise
        module = types.ModuleType(module_name)

        class ChatVertexAI:
            """Compatibility placeholder; Ollama does not use Vertex AI."""

        module.ChatVertexAI = ChatVertexAI
        sys.modules[module_name] = module


class RAGEvaluator:
    def __init__(self, llm=None, embeddings=None):
        self.llm = llm
        self.embeddings = embeddings

    def prepare_evaluation_data(
        self,
        questions: list[str],
        answers: list[str],
        contexts: list[list[str]],
        ground_truths: Optional[list[str]] = None,
    ) -> Dataset:
        if len(questions) != len(answers) or len(questions) != len(contexts):
            raise ValueError("questions / answers / contexts length mismatch")
        if ground_truths is None:
            ground_truths = [""] * len(questions)
        if len(ground_truths) != len(questions):
            raise ValueError("questions / ground_truths length mismatch")
        return Dataset.from_dict(
            {
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truth": ground_truths,
            }
        )

    def run_evaluation(self, dataset: Dataset) -> dict:
        _ensure_ragas_import_compat()
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        metrics = [context_precision, context_recall, faithfulness, answer_relevancy]
        kwargs = {
            "dataset": dataset,
            "metrics": metrics,
            "raise_exceptions": True,
        }
        if self.llm is not None:
            kwargs["llm"] = self.llm
        if self.embeddings is not None:
            kwargs["embeddings"] = self.embeddings
        results = evaluate(**kwargs)
        print("[eval] RAGAS done")
        return results

    def evaluate_retrieval_quality(
        self, retriever, test_queries: list[dict], top_k: int = 3
    ) -> pd.DataFrame:
        results = []
        for item in test_queries:
            query = item["query"]
            expected_ids = set(item.get("expected_chunk_ids", []))
            if not expected_ids:
                continue

            retrieved = retriever.retrieve_with_rerank(query, top_k=top_k)
            retrieved_ids = [d.metadata.get("chunk_id") for d in retrieved]

            metrics = retrieval_metrics(
                retrieved_ids,
                expected_ids,
                k=top_k,
            )
            results.append({"query": query, **metrics})

        df = pd.DataFrame(results)
        if not df.empty:
            print(
                f"[eval] retrieval: hit_rate={df['hit_rate'].mean():.3f} "
                f"recall@{top_k}={df['recall_at_k'].mean():.3f} "
                f"mrr={df['mrr'].mean():.3f} (n={len(df)})"
            )
        return df
