"""RAG 系统评估：检索层用 Hit Rate / MRR，生成层用 RAGAS 四件套。"""
from __future__ import annotations

from typing import Optional

import pandas as pd
from datasets import Dataset


class RAGEvaluator:
    def __init__(self, llm=None):
        self.llm = llm

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
        return Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })

    def run_evaluation(self, dataset: Dataset) -> dict:
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        metrics = [context_precision, context_recall, faithfulness, answer_relevancy]
        kwargs = {"dataset": dataset, "metrics": metrics}
        if self.llm is not None:
            kwargs["llm"] = self.llm
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

            hits = len(set(retrieved_ids) & expected_ids)
            hit_rate = hits / len(expected_ids)

            reciprocal_rank = 0.0
            for rank, rid in enumerate(retrieved_ids, start=1):
                if rid in expected_ids:
                    reciprocal_rank = 1.0 / rank
                    break

            results.append({
                "query": query,
                "hit_rate": hit_rate,
                "mrr": reciprocal_rank,
                "retrieved": retrieved_ids,
            })

        df = pd.DataFrame(results)
        if not df.empty:
            print(
                f"[eval] retrieval: hit_rate={df['hit_rate'].mean():.3f} "
                f"mrr={df['mrr'].mean():.3f} (n={len(df)})"
            )
        return df
