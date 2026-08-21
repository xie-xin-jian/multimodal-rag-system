"""多模态RAG知识库问答系统主入口。

链路：多模态加载 -> 切分 -> 混合索引(BM25+Dense) -> 查询改写 -> Rerank -> LLM 生成。
"""
from __future__ import annotations

import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from langchain_ollama import OllamaEmbeddings, OllamaLLM

from data_loader import MultiModalDataLoader
from retriever import HybridRetriever
from evaluator import RAGEvaluator


_ANSWER_PROMPT = """你是基于知识库的问答助手。请只依据下面给出的参考资料回答问题；
资料中没有的信息不要编造，回答末尾用 [来源: 文件名#chunk序号] 标注引用。

参考资料：
{context}

问题：{question}

回答："""


class MiniRAGSystem:
    def __init__(
        self,
        embed_model: str = "nomic-embed-text",
        llm_model: str = "qwen2.5",
        persist_directory: str = "chroma_db",
    ):
        self.embeddings = OllamaEmbeddings(model=embed_model)
        self.llm = OllamaLLM(model=llm_model)
        self.data_loader = MultiModalDataLoader(chunk_size=500, chunk_overlap=50)
        self.retriever = HybridRetriever(persist_directory=persist_directory)
        self.evaluator = RAGEvaluator(llm=self.llm)
        self._loaded = self.retriever.load_index(self.embeddings)

    # ---- 索引 ----

    def ingest_knowledge(self, file_paths: list[str]) -> None:
        all_chunks = []
        for path in file_paths:
            if not os.path.exists(path):
                print(f"[ingest] skip, not found: {path}")
                continue
            docs = self.data_loader.load_file(path)
            chunks = self.data_loader.process_documents(docs)
            all_chunks.extend(chunks)
            print(f"[ingest] {path} -> {len(chunks)} chunks")

        if not all_chunks:
            raise RuntimeError("no documents ingested")

        self.retriever.build_index(all_chunks, self.embeddings)
        self._loaded = True
        print(f"[ingest] done, total {len(all_chunks)} chunks indexed")

    def add_documents(self, file_paths: list[str]) -> None:
        new_chunks = []
        for path in file_paths:
            if not os.path.exists(path):
                continue
            docs = self.data_loader.load_file(path)
            new_chunks.extend(self.data_loader.process_documents(docs))
        if new_chunks:
            self.retriever.add_documents(new_chunks, self.embeddings)

    def remove_document(self, filename: str) -> int:
        """删除知识库里的指定文档，返回删除的 chunk 数量。"""
        return self.retriever.remove_document(filename)

    # ---- 问答 ----

    def ask(self, query: str, use_query_expansion: bool = True) -> dict:
        if use_query_expansion:
            docs = self.retriever.query_expansion_retrieval(query, self.llm)
        else:
            docs = self.retriever.retrieve_with_rerank(query)

        if not docs:
            return {
                "query": query,
                "answer": f"未在知识库中检索到与 '{query}' 相关的内容。",
                "sources": [],
                "contexts_used": 0,
            }

        context = "\n\n".join(d.page_content for d in docs)
        sources = sorted({
            f"{os.path.basename(d.metadata.get('source', ''))}#{d.metadata.get('chunk_index', '')}"
            for d in docs
        })

        prompt = _ANSWER_PROMPT.format(context=context, question=query)
        answer = self.llm.invoke(prompt).strip()

        return {
            "query": query,
            "answer": answer,
            "sources": sources,
            "contexts_used": len(docs),
        }

    # ---- 评估 ----

    def run_evaluation(self, test_data: list[dict]) -> dict:
        print("\n" + "=" * 40)
        print("RAG System Evaluation")
        print("=" * 40)

        # 检索层：Hit Rate / MRR
        retrieval_df = self.evaluator.evaluate_retrieval_quality(
            self.retriever, test_data
        )

        # 生成层：RAGAS 四件套
        questions, answers, contexts, ground_truths = [], [], [], []
        for item in test_data:
            q = item["query"]
            docs = self.retriever.retrieve_with_rerank(q)
            res = self.ask(q, use_query_expansion=False)
            questions.append(q)
            answers.append(res["answer"])
            contexts.append([d.page_content for d in docs])
            ground_truths.append(item.get("ground_truth", ""))

        dataset = self.evaluator.prepare_evaluation_data(
            questions, answers, contexts, ground_truths
        )
        ragas_results = self.evaluator.run_evaluation(dataset)
        return {"retrieval": retrieval_df, "ragas": ragas_results}


if __name__ == "__main__":
    system = MiniRAGSystem()
    print("多模态RAG知识库问答系统 ready. See README for usage.")
