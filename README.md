# Mini RAG System

基于 LangChain + Chroma 构建的多模态 RAG（Retrieval-Augmented Generation）系统。
核心是一套 **混合检索 + Cross-Encoder 精排** 的召回框架，针对传统 RAG 在复杂文档解析、
长文本上下文丢失、检索精度不足等场景做了工程化处理。

## 技术栈

Python / LangChain / Chroma / sentence-transformers (Cross-Encoder) / RAGAS / pdfplumber / pytesseract / Ollama

## 核心设计

### 多模态文档解析与清洗

- PDF 文本提取走 `pdfplumber`，对单页可提取文本低于阈值的扫描页自动渲染为图片做 OCR 兜底；
- 页内嵌入图片按坐标裁剪后单独 OCR，文本与图片信息合并入 chunk；
- 切分前做清洗：去除行尾空白、孤立页码行、连续空行；
- 每个 chunk 绑定 `source / page / chunk_id / content_hash` 元数据，`chunk_id` 跨会话稳定，支持增量去重。

### 混合检索与 Rerank 精排

- **召回层**：BM25（稀疏）+ Dense Vector（稠密，Chroma）通过 `EnsembleRetriever` 加权融合，权重可配置；
- **精排层**：Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) 对候选 chunk 重新打分，取 Top-K；
- **查询改写**：LLM 将原问题改写为多个语义等价的子查询，多路召回后按 `chunk_id` 去重合并；
- `HybridRetriever` 继承 `BaseRetriever`，可直接接入 LangChain 的 LCEL / Retrieval 链路。

### 增量索引

- 向量库走 Chroma 的 `add_documents` 增量接口；
- BM25 索引在新增文档后整体重建（in-memory，重建代价可接受），保证稀疏检索与稠密检索的视图一致。

### 基于 RAGAS 的可量化评估

- **检索层**：Hit Rate、MRR（Mean Reciprocal Rank）；
- **生成层**：RAGAS 四件套 —— `context_precision` / `context_recall` / `faithfulness` / `answer_relevancy`；
- 评估失败直接抛出异常，不返回兜底假分数。

## 项目结构

```
多模态RAG知识库问答系统/
├── data_loader.py      # 多模态文档加载、清洗、切分
├── retriever.py        # HybridRetriever：BM25+Dense+Rerank+查询改写
├── evaluator.py        # RAGAS / Hit Rate / MRR 评估
├── main.py             # 系统主入口 MiniRAGSystem
├── requirements.txt
└── README.md
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 准备本地模型（Ollama）
#    安装：https://ollama.ai/
ollama pull nomic-embed-text
ollama pull qwen2.5

# 3. 准备 OCR 语言包（用于扫描版 PDF）
#    安装 tesseract，并下载 chi_sim 语言包到 TESSDATA_PREFIX 目录

# 4. 使用
python
```

```python
from main import MiniRAGSystem

system = MiniRAGSystem()

# 建立知识库
system.ingest_knowledge(["data/docs.pdf", "data/notes.md"])

# 增量追加文档
system.add_documents(["data/new.pdf"])

# 提问（默认走查询改写 + Rerank）
res = system.ask("这个项目的核心模块有哪些？")
print(res["answer"])
print("sources:", res["sources"])

# 关闭查询改写，仅走单路混合检索 + Rerank
res = system.ask("API 端口配置在哪？", use_query_expansion=False)

# 评估
test_data = [
    {"query": "如何配置鉴权？", "expected_chunk_ids": ["data/docs.pdf#3"], "ground_truth": "..."},
    {"query": "默认端口是多少？", "expected_chunk_ids": ["data/docs.pdf#7"], "ground_truth": "..."},
]
report = system.run_evaluation(test_data)
print(report["retrieval"])
print(report["ragas"])
```

## 设计取舍

- **为什么混合检索？** 稠密向量擅长语义近似（"乔布斯创立的科技公司" ↔ "苹果公司"），BM25 擅长精确关键词命中（API 名、错误码）。两者互补，单路都会有盲区。
- **为什么需要 Rerank？** 召回阶段优先保证 Recall（候选 10+），Cross-Encoder 用更重的双塔交互模型对候选精排，提升最终 Top-K 的 Precision。
- **为什么 BM25 重建而不是增量？** `rank_bm25` 的 IDF 统计依赖全量文档，增量更新需要重算 IDF，整体重建比维护增量更简单且代价可接受。
