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
- 每个 chunk 绑定 `source / page / chunk_id / content_hash` 元数据；Web 上传使用逻辑文档名生成稳定 `chunk_id`，重复上传同一文档会替换旧 chunk。

### 混合检索与 Rerank 精排

- **召回层**：BM25（稀疏）+ Dense Vector（稠密，Chroma）通过 `EnsembleRetriever` 加权融合，权重可配置；
- **精排层**：多语言 Cross-Encoder (`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`) 对候选 chunk 重新打分，取 Top-K；
- 默认只读取本地模型缓存；若主模型不可用，会回退到已缓存的 `ms-marco-MiniLM-L-6-v2`，设置 `RAG_RERANKER_ALLOW_DOWNLOAD=1` 可允许在线下载主模型；
- **查询改写**：LLM 将原问题改写为多个语义等价的子查询，合并全部候选并按 `chunk_id` 去重后，使用原始问题统一精排；
- **中文稀疏检索**：BM25 对中文使用字符、二元和三元 token，英文及代码标识符保持完整 token；
- `HybridRetriever` 继承 `BaseRetriever`，可直接接入 LangChain 的 LCEL / Retrieval 链路。

### 增量索引

- 向量库使用稳定的 chunk ID 进行 upsert，同一逻辑文档重新上传时先替换旧 chunk；
- BM25 索引在新增文档后整体重建（in-memory，重建代价可接受），保证稀疏检索与稠密检索的视图一致。

### 基于 RAGAS 的可量化评估

- **检索层**：Hit Rate、MRR（Mean Reciprocal Rank）；
- **生成层**：RAGAS 四件套 —— `context_precision` / `context_recall` / `faithfulness` / `answer_relevancy`，复用本地 Ollama embeddings；
- 评估失败直接抛出异常，不返回兜底假分数。

## 项目结构

```
多模态RAG知识库问答系统/
├── data_loader.py      # 多模态文档加载、清洗、切分
├── retriever.py        # HybridRetriever：BM25+Dense+Rerank+查询改写
├── evaluator.py        # RAGAS / Hit Rate / MRR 评估
├── main.py             # 系统主入口 MiniRAGSystem
├── app.py              # Flask Web UI 与 REST API
├── rag_utils.py        # 中文 BM25 分词、路径安全与检索指标
├── tests/              # 不依赖模型的单元测试
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

## Web 启动

```bash
python app.py
```

默认访问地址为 `http://127.0.0.1:5000`。可通过环境变量覆盖：

```bash
RAG_EMBED_MODEL=nomic-embed-text
RAG_LLM_MODEL=qwen2.5
RAG_PERSIST_DIR=chroma_db
RAG_HOST=127.0.0.1
RAG_PORT=5000
RAG_RERANKER_ALLOW_DOWNLOAD=0
```

## 测试

```bash
python -m unittest discover -s tests -v
```

当前测试覆盖中文 BM25 分词、安全文件路径、标准 Hit Rate / Recall / MRR 计算，以及 Chroma 索引的构建、替换、删除和重新加载。

## 设计取舍

- **为什么混合检索？** 稠密向量擅长语义近似（"乔布斯创立的科技公司" ↔ "苹果公司"），BM25 擅长精确关键词命中（API 名、错误码）。两者互补，单路都会有盲区。
- **为什么需要 Rerank？** 召回阶段优先保证 Recall（候选 10+），Cross-Encoder 用更重的双塔交互模型对候选精排，提升最终 Top-K 的 Precision。
- **为什么 BM25 重建而不是增量？** `rank_bm25` 的 IDF 统计依赖全量文档，增量更新需要重算 IDF，整体重建比维护增量更简单且代价可接受。
