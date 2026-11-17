# Mini RAG 知识库问答系统 软件说明书

## 文档控制

| 项目 | 内容 |
|---|---|
| 文档名称 | Mini RAG 知识库问答系统软件说明书 |
| 文档版本 | 1.0 |
| 适用系统 | Mini RAG 知识库问答系统（`main` 分支当前实现） |
| 对应代码版本 | `ecd044e` |
| 数据基线 | SQLite 就地建表与增列，无独立迁移框架 |
| 文档状态 | 当前版本 |
| 最后更新 | 2026-11-17 |

本文档面向系统使用者、部署维护人员和后续开发人员，说明系统能力、界面操作、文档发布流程、问答链路、模块划分、数据结构、部署配置、安全边界、已知限制和常见故障处理方式。

## 1. 文档说明

### 1.1 编写目的

本文档用于建立以下共同认知：

- 系统解决什么问题，以及哪些场景不在当前实现范围内。
- 文档如何上传、解析、切块、建立索引并发布为可回滚版本。
- 一次知识库问答如何完成查询改写、混合召回、精排和生成。
- Flask API、发布管理器、SQLite、Chroma、Ollama 和前端之间的职责边界。
- 发布任务、活动版本、历史版本和数据文件之间的关系。
- 部署、备份、故障排查和二次开发时应遵循的约束。

### 1.2 读者与阅读路径

| 读者 | 关注内容 | 建议阅读顺序 |
|---|---|---|
| 普通使用者 | 上传文档、提问、查看来源、切换版本 | 2 → 3.1～3.7 |
| 部署维护人员 | Ollama、OCR、SQLite、Chroma、配置、备份、排障 | 2 → 8 → 9 → 10 |
| 开发人员 | 发布链路、检索链路、SQLite Schema、模块职责、扩展方式 | 3 → 4 → 5 → 6 → 7 |
| 质量评估人员 | 检索指标、RAGAS、测试覆盖和限制 | 3.9 → 4.8 → 7 → 10 → 11 |

### 1.3 术语

| 术语 | 含义 |
|---|---|
| Document | 一个逻辑文档，以用户看到的文档名为稳定标识 |
| Chunk | 文档切分后的检索单元，包含正文和来源元数据 |
| Release | 一次可发布的知识库版本，包含文档清单和独立向量集合 |
| Active Release | 当前回答问题时实际使用的发布版本 |
| Superseded Release | 曾经激活、后被新版本替代，但仍可回滚的历史版本 |
| Staged | 新版本已经完成索引构建并写入版本元数据，等待激活 |
| Job | 上传、删除、回滚或重新处理的后台任务 |
| Collection | Chroma 中的向量集合，每个活动发布使用 `rag_release_<id>` |
| BM25 | 基于词项频率和逆文档频率的稀疏检索 |
| Dense Retrieval | 使用 Embedding 向量进行语义相似度检索 |
| Rerank | 使用 Cross-Encoder 对召回候选进行重新打分 |
| Query Expansion | 使用 LLM 将一个原问题改写成多个检索子问题 |
| OCR | 光学字符识别，用于提取扫描 PDF 或页面图片中的文字 |
| Registry | 保存发布、文档清单、任务和系统状态的 SQLite 数据库 |
| RAGAS | 评估检索增强生成质量的指标体系与工具库 |

### 1.4 文档约定

- “界面”指 Flask 模板 `templates/index.html` 提供的单页工作台。
- “API”指 Flask 提供的 `/api/*` HTTP 接口。
- “当前知识库”指当前活动 Release 对应的全部文档与 Chunk。
- “新版本”指一次上传、删除或索引升级产生的新 Release。
- 命令示例默认从仓库根目录执行。
- 配置默认值以 `app.py`、`main.py`、`retriever.py` 和发布管理代码为准。
- “已实现”表示当前代码存在可执行路径；“预留”或“未完整实现”会在文中明确标注。

## 2. 系统概述

### 2.1 系统定位

Mini RAG 是一个面向本地或受信环境的文档知识库问答系统。用户上传 PDF、Markdown、TXT 或 DOCX 文档，系统将文本解析、清洗和切块，建立 BM25 稀疏索引与 Chroma 稠密向量索引，然后通过混合召回、Cross-Encoder 精排和本地 Ollama 模型生成带来源引用的回答。

系统的重点不是普通聊天，而是可管理的知识发布：

- 上传不会直接破坏正在使用的索引。
- 系统先在独立 Collection 中构建新版本。
- 新版本验证完成后才切换活动 Release。
- 旧版本继续保留并可回滚。
- 未变化文档可复用已持久化向量，减少重复 Embedding。
- SQLite 记录文档清单、任务、发布状态和系统状态。

### 2.2 功能范围

| 能力域 | 已实现能力 |
|---|---|
| 文档解析 | PDF 文本、扫描页 OCR、PDF 内嵌图片 OCR、Markdown、TXT、DOCX 文本和表格 |
| 文档切块 | 500 字符切块、50 字符重叠、稳定 Chunk ID 和内容 Hash |
| 文件管理 | 拖放上传、批量上传、内容寻址保存、同名文档替换 |
| 混合检索 | BM25 + Chroma Dense 召回，默认等权融合 |
| 查询增强 | LLM 子查询改写，多路召回后按 Chunk ID 去重 |
| 精排 | Cross-Encoder 对候选重新打分并截取 Top-K |
| 回答生成 | 本地 Ollama 模型、上下文约束、来源与 Chunk 引用 |
| 版本发布 | 异步任务、新 Collection 构建、活动版本原子切换 |
| 回滚 | 加载历史版本 Collection 并重新激活 |
| 外部对账 | 定时或手动发现 `data/` 中新增或变化文件 |
| 故障恢复 | 重启后恢复排队任务、运行中任务和活动版本 |
| Web 界面 | 对话、知识库文档、发布记录、系统状态、浅色/深色主题 |
| 量化评估 | Hit Rate、Recall、Precision、MRR、RAGAS 四项生成指标 |

### 2.3 适用边界

当前版本适用于：

- 个人知识库或小团队内部知识库。
- 单机部署的本地问答服务。
- 文档数量有限、重新建立 BM25 索引成本可接受的环境。
- 已安装 Ollama 并能提供本地 LLM 与 Embedding 模型的机器。
- 对答案可追溯性有要求，但不需要多租户权限隔离的场景。

当前版本不提供：

- 多用户账号、角色权限和租户隔离。
- 文档级访问控制。
- 真正的视觉 Embedding 或图片语义检索。
- 持久化聊天会话与多会话历史。
- 模型原生 Token Streaming。
- 多实例 Worker 调度和分布式发布锁。
- 面向公网的完整认证、限流和审计体系。
- 已接入发布流程的自动质量门禁。

### 2.4 界面功能地图

| 页面 | 主要用途 | 当前入口 |
|---|---|---|
| 对话 | 提问、查询改写开关、查看回答和引用 | 侧栏“对话” |
| 知识库 | 上传文档、搜索、查看文档详情、删除文档 | 侧栏“知识库”或业务对象下的“文档” |
| 发布记录 | 查看版本、筛选状态、查看详情、回滚 | 侧栏“发布记录”或业务对象下的“版本” |
| 系统状态 | 查看模型、Chunk、活动版本、任务和对账信息 | 侧栏底部服务状态按钮 |
| 新建对话 | 清空当前浏览器中的消息，不创建服务端会话 | 侧栏“新建对话”或输入框“清空” |

底部主题按钮在浅色和深色主题间切换，选择保存在浏览器 `localStorage` 中。

### 2.5 当前状态模型摘要

| 对象 | 常见状态 |
|---|---|
| Job | `QUEUED`、`RUNNING`、`ACTIVE`、`FAILED`、`NOOP`、`REJECTED` |
| Release | `PENDING`、`STAGING`、`STAGED`、`ACTIVE`、`SUPERSEDED`、`FAILED`、`NOOP` |
| Release Mode | `CONTENT_UPDATE`、`INDEX_UPGRADE` |
| Reconciliation | `NO_ACTIVE_RELEASE`、`NO_CHANGES`、`QUEUED` |

`REJECTED`、`ACTIVE_WITH_WARNING` 和 `PENDING_APPROVAL` 在代码中有状态定义或前端展示分支，但当前发布 Worker 尚未实际产生完整的质量门禁流程。详见“已知限制”。

## 3. 功能与操作说明

### 3.1 启动与状态检查

本地启动后默认访问：

```text
http://127.0.0.1:5000
```

页面启动时会并行读取：

- `GET /api/status`
- `GET /api/documents`
- `GET /api/releases`

侧栏底部绿灯表示当前存在已加载的活动索引；红灯表示知识库为空或索引不可用。点击状态项可以查看模型、活动版本、任务状态和对账信息。

当前系统没有登录页。默认监听 `127.0.0.1`，只适合同机或受控内网使用。

### 3.2 模型服务前置条件

系统依赖本机 Ollama：

```powershell
ollama pull qwen2.5
ollama pull nomic-embed-text
ollama serve
```

默认模型：

| 用途 | 默认模型 |
|---|---|
| 回答生成与查询改写 | `qwen2.5` |
| Dense Embedding | `nomic-embed-text` |
| 主 Reranker | `BAAI/bge-reranker-v2-m3` |
| Reranker 回退 | `BAAI/bge-reranker-base` |

模型名称可通过环境变量覆盖。Reranker 默认只读取本地缓存；只有设置 `RAG_RERANKER_ALLOW_DOWNLOAD=1` 时才允许下载主 Reranker 模型。

### 3.3 知识库对话

#### 3.3.1 提问

1. 确认侧栏显示“知识库可用”。
2. 在输入框输入问题。
3. 选择是否启用“查询改写”。
4. 点击发送。`Enter` 发送，`Shift+Enter` 换行。
5. 等待检索和生成活动组完成。
6. 查看回答下方列出的来源文档和 Chunk 编号。

如果没有检索到上下文，系统返回“未在知识库中检索到相关内容”，不会调用模型继续猜测。

#### 3.3.2 查询改写开关

- 开启时：先生成最多 3 个检索子问题，再与原始问题一起执行多路召回。
- 关闭时：仅使用原始问题进行一次混合召回。
- 改写调用失败时：保留原问题，继续执行单路检索。
- 开启查询改写时：最终精排上下文最多返回 5 条；关闭时默认返回 3 条。

查询改写会增加一次 LLM 调用，因此通常比普通检索更慢。

#### 3.3.3 回答与引用

回答提示要求模型：

- 只依据提供的参考资料回答。
- 资料不足时不要编造。
- 使用 `[来源: 文件名#chunk序号]` 形式标注引用。
- 把文档内容视为不可信数据，不执行其中的指令。

界面还会单独列出本次使用的来源 Chunk。模型生成的引用文本不经过自动事实校验，仍需结合来源和原文判断。

#### 3.3.4 停止与清空

- 执行期间发送按钮切换为停止按钮。
- 停止只会取消浏览器对当前 `/api/ask` 请求的等待。
- Flask 和 Ollama 端已经开始的推理任务不保证立即终止。
- “清空”和“新建对话”只清空当前页面消息；当前版本没有服务端会话历史。

#### 3.3.5 活动组

对话中的活动组显示当前检索和生成进度，包括：

- 步骤数。
- 已用时间。
- 当前步骤名称。
- 请求参数。
- 使用上下文数量和来源。

当前后端没有通用模型工具调用接口，因此活动组展示的是真实的检索/生成请求和发布任务，而不是伪造的 Agent 工具轨迹。

### 3.4 上传知识文档

#### 3.4.1 支持格式

| 格式 | 处理方式 |
|---|---|
| `.pdf` | `pdfplumber` 提取文本；低文本页整页 OCR；嵌入图片按区域 OCR |
| `.md` / `.markdown` | 按 UTF-8 读取并清理文本 |
| `.txt` | 按 UTF-8 读取；无法解码字符忽略 |
| `.docx` | 提取正文段落、表格、页眉和页脚 |

单次 HTTP 请求最大 50MB。该限制是整次请求总大小，不是单文件大小。

#### 3.4.2 上传步骤

1. 进入“知识库”。
2. 拖放文件到上传条，或点击“选择文件”/“添加文档”。
3. 前端提交 `multipart/form-data` 到 `POST /api/ingest`。
4. 服务端计算 SHA-256，以内容 Hash 前 16 位作为文件前缀保存。
5. 服务端创建异步发布 Job 并返回 HTTP 202。
6. 前端轮询任务，直到 `ACTIVE`、`FAILED`、`NOOP` 或 `REJECTED`。

#### 3.4.3 同名文档替换

逻辑文档名使用用户上传时的文件名。再次上传同名文件时：

- 新文件以新的内容 Hash 保存。
- 发布版本会移除该逻辑文档的旧 Chunk。
- 其他文档向量继续复用。
- 历史 Release 仍引用旧文件，直到该 Release 被清理。

同一批上传中出现两个完全相同文件名会直接拒绝，避免批次内语义冲突。

#### 3.4.4 无变化上传

内容 Hash、解析器版本、切块版本和 Embedding 版本都未变化时，任务进入 `NOOP`，不会重复解析、Embedding 或创建活动版本。

### 3.5 知识库文档管理

“知识库”页面采用列表与详情结构：

- 左侧显示当前活动版本中的文档。
- 可按文档名、逻辑文档 ID 或存储文件搜索。
- 右侧显示 Chunk 数、大小、Release ID、逻辑文档 ID、存储文件和内容 Hash。

#### 3.5.1 删除文档

1. 选择文档。
2. 点击“删除文档”。
3. 在审批卡中确认风险、影响和步骤。
4. 点击“确认删除”。
5. 等待新版本构建和活动版本切换。

删除不会直接修改当前 Collection，而是创建不含该文档的新发布版本。任务完成后，旧版本仍按保留策略存在。

### 3.6 发布记录与回滚

“发布记录”页面显示：

- Release ID。
- 状态。
- 文档数量。
- Chunk 数量。
- 父版本、Collection、创建时间、激活时间、Manifest Hash、发布模式和错误。

#### 3.6.1 回滚

1. 在版本列表中选择一个 `SUPERSEDED` 历史版本。
2. 点击“回滚到此版本”。
3. 在审批卡中确认影响。
4. 点击“确认回滚”。
5. 系统加载目标版本的 Chroma Collection，并重新切换活动版本。

回滚不会重建索引。若目标 Collection 文件丢失或损坏，任务失败，当前活动版本保持不变。

### 3.7 外部文件对账

对账用于发现绕过 Web API 直接放入 `data/` 或修改的文件。

触发方式：

- 后台线程每 `RAG_RECONCILE_INTERVAL_SECONDS` 秒自动执行，默认 300 秒。
- 知识库页或发布记录页的手动“文件对账”按钮。
- `POST /api/reconcile`。

对账逻辑：

1. 扫描 `data/` 下支持的文件。
2. 按显示文件名分组。
3. 同名多文件时选择修改时间最新的文件。
4. 计算 SHA-256，与活动版本文档 Hash 比较。
5. 将新增或变化文件提交为发布 Job。

当前对账不会识别“活动版本中有、磁盘上已经删除”的文档。删除操作应通过 Web API 执行。

### 3.8 任务查看与重试

任务状态：

| 状态 | 含义 |
|---|---|
| `QUEUED` | 已持久化并等待 Worker |
| `RUNNING` | Worker 正在处理 |
| `ACTIVE` | 发布或回滚成功 |
| `FAILED` | 已达到最大尝试次数或发生不可恢复错误 |
| `NOOP` | 内容或配置没有变化 |
| `REJECTED` | 预留的质量门禁拒绝状态 |

上传、删除和回滚默认最多尝试 3 次。失败重试使用指数退避，单次等待最多 5 秒。

手动重试：

```text
POST /api/jobs/<job_id>/retry
```

只有 `FAILED` 或 `REJECTED` 任务允许手动重试。

### 3.9 质量评估

评估通过 Python 调用，不提供 Web API 或管理页面。

#### 3.9.1 测试数据格式

```python
test_data = [
    {
        "query": "默认端口是多少？",
        "expected_chunk_ids": ["README.md#2"],
        "ground_truth": "默认端口是 5000。",
    }
]
```

#### 3.9.2 运行评估

```python
from main import MiniRAGSystem

system = MiniRAGSystem()
report = system.run_evaluation(test_data)
print(report["retrieval"])
print(report["ragas"])
```

检索层指标：

| 指标 | 含义 |
|---|---|
| Hit Rate@K | Top-K 是否至少命中一个期望 Chunk |
| Recall@K | 命中的期望 Chunk 占全部期望 Chunk 的比例 |
| Precision@K | Top-K 中命中 Chunk 的比例 |
| MRR | 第一个正确 Chunk 的倒数排名 |

生成层使用 RAGAS：

- `context_precision`
- `context_recall`
- `faithfulness`
- `answer_relevancy`

评估会实际调用 LLM、Reranker 和 Embedding，耗时与资源消耗明显高于普通单元测试。

### 3.10 CLI 方式建立知识库

除 Web 发布链外，`MiniRAGSystem` 仍提供命令行索引接口：

```python
from main import MiniRAGSystem

system = MiniRAGSystem()
system.ingest_knowledge(["data/docs.pdf", "data/notes.md"])
system.add_documents(["data/new.pdf"])

result = system.ask("项目的核心模块有哪些？")
print(result["answer"])
print(result["sources"])
```

这种方式直接写入默认 `langchain` Collection，不创建上传 Job。应用首次启动且 Registry 无活动版本时，会把现有索引引导为初始活动 Release。

### 3.11 HTTP API

系统提供 REST JSON API，主要接口如下：

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/api/status` | 活动版本、文档、模型、任务和对账状态 |
| POST | `/api/ask` | 知识库问答 |
| POST | `/api/ingest` | 上传文档并创建发布任务 |
| GET | `/api/documents` | 活动版本文档列表 |
| POST | `/api/delete` | 创建文档删除任务 |
| GET | `/api/jobs/<job_id>` | 查询任务 |
| POST | `/api/jobs/<job_id>/retry` | 重试失败任务 |
| GET | `/api/releases` | 最近 12 个发布版本 |
| POST | `/api/rollback` | 回滚到历史版本 |
| POST | `/api/reconcile` | 立即执行文件对账 |

问答请求示例：

```http
POST /api/ask
Content-Type: application/json

{
  "query": "系统默认端口是多少？",
  "use_query_expansion": true
}
```

当前 API 没有认证、版本前缀或 OpenAPI 文档。

## 4. 核心业务流程

### 4.1 上传与版本发布

```text
Browser
  -> POST /api/ingest
  -> validate extension and request size
  -> SHA-256 content-addressed save under data/
  -> KnowledgeReleaseManager.enqueue_upload
  -> SQLite release + job
  -> background worker
       -> load changed documents
       -> parse, clean and split
       -> embed changed chunks only
       -> reuse active vectors for unchanged documents
       -> write rag_release_<id>
       -> build in-memory BM25
       -> update manifest and STAGED state
       -> activate release and swap retriever
  -> frontend polls job until terminal state
```

### 4.2 两种发布模式

| 模式 | 触发条件 | 处理方式 |
|---|---|---|
| `CONTENT_UPDATE` | 解析器、切块器和 Embedding 版本均未变化 | 复用未受影响文档向量，只处理上传或变化文档 |
| `INDEX_UPGRADE` | 任一元数据版本与活动文档不一致 | 用当前解析、切块和 Embedding 配置重建全部文档 |

发布版本、切块方式、Embedding 模型发生变化时，应同步修改相应的 `RAG_*_VERSION` 环境变量，确保旧向量不会与新配置混用。

### 4.3 查询与生成链路

```text
Question
  -> optional LLM query rewrite
  -> BM25 retrieval
  -> Chroma dense retrieval
  -> EnsembleRetriever weighted fusion
  -> deduplicate by chunk ID
  -> CrossEncoder rerank against original question
  -> construct bounded context
  -> Ollama answer generation
  -> answer + source list
```

详细过程：

1. Flask 检查知识库是否已加载。
2. 可选步骤：LLM 生成多个检索子问题。
3. 对原问题和子问题分别执行混合召回。
4. BM25 返回候选，Dense 检索返回候选。
5. Ensemble Retriever 按 0.5/0.5 权重融合。
6. 按 Chunk ID 去重。
7. Cross-Encoder 用原始问题统一重排。
8. 将来源、页码和正文组装为上下文。
9. Ollama 生成回答。
10. 返回回答、来源 Chunk 数和上下文数量。

### 4.4 删除流程

```text
Browser approval
  -> POST /api/delete
  -> create DELETE job
  -> snapshot active vectors without target document
  -> write new collection
  -> switch active release
  -> retain old release under cleanup policy
```

删除最后一个文档时，新 Release 的 Chunk 数为 0，活动 Retriever 被清空，系统状态变为未加载。

### 4.5 回滚流程

```text
Browser approval
  -> POST /api/rollback
  -> validate target state
  -> load target Chroma collection
  -> rebuild BM25 from collection documents
  -> mark current release SUPERSEDED
  -> mark target release ACTIVE
  -> replace in-process retriever
```

回滚是版本切换，不是索引重建。目标版本必须为 `ACTIVE`、`ACTIVE_WITH_WARNING` 或 `SUPERSEDED`。

### 4.6 任务幂等与恢复

- 上传请求键包含活动版本、文档 ID、内容 Hash、解析器版本、切块版本和 Embedding 版本。
- 相同请求已经处于 `QUEUED`、`RUNNING`、`ACTIVE` 或 `NOOP` 时复用已有 Job。
- 失败或拒绝的 Job 再次提交时按新请求处理。
- 进程重启时，`RUNNING` Job 回到 `QUEUED`，`STAGING` Release 回到 `PENDING`。
- 已排队 Job 会重新加入 Worker 队列。
- 活动版本在进程启动时从 Registry 恢复。

### 4.7 对账流程

```text
scan data/
  -> group by display filename
  -> choose latest modified file per name
  -> hash each candidate
  -> compare with active release manifest
  -> enqueue changed/new documents
  -> record reconciliation state
```

对账是补充入口，不应替代正常上传流程。通过 API 删除文档是当前唯一完整支持外部可见删除的业务路径。

### 4.8 评估流程

```text
test_data
  -> retrieval with rerank
  -> calculate Hit / Recall / Precision / MRR
  -> ask questions with query expansion disabled
  -> collect answer and contexts
  -> prepare RAGAS Dataset
  -> evaluate generation metrics
  -> return retrieval DataFrame and RAGAS result
```

检索指标评估和生成评估使用相同原始查询，但评估流程不会启用查询改写，以保持检索配置可重复。

## 5. 模块划分

### 5.1 分层架构

```text
Presentation
  Jinja HTML, CSS variables, browser state, Markdown renderer

HTTP API
  Flask routes, upload validation, security headers, JSON responses

Application
  MiniRAGSystem, KnowledgeReleaseManager, release jobs

Retrieval and Generation
  HybridRetriever, BM25, Chroma, CrossEncoder, Ollama LLM

Parsing and Evaluation
  MultiModalDataLoader, PDF/OCR/DOCX parsing, RAGAS

Persistence
  SQLite registry, Chroma collections, content-addressed source files
```

### 5.2 模块职责

| 模块 | 职责 | 不负责 |
|---|---|---|
| `templates/index.html` | 页面布局、主题、聊天、文档/版本管理、审批确认、API 调用 | 发布事务、检索和模型推理 |
| `app.py` | Flask 路由、参数校验、文件接收、启动配置 | 索引构建细节 |
| `main.py` | 系统组合、问答、命令行索引、评估编排 | HTTP 和后台任务队列 |
| `data_loader.py` | PDF/OCR/Markdown/TXT/DOCX 解析、清洗和切块 | Embedding 和版本发布 |
| `retriever.py` | Chroma、BM25、Ensemble、Reranker、查询改写 | 文档解析和发布状态 |
| `knowledge_release.py` | 公共发布请求、Job 创建、去重、重试 | 具体发布步骤 |
| `release_registry.py` | SQLite Schema、状态查询、启动恢复、版本清单 | 索引执行 |
| `release_worker.py` | 后台 Worker、新版本构建、激活、回滚、对账、清理 | HTTP 参数处理 |
| `release_models.py` | 状态常量、上传数据模型、时间和 Hash 工具 | 业务执行 |
| `evaluator.py` | 检索指标、RAGAS 数据准备和执行 | Web 展示 |
| `rag_utils.py` | 中文 BM25 分词、路径安全、Hash、通用指标 | 业务编排 |
| `tests/` | 无真实模型依赖的单元测试 | 完整模型与 OCR 集成验证 |

### 5.3 依赖方向

- 前端只依赖 `/api/*` 契约。
- API 依赖 `MiniRAGSystem` 和 `KnowledgeReleaseManager`。
- `MiniRAGSystem` 依赖解析器、Retriever、LLM 和评估器。
- `HybridRetriever` 依赖 Chroma、BM25、Embedding 和 CrossEncoder。
- Worker 依赖 Registry、解析器、Embedder 和 Chroma。
- Registry 不依赖前端，也不负责模型调用。
- 数据文件、SQLite 和 Chroma 由发布层协调，前端不能直接修改。

## 6. 代码结构

```text
app.py                         Flask 应用和 REST API
main.py                        MiniRAGSystem 主入口
data_loader.py                 多格式文档解析、OCR、切块
retriever.py                   BM25 + Dense + Rerank + 查询改写
rag_utils.py                   中文分词、路径安全和指标
evaluator.py                   检索指标与 RAGAS
release_models.py              Release/Job 状态与 UploadItem
release_registry.py            SQLite Schema、事务、状态和恢复
release_worker.py              Worker、发布、回滚、对账、清理
knowledge_release.py           发布管理器公开请求 API
templates/
  index.html                   单页工作台、主题、Markdown、交互
tests/
  test_rag_utils.py            分词、路径、指标
  test_knowledge_release.py    发布、删除、回滚、恢复、对账
  test_index_lifecycle.py      Chroma 生命周期与 Reranker 回退
  check_frontend.mjs           前端内联脚本语法检查
requirements.txt               Python 依赖
README.md                      项目概览和基础用法
```

运行产物通常位于：

```text
data/                          上传和源文档文件
chroma_db/                     Chroma 持久化目录
knowledge_registry.sqlite3     SQLite 发布注册表
knowledge_registry.sqlite3-wal
knowledge_registry.sqlite3-shm
```

这些运行数据默认被 `.gitignore` 排除。

### 6.1 关键入口

| 场景 | 入口 |
|---|---|
| Web 启动 | `python app.py` |
| Flask API | `app.py` |
| 问答 | `MiniRAGSystem.ask` |
| 文档解析 | `MultiModalDataLoader.load_file` |
| 文本切块 | `MultiModalDataLoader.process_documents` |
| 混合检索 | `HybridRetriever.retrieve_with_rerank` |
| 查询改写 | `HybridRetriever.query_expansion_retrieval` |
| 上传 Job | `KnowledgeReleaseManager.enqueue_upload` |
| 删除 Job | `KnowledgeReleaseManager.enqueue_delete` |
| 回滚 Job | `KnowledgeReleaseManager.enqueue_rollback` |
| Worker 主循环 | `ReleaseWorkerBase._worker_loop` |
| 发布激活 | `ReleaseWorkerBase._activate_release` |
| SQLite 初始化 | `ReleaseRegistryBase._init_registry` |
| 评估 | `MiniRAGSystem.run_evaluation` |
| 前端视图切换 | `templates/index.html` 的 `setView` |

### 6.2 新增文档格式

新增格式时应完成：

1. 在 Flask `ALLOWED_EXT` 中增加扩展名。
2. 在 `MultiModalDataLoader.load_file` 增加分派。
3. 返回带 `source`、`file_type` 和稳定 Hash 元数据的 `Document`。
4. 确认切块后的来源、页码和内容 Hash 可追溯。
5. 更新前端上传提示和 `accept` 属性。
6. 增加解析、空文件、损坏文件和安全路径测试。
7. 修改解析器版本并触发版本化重建。

### 6.3 修改切块或解析逻辑

解析器或切块器行为变化会改变 Chunk ID、内容和向量语义。变更时应：

1. 更新 `RAG_PARSER_VERSION` 或 `RAG_CHUNKING_VERSION`。
2. 确认 Release Worker 进入 `INDEX_UPGRADE`。
3. 重建完整活动知识库。
4. 运行检索指标与人工问答抽查。
5. 保留旧 Release，直到新版本验证完成。

### 6.4 修改 Embedding 或 Reranker

- 修改 Embedding 模型时，必须修改 `RAG_EMBEDDING_VERSION`。
- 不同 Embedding 模型的向量不能在同一 Release 中混用。
- Reranker 不参与持久化向量，切换 Reranker 不会要求重建索引。
- 新的 Reranker 首次使用前应提前缓存，避免首次问答出现长时间等待。

### 6.5 修改 SQLite 数据模型

当前没有 Alembic 等迁移框架。修改前应：

1. 在 `_init_registry` 中使用兼容的 `CREATE TABLE IF NOT EXISTS` 和 `_ensure_column`。
2. 为已有 Registry 设计增列和旧数据兼容逻辑。
3. 更新字典序列化、启动恢复和前端字段。
4. 增加旧数据库启动测试。
5. 在升级前备份 SQLite 主文件、WAL 和 SHM。

### 6.6 修改前端

当前前端不是 React，也没有 TypeScript 构建步骤。修改时应：

1. 保持单套 CSS 设计变量和 light/dark 主题。
2. 不改变 API 路径和响应字段。
3. 对用户文档和模型输出使用安全文本或受控 Markdown 渲染。
4. 在真实浏览器中检查桌面、900px 以下宽度、主题和空状态。
5. 运行 JavaScript 语法检查或 `tests/check_frontend.mjs`，前提是环境有 Node.js。
6. 验证增删改、任务轮询、回滚和错误呈现。

## 7. 数据与状态

### 7.1 SQLite 表

| 表 | 用途 |
|---|---|
| `system_state` | `active_release_id` 和最近对账状态等键值 |
| `releases` | 发布版本、状态、Collection、Chunk 数、Manifest、时间、错误和发布模式 |
| `release_documents` | 每个 Release 的逻辑文档清单和版本元数据 |
| `jobs` | 上传、删除、回滚任务、状态、载荷、尝试次数和时间 |

`releases` 还包含以下预留或辅助列：

- `quality_state`
- `quality_report_json`
- `quality_gate_version`
- `force_activated`
- `force_activated_at`
- `force_reason`
- `release_mode`

SQLite 启用 WAL 和 Foreign Keys。业务代码通过 RLock 和事务协调单进程内读写。

### 7.2 发布文档清单

`release_documents` 保存：

| 字段 | 含义 |
|---|---|
| `release_id` | 所属发布版本 |
| `doc_id` | 逻辑文档 ID |
| `display_name` | 用户可见文件名 |
| `source_path` | 内容寻址源文件绝对路径 |
| `content_hash` | 源文件 SHA-256 |
| `parser_version` | 解析器版本 |
| `chunking_version` | 切块策略版本 |
| `embedding_version` | Embedding 配置版本 |
| `chunk_count` | 该文档 Chunk 数 |
| `size_bytes` | 文件大小 |

### 7.3 Chunk 元数据

解析和切块阶段提供：

- `source`
- `page`
- `file_type`
- `page_hash`
- `ocr_used`
- `document_name`
- `chunk_id`
- `chunk_index`
- `content_hash`

发布 Worker 额外写入：

- `doc_id`
- `document_hash`
- `parser_version`
- `chunking_version`
- `embedding_version`
- `release_id`

稳定 Chunk ID 格式为：

```text
<document_name>#<chunk_index>
```

### 7.4 Chroma 数据

| Collection | 用途 |
|---|---|
| `langchain` | 命令行直接建立或首次引导的默认 Collection |
| `rag_release_<id>` | 版本化发布 Collection |

每个 Collection 包含：

- Chunk ID。
- Chunk 正文。
- 元数据。
- 持久化 Embedding。

BM25 索引不持久化。每次加载 Release 时，从 Collection 的文档和元数据在内存中重建。

### 7.5 Release 状态

| 状态 | 含义 |
|---|---|
| `PENDING` | 已创建发布记录并等待 Worker |
| `STAGING` | Worker 正在构建新版本 |
| `STAGED` | Collection 和 Manifest 已构建，等待激活 |
| `ACTIVE` | 当前活动版本 |
| `SUPERSEDED` | 被新活动版本替代 |
| `FAILED` | 构建失败 |
| `NOOP` | 请求未产生有效变化 |
| `REJECTED` | 预留质量拒绝状态 |
| `ACTIVE_WITH_WARNING` | 预留带警告激活状态 |

### 7.6 Job 状态

非终态：

```text
QUEUED / RUNNING
```

终态：

```text
ACTIVE / FAILED / NOOP / REJECTED
```

### 7.7 数据保留与清理

- 默认保留最近 5 个 Release。
- 活动 Release 始终保留。
- 可清理状态为 `SUPERSEDED`、`FAILED`、`REJECTED` 和 `NOOP`。
- 清理旧 Release 时同步删除对应 Job、文档清单、Chroma Collection 和不再被保留版本引用的源文件。
- 保留数量通过 `RAG_RETAIN_RELEASES` 配置，代码下限为 2。

### 7.8 备份

完整备份至少包括：

```text
knowledge_registry.sqlite3
knowledge_registry.sqlite3-wal
knowledge_registry.sqlite3-shm
chroma_db/
data/
```

如果只备份源文档而不备份 Registry 和 Chroma，系统仍可重建，但会丢失 Release 历史、活动版本关系和任务状态。

## 8. 部署与配置

### 8.1 运行要求

- Python 3.10+；当前测试环境为 Python 3.13.5。
- Ollama。
- `qwen2.5` 模型。
- `nomic-embed-text` Embedding 模型。
- Tesseract OCR；扫描 PDF 和图片 OCR 需要 `chi_sim` 与 `eng` 语言包。
- 可读写的 `data/`、Chroma 和 Registry 目录。
- 足够容纳 Reranker 与本地 LLM 的内存和显存。

项目没有 Dockerfile、Compose、前端构建产物或独立异步任务服务。

### 8.2 安装

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Linux/macOS：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

安装 OCR：

```text
安装 Tesseract
下载 chi_sim.traineddata 和 eng.traineddata
配置 TESSDATA_PREFIX
```

### 8.3 初始化模型

```powershell
ollama pull qwen2.5
ollama pull nomic-embed-text
ollama serve
```

先单独验证模型：

```powershell
ollama run qwen2.5 "回答 OK"
ollama ps
```

### 8.4 启动应用

```powershell
python app.py
```

默认地址：

```text
http://127.0.0.1:5000
```

应用启动时会：

1. 加载或初始化默认 Chroma Collection。
2. 初始化 SQLite Registry。
3. 恢复中断任务。
4. 引导或恢复活动 Release。
5. 启动单线程发布 Worker。
6. 在配置大于 0 时启动对账线程。

### 8.5 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `RAG_EMBED_MODEL` | `nomic-embed-text` | Ollama Embedding 模型 |
| `RAG_LLM_MODEL` | `qwen2.5` | Ollama LLM 模型 |
| `RAG_PERSIST_DIR` | `chroma_db` | Chroma 持久化目录 |
| `RAG_REGISTRY_PATH` | `knowledge_registry.sqlite3` | SQLite Registry |
| `RAG_PARSER_VERSION` | `parser-v1` | 触发索引升级的解析版本 |
| `RAG_CHUNKING_VERSION` | `chunk-500-50-v1` | 触发索引升级的切块版本 |
| `RAG_EMBEDDING_VERSION` | Embedding 模型名 | 触发索引升级的 Embedding 版本 |
| `RAG_RETAIN_RELEASES` | `5` | 保留最近发布数量 |
| `RAG_RECONCILE_INTERVAL_SECONDS` | `300` | 自动对账间隔；0 表示禁用 |
| `RAG_HOST` | `127.0.0.1` | Flask 监听地址 |
| `RAG_PORT` | `5000` | Flask 监听端口 |
| `RAG_RERANKER_ALLOW_DOWNLOAD` | 未设置 | 设置为 `1` 允许下载主 Reranker |
| `HF_ENDPOINT` | `https://hf-mirror.com` | Hugging Face 镜像默认值 |

### 8.6 测试

```powershell
python -m unittest discover -s tests -v
```

当前共有 18 个单元测试，覆盖：

- 中文和混合英文 BM25 分词。
- 路径穿越和上传文件名安全。
- 检索指标。
- Chroma 构建、替换、删除和重新加载。
- Reranker 本地回退。
- 发布、幂等、删除、回滚和最后一次文档删除。
- 失败任务重试和重启恢复。
- 外部文件变化对账。
- 旧 Release 和源文件清理。

测试使用 Fake Embedding，不验证真实 Ollama 输出质量、真实 CUDA 启动、OCR 准确率或 RAGAS 稳定性。

## 9. 安全与权限边界

### 9.1 已实现的风险控制

1. 上传扩展名白名单。
2. 上传文件名去除路径、控制字符和 Windows 非法字符。
3. 删除路径解析后必须仍位于 `data/` 内。
4. 上传文件按内容 Hash 保存，降低同名覆盖混淆。
5. 文档内容在提示中被标记为不可信数据。
6. 前端 Markdown 渲染先转义 HTML，再处理有限 Markdown 语法。
7. 高风险删除和回滚在前端显示影响、步骤并需确认。
8. 响应包含 `X-Content-Type-Options`、`X-Frame-Options` 和 `Referrer-Policy`。
9. 默认只监听 `127.0.0.1`。

### 9.2 不能替代的安全措施

- 系统没有身份认证和授权。
- 没有 CSRF Token；默认同机低风险，但公网部署不安全。
- 没有查询长度、调用频率和并发限制。
- API 会把底层异常字符串返回前端，可能包含本地路径、模型或运行库信息。
- 提示词对文档注入只有指令层防护，没有内容隔离或输出策略引擎。
- 前端“审批”是用户确认，不是服务端 Pending Action 状态机。
- Chroma、SQLite 和源文件没有加密。
- Ollama 服务自身安全边界决定系统能否安全调用模型。

### 9.3 部署建议

- 仅部署在可信本机或受控内网。
- 在反向代理层增加 TLS、登录、访问控制、限流和请求审计。
- 使用专用低权限账号运行 Flask 和 Ollama。
- 不要将 Ollama、Chroma 或 SQLite 端口直接暴露到公网。
- 对上传内容执行恶意文件扫描和文档安全检查。
- 定期备份 Registry、Chroma 和源文件。
- 将模型服务与业务 API 放在同一受控网络。
- 对公开知识库执行禁止敏感信息入库检查。

## 10. 故障处理

| 现象 | 检查项 | 处理方式 |
|---|---|---|
| 页面打不开 | Flask 进程、端口、反向代理 | 检查 `RAG_HOST`、`RAG_PORT` 和进程日志 |
| 页面显示知识库未加载 | Registry、Chroma、活动 Release | 上传文档或调用 `/api/status` 检查状态 |
| 上传返回“不支持文件” | 扩展名和文件选择 | 只使用 PDF、MD、TXT、DOCX |
| 上传返回 413 | 单次请求总大小 | 拆分上传或调整反向代理/Flask 上限 |
| 任务长期 `QUEUED` | Worker 进程、Registry 和线程状态 | 重启应用并检查任务恢复 |
| 任务失败 | Job `error`、源文件、解析器和模型 | 修复原因后调用重试接口 |
| 文档没有 Chunk | 文件内容、OCR、PDF 文本层 | 检查 OCR 环境、语言包和文件是否损坏 |
| 回答为 Ollama 错误文本 | Ollama 服务、模型、显存和 Runner | 直接执行最小 Ollama 请求定位 |
| `CUDA shared object initialization failed` | Ollama 版本、NVIDIA 驱动、显存 | 见 10.1 专项排查 |
| 首次回答很慢 | Reranker 懒加载、Embedding、LLM 加载 | 预热模型并确认本地缓存 |
| Reranker 加载失败 | 本地缓存、模型名、HF 镜像 | 缓存备用模型或设置允许下载 |
| 回滚失败 | 目标状态、Collection 文件 | 检查 Release 和 Chroma 数据 |
| 删除文档 404 | 活动版本文档和 `data/` 文件 | 从 `/api/documents` 获取当前 `file` |
| 对账无变化 | 文件名、Hash、活动清单 | 确认文件通过支持扩展名保存并实际改变内容 |
| 外部删除没有生效 | 当前对账实现 | 通过 Web API 删除，不使用文件系统删除 |
| RAGAS 失败 | 模型兼容性、网络、数据和指标参数 | 先用小样本验证 Embedding、LLM 和 ground truth |
| Windows 临时目录清理失败 | Chroma/HNSW 文件句柄 | 关闭进程后清理，测试使用可忽略清理错误 |
| SQLite locked | 多进程同时运行 | 单进程运行；重启并避免多 Worker |

### 10.1 Ollama 与 CUDA 故障

如果界面出现类似：

```text
llama-server process has terminated: exit status 0xc0000409
CUDA error: shared object initialization failed
```

这不是知识库生成的回答，而是 `/api/ask` 捕获了 Ollama Runner 启动异常后返回的错误。`0xc0000409` 是 Windows 原生进程快速失败码，常见于原生运行库异常，不表示用户输入已被恶意利用。

建议顺序：

1. 检查 Ollama 是否运行：

```powershell
ollama --version
ollama ps
ollama list
```

2. 检查 GPU 和显存：

```powershell
nvidia-smi
```

3. 执行最小生成请求：

```powershell
ollama run qwen2.5 "只回答 OK"
```

4. 同时驻留 LLM 与 Embedding 时检查显存：

```powershell
ollama ps
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

5. 若显存接近上限，关闭占用 GPU 的程序，或改用更小的模型，例如 `qwen2.5:3b`。

6. 若错误反复出现，重启 Ollama，更新或重装 NVIDIA 驱动，并确认 Ollama 版本与 CUDA 运行库兼容。

7. 为确认是否为 GPU 路径问题，可临时使用 CPU 推理或更小上下文进行对照。

执行期间的生成模型与 Embedding 模型可能同时驻留。6GB 显存环境中，7B 量化模型与 Embedding 同时加载后剩余空间可能很小，这会增加 Runner 启动失败的稳定性风险。

### 10.2 排障优先顺序

1. `GET /api/status`
2. 当前活动 Release 和文档清单
3. Job 状态与错误
4. Ollama 最小请求
5. `data/` 与 Chroma Collection
6. SQLite Registry 和 WAL
7. Flask 与 Ollama 进程日志
8. Reranker 和 OCR 环境

## 11. 已知限制

- 所谓“多模态”当前主要是图片和扫描页 OCR，没有视觉 Embedding、图片检索或 VLM 生成。
- DOCX 内图片不会 OCR，只提取文字段落、表格、页眉和页脚。
- PDF 行号为纯数字的行可能被清洗规则移除。
- BM25 无法增量更新，每次发布或加载 Release 都需要重建内存索引。
- Reranker 是懒加载，首次问答可能明显变慢。
- 发布 Worker 和任务协调包含进程内状态，只适合单进程部署。
- SQLite 是单文件 Registry，多进程 Worker 同时写入不安全。
- 前端聊天历史不持久化，刷新页面后丢失。
- 停止按钮只中断前端请求等待，不能保证终止服务端或 Ollama 推理。
- 当前没有模型原生 Token Streaming。
- 系统没有登录、用户、租户和文档级权限。
- 前端“审批”仅覆盖删除和回滚确认，没有服务端 Pending Action 持久化和审批超时。
- Release Schema 和前端存在质量状态预留，但 Worker 尚未执行自动质量门禁。
- 质量字段没有完整暴露到 Release 列表 API。
- 外部对账不检测文件删除。
- 未变化文档向量的增量发布依赖当前向量快照，活动 Retriever 异常时无法完成增量构建。
- 检索和生成错误直接返回底层异常，可能暴露本地实现细节。
- RAGAS 没有 Web 页面、API 或 CI 基线门禁。
- 单元测试不覆盖真实模型、CUDA、OCR 和端到端答案质量。
- README 中 Reranker 名称可能与当前 `retriever.py` 默认值存在版本文档漂移，以代码实际值为准。

## 12. 维护与变更流程

一次涉及运行行为的变更建议按以下顺序完成：

1. 在独立分支中修改代码。
2. 明确是否影响解析器、切块、Embedding、Reranker、Prompt 或 API 契约。
3. 若影响向量语义，更新相应版本变量并规划全量重建。
4. 增加或更新单元测试和安全用例。
5. 启动真实 Ollama，执行最小生成和完整问答。
6. 运行检索指标与人工答案抽查。
7. 在浏览器中验证桌面、窄屏、浅色/深色、错误和空状态。
8. 检查旧 Registry、旧 Release 和回滚兼容性。
9. 更新 README、本说明书和环境配置说明。
10. 审查差异后提交。

文档维护原则：

- 默认配置以代码为准，不以旧 README 或示例为准。
- 发布状态以 Registry 和 Worker 实际写入状态为准。
- API 以 Flask 路由和前端实际字段为准。
- 不得把预留字段描述为已实现功能。
- 安全章节必须同时列出已实现防护和未覆盖风险。
- 模型或驱动故障排查应提供可复现命令，而不是只展示错误截图。

## 13. 附录

### 13.1 主要 API 速查

```text
GET  /
GET  /api/status
POST /api/ask
POST /api/ingest
GET  /api/documents
POST /api/delete
GET  /api/jobs/<job_id>
POST /api/jobs/<job_id>/retry
GET  /api/releases
POST /api/rollback
POST /api/reconcile
```

### 13.2 状态速查

Job：

```text
QUEUED / RUNNING / ACTIVE / FAILED / NOOP / REJECTED
```

Release：

```text
PENDING / STAGING / STAGED / ACTIVE / SUPERSEDED / FAILED / NOOP
```

发布模式：

```text
CONTENT_UPDATE / INDEX_UPGRADE
```

### 13.3 默认配置速查

```text
RAG_EMBED_MODEL=nomic-embed-text
RAG_LLM_MODEL=qwen2.5
RAG_PERSIST_DIR=chroma_db
RAG_REGISTRY_PATH=knowledge_registry.sqlite3
RAG_PARSER_VERSION=parser-v1
RAG_CHUNKING_VERSION=chunk-500-50-v1
RAG_EMBEDDING_VERSION=nomic-embed-text
RAG_RETAIN_RELEASES=5
RAG_RECONCILE_INTERVAL_SECONDS=300
RAG_HOST=127.0.0.1
RAG_PORT=5000
RAG_RERANKER_ALLOW_DOWNLOAD=0
```

### 13.4 检索参数速查

| 参数 | 默认值 |
|---|---|
| Chunk Size | 500 |
| Chunk Overlap | 50 |
| BM25 Weight | 0.5 |
| Dense Weight | 0.5 |
| Candidate K | 10 |
| Final Top K | 3 |
| Query Expansion Subqueries | 最多 3 |
| Expanded Retrieval Context | 最多 5 |
| 主 Reranker | `BAAI/bge-reranker-v2-m3` |
| 回退 Reranker | `BAAI/bge-reranker-base` |

### 13.5 相关文档

| 文档 | 用途 |
|---|---|
| `README.md` | 项目概览、快速开始、模块和基础设计说明 |
| `docs/software-manual.md` | 当前正式软件说明书 |
| `app.py` | HTTP API 和环境变量的事实来源 |
| `release_registry.py` | SQLite 表结构的事实来源 |
| `release_worker.py` | 发布、回滚、对账和清理的事实来源 |
| `retriever.py` | 检索、精排默认参数的事实来源 |
| `requirements.txt` | Python 依赖范围 |

### 13.6 修订记录

| 日期 | 版本 | 说明 |
|---|---|---|
| 2026-11-17 | 1.0 | 建立正式软件说明书，覆盖功能、操作、发布链路、问答链路、架构、数据结构、部署、安全和排障 |
