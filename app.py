"""多模态RAG知识库问答系统 Web UI 后端。

用 Flask 暴露 REST API，前端单页 HTML 负责聊天交互。
启动：python app.py  →  http://127.0.0.1:5000
"""
from __future__ import annotations

import os
import re
import uuid

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from flask import Flask, request, jsonify, render_template

from main import MiniRAGSystem

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB

# 全局 RAG 系统实例（启动时加载一次）
rag = MiniRAGSystem()

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

ALLOWED_EXT = {".pdf", ".md", ".txt", ".markdown", ".docx"}


def _allowed_file(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXT


def _safe_filename(filename: str) -> str:
    """清理文件名但保留中文。"""
    name = filename.replace("/", "_").replace("\\", "_").replace("\0", "")
    name = re.sub(r"\s+", " ", name)
    name = re.sub(r"\.{2,}", ".", name)
    return name.strip(" .")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def status():
    chunk_count = len(rag.retriever._all_chunks) if rag._loaded else 0
    # 统计不重复的源文件
    sources_set = set()
    for chunk in rag.retriever._all_chunks:
        src = chunk.metadata.get("source", "")
        if src:
            sources_set.add(os.path.basename(src))

    return jsonify({
        "loaded": rag._loaded,
        "chunk_count": chunk_count,
        "documents": sorted(sources_set),
        "embed_model": "nomic-embed-text",
        "llm_model": "qwen2.5",
    })


@app.route("/api/ask", methods=["POST"])
def ask():
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "missing 'query' field"}), 400

    query = data["query"].strip()
    if not query:
        return jsonify({"error": "query is empty"}), 400

    use_expansion = data.get("use_query_expansion", True)

    if not rag._loaded:
        return jsonify({"error": "知识库为空，请先上传文档"}), 400

    try:
        result = rag.ask(query, use_query_expansion=use_expansion)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/ingest", methods=["POST"])
def ingest():
    if "files" not in request.files:
        return jsonify({"error": "no files uploaded"}), 400

    files = request.files.getlist("files")
    saved_paths = []

    for f in files:
        if not f or not f.filename:
            continue
        if not _allowed_file(f.filename):
            continue

        # 保留中文，只做安全清理（替换路径分隔符、去掉空字符）
        stem, ext = os.path.splitext(f.filename)
        safe_stem = _safe_filename(stem) or "document"
        safe_name = f"{safe_stem}{ext.lower()}"
        unique_name = f"{uuid.uuid4().hex[:8]}_{safe_name}"
        save_path = os.path.join(DATA_DIR, unique_name)
        f.save(save_path)
        saved_paths.append(save_path)

    if not saved_paths:
        return jsonify({"error": "no valid files (supported: PDF, MD, TXT, DOCX)"}), 400

    try:
        if rag._loaded:
            rag.add_documents(saved_paths)
        else:
            rag.ingest_knowledge(saved_paths)
        display_names = []
        for p in saved_paths:
            bn = os.path.basename(p)
            display_names.append(bn[9:] if len(bn) >= 9 and bn[8] == "_" else bn)
        return jsonify({"success": True, "files": display_names})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/documents")
def documents():
    """列出 data 目录下所有文档"""
    docs = []
    for f in os.listdir(DATA_DIR):
        if os.path.splitext(f)[1].lower() in ALLOWED_EXT:
            display_name = f
            if len(f) >= 9 and f[8] == "_":
                display_name = f[9:]
            docs.append({
                "name": display_name,
                "file": f,
                "size": os.path.getsize(os.path.join(DATA_DIR, f)),
            })
    docs.sort(key=lambda x: x["name"])
    return jsonify(docs)


@app.route("/api/delete", methods=["POST"])
def delete_document():
    """从知识库中删除指定文档"""
    data = request.get_json(force=True)
    filename = data.get("file", "")
    if not filename:
        return jsonify({"error": "missing 'file' parameter"}), 400

    # 解析完整文件名（带 UUID 前缀）
    # documents 接口返回的 "file" 字段就是实际文件名
    actual_file = filename
    if not os.path.exists(os.path.join(DATA_DIR, actual_file)):
        # 可能只传了显示名，尝试匹配完整文件名
        for f in os.listdir(DATA_DIR):
            if os.path.splitext(f)[1].lower() in ALLOWED_EXT:
                display = f[9:] if len(f) >= 9 and f[8] == "_" else f
                if display == filename:
                    actual_file = f
                    break

    try:
        removed = rag.remove_document(actual_file)
        return jsonify({"success": True, "removed": removed})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print(f"\n{'=' * 50}")
    print("  多模态RAG知识库问答系统 Web UI")
    print(f"  知识库状态: {'已加载' if rag._loaded else '空'}")
    print(f"  Chunk 数量: {len(rag.retriever._all_chunks)}")
    print(f"  访问地址: http://127.0.0.1:5000")
    print(f"{'=' * 50}\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
