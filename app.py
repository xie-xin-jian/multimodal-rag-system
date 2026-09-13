"""多模态RAG知识库问答系统 Web UI 后端。

用 Flask 暴露 REST API，前端单页 HTML 负责聊天交互。
启动：python app.py  →  http://127.0.0.1:5000
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from flask import Flask, request, jsonify, render_template

from main import MiniRAGSystem
from rag_utils import (
    display_filename,
    resolve_within,
    safe_upload_filename,
    sha256_file,
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB

# 全局 RAG 系统实例（启动时加载一次）
rag = MiniRAGSystem(
    embed_model=os.environ.get("RAG_EMBED_MODEL", "nomic-embed-text"),
    llm_model=os.environ.get("RAG_LLM_MODEL", "qwen2.5"),
    persist_directory=os.environ.get("RAG_PERSIST_DIR", "chroma_db"),
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

ALLOWED_EXT = {".pdf", ".md", ".txt", ".markdown", ".docx"}


def _allowed_file(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXT


def _resolve_data_file(filename: str) -> Path:
    return resolve_within(DATA_DIR, filename)


@app.after_request
def add_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def status():
    chunk_count = rag.retriever.chunk_count if rag._loaded else 0
    sources_set = set(rag.retriever.document_names) if rag._loaded else set()

    return jsonify(
        {
            "loaded": rag._loaded,
            "chunk_count": chunk_count,
            "documents": sorted(sources_set),
            "embed_model": rag.embed_model,
            "llm_model": rag.llm_model,
        }
    )


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
    document_names = []
    created_paths = []
    replaced_paths = set()
    batch_names = set()

    for f in files:
        if not f or not f.filename:
            continue
        if not _allowed_file(f.filename):
            continue

        safe_name = safe_upload_filename(f.filename)
        stem, ext = os.path.splitext(safe_name)
        safe_stem = stem or "document"
        safe_name = f"{safe_stem}{ext.lower()}"
        if safe_name in batch_names:
            for created_path in created_paths:
                if os.path.exists(created_path):
                    os.remove(created_path)
            return jsonify({"error": f"duplicate filename: {safe_name}"}), 400
        batch_names.add(safe_name)

        digest = sha256_file(f.stream)
        unique_name = f"{digest[:16]}_{safe_name}"
        save_path = os.path.join(DATA_DIR, unique_name)
        if not os.path.exists(save_path):
            f.save(save_path)
            created_paths.append(save_path)
        saved_paths.append(save_path)
        document_name = safe_name
        document_names.append(document_name)

        for existing in os.listdir(DATA_DIR):
            existing_path = os.path.join(DATA_DIR, existing)
            if (
                os.path.isfile(existing_path)
                and existing_path != save_path
                and display_filename(existing) == document_name
            ):
                replaced_paths.add(existing_path)

    if not saved_paths:
        return jsonify({"error": "no valid files (supported: PDF, MD, TXT, DOCX)"}), 400

    try:
        if rag._loaded:
            rag.add_documents(saved_paths, document_names=document_names)
        else:
            rag.ingest_knowledge(saved_paths, document_names=document_names)

        for old_path in replaced_paths:
            if os.path.exists(old_path):
                os.remove(old_path)

        return jsonify({"success": True, "files": document_names})
    except Exception as e:
        for created_path in created_paths:
            if os.path.exists(created_path):
                os.remove(created_path)
        return jsonify({"error": str(e)}), 500


@app.route("/api/documents")
def documents():
    """列出 data 目录下所有文档"""
    docs = []
    for f in os.listdir(DATA_DIR):
        if os.path.splitext(f)[1].lower() in ALLOWED_EXT:
            docs.append(
                {
                    "name": display_filename(f),
                    "file": f,
                    "size": os.path.getsize(os.path.join(DATA_DIR, f)),
                }
            )
    docs.sort(key=lambda x: x["name"])
    return jsonify(docs)


@app.route("/api/delete", methods=["POST"])
def delete_document():
    """从知识库中删除指定文档"""
    data = request.get_json(force=True)
    filename = data.get("file", "")
    if not filename:
        return jsonify({"error": "missing 'file' parameter"}), 400

    try:
        candidate = _resolve_data_file(filename)
    except ValueError:
        return jsonify({"error": "invalid filename"}), 400

    actual_file = candidate.name if candidate.exists() else ""
    if not actual_file:
        for current in os.listdir(DATA_DIR):
            if display_filename(current) == filename:
                actual_file = current
                break

    if not actual_file:
        return jsonify({"error": "document not found"}), 404

    try:
        removed = rag.remove_document(actual_file)
        return jsonify({"success": True, "removed": removed})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    host = os.environ.get("RAG_HOST", "127.0.0.1")
    port = int(os.environ.get("RAG_PORT", "5000"))
    print(f"\n{'=' * 50}")
    print("  多模态RAG知识库问答系统 Web UI")
    print(f"  知识库状态: {'已加载' if rag._loaded else '空'}")
    print(f"  Chunk 数量: {rag.retriever.chunk_count}")
    print(f"  访问地址: http://{host}:{port}")
    print(f"{'=' * 50}\n")
    app.run(host=host, port=port, debug=False)
