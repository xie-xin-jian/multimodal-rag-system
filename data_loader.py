"""多模态文档加载与切块。

支持 PDF（含扫描页 OCR 兜底）与 Markdown，输出带元数据的 LangChain Document。
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import Iterable

import pdfplumber
import pytesseract
from docx import Document as DocxDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# 扫描版 PDF 常见特征：单页可提取文本极少。低于该阈值则整页 OCR。
_MIN_TEXT_LEN = 50


class MultiModalDataLoader:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        # 按 markdown 结构 + 段落 + 句子层级切分，尽量保留语义边界
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
        )

    def load_file(self, file_path: str) -> list[Document]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(file_path)

        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            return self._load_pdf(file_path)
        if ext in {".md", ".markdown"}:
            return self._load_markdown(file_path)
        if ext == ".txt":
            return self._load_text(file_path)
        if ext == ".docx":
            return self._load_word(file_path)
        raise ValueError(f"unsupported file type: {ext}")

    def _load_pdf(self, file_path: str) -> list[Document]:
        docs: list[Document] = []
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                text = self._clean_text(text)
                ocr_used = False

                # 文本过少多半是扫描页，整页渲染后 OCR 兜底
                if len(text) < _MIN_TEXT_LEN:
                    page_ocr_text = self._ocr_page(page)
                    if page_ocr_text:
                        text = page_ocr_text
                        ocr_used = True

                # 整页 OCR 已覆盖页内图片，避免对同一扫描图重复识别。
                if not ocr_used:
                    for img in page.images:
                        crop_text = self._ocr_image_region(page, img)
                        if crop_text:
                            text = f"{text}\n{crop_text}".strip()
                            ocr_used = True

                if not text:
                    continue

                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": file_path,
                            "page": page_num,
                            "file_type": "pdf",
                            "page_hash": hashlib.md5(text.encode("utf-8")).hexdigest(),
                            "ocr_used": ocr_used,
                        },
                    )
                )
        return docs

    def _load_markdown(self, file_path: str) -> list[Document]:
        with open(file_path, "r", encoding="utf-8") as f:
            text = self._clean_text(f.read())
        if not text:
            return []
        return [
            Document(
                page_content=text,
                metadata={
                    "source": file_path,
                    "file_type": "md",
                    "page_hash": hashlib.md5(text.encode("utf-8")).hexdigest(),
                },
            )
        ]

    def _load_text(self, file_path: str) -> list[Document]:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = self._clean_text(f.read())
        if not text:
            return []
        return [
            Document(
                page_content=text,
                metadata={
                    "source": file_path,
                    "file_type": "txt",
                    "page_hash": hashlib.md5(text.encode("utf-8")).hexdigest(),
                },
            )
        ]

    def _load_word(self, file_path: str) -> list[Document]:
        doc = DocxDocument(file_path)
        paragraphs: list[str] = []

        # 1) 正文段落
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                paragraphs.append(text)

        # 2) 表格内容：每个单元格一行，保留语义关联
        for table in doc.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if cells:
                    paragraphs.append(" | ".join(cells))

        # 3) 页眉页脚
        for section in doc.sections:
            header = section.header
            footer = section.footer
            if header and not header.is_linked_to_previous:
                ht = " ".join(
                    p.text.strip() for p in header.paragraphs if p.text.strip()
                )
                if ht:
                    paragraphs.append(f"[页眉] {ht}")
            if footer and not footer.is_linked_to_previous:
                ft = " ".join(
                    p.text.strip() for p in footer.paragraphs if p.text.strip()
                )
                if ft:
                    paragraphs.append(f"[页脚] {ft}")

        text = self._clean_text("\n".join(paragraphs))
        if not text:
            return []

        return [
            Document(
                page_content=text,
                metadata={
                    "source": file_path,
                    "file_type": "docx",
                    "page_hash": hashlib.md5(text.encode("utf-8")).hexdigest(),
                },
            )
        ]

    def process_documents(self, documents: Iterable[Document]) -> list[Document]:
        """切分并绑定 chunk 级元数据。

        document_name 优先作为逻辑文档名，使 Web 上传文件替换后仍能稳定去重。
        """
        chunks = self.text_splitter.split_documents(list(documents))
        for idx, chunk in enumerate(chunks):
            source = chunk.metadata.get("source", "unknown")
            document_name = chunk.metadata.get("document_name", source)
            chunk_id = f"{document_name}#{idx}"
            chunk.metadata["chunk_id"] = chunk_id
            chunk.metadata["chunk_index"] = idx
            chunk.metadata["content_hash"] = hashlib.md5(
                chunk.page_content.encode("utf-8")
            ).hexdigest()
            chunk.id = chunk_id
        return chunks

    @staticmethod
    def _clean_text(text: str) -> str:
        # 去掉连续空行、行尾空白、单独的页码行（仅含数字）
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
        return text.strip()

    @staticmethod
    def _ocr_page(page) -> str:
        try:
            img = page.to_image(resolution=200).original
            return pytesseract.image_to_string(img, lang="chi_sim+eng")
        except Exception as e:
            print(f"[ocr] page-level ocr failed: {e}")
            return ""

    @staticmethod
    def _ocr_image_region(page, img_meta) -> str:
        try:
            x0, top, x1, bottom = (
                img_meta["x0"],
                img_meta["top"],
                img_meta["x1"],
                img_meta["bottom"],
            )
            cropped = (
                page.within((x0, top, x1, bottom)).to_image(resolution=200).original
            )
            text = pytesseract.image_to_string(cropped, lang="chi_sim+eng")
            return text.strip()
        except Exception as e:
            print(f"[ocr] region ocr failed: {e}")
            return ""
