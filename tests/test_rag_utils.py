from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rag_utils import (
    display_filename,
    resolve_within,
    retrieval_metrics,
    safe_upload_filename,
    tokenize_for_bm25,
)


class TokenizerTests(unittest.TestCase):
    def test_chinese_technical_terms_overlap(self) -> None:
        query_tokens = set(tokenize_for_bm25("默认端口是多少？"))
        document_tokens = set(tokenize_for_bm25("服务默认端口为 5000。"))
        self.assertTrue({"默认", "认端", "端口"} <= query_tokens)
        self.assertTrue({"默认", "端口"} <= document_tokens)

    def test_mixed_english_and_code_tokens(self) -> None:
        tokens = tokenize_for_bm25("调用 get_user_info，错误码 E4031。")
        self.assertIn("get_user_info", tokens)
        self.assertIn("e4031", tokens)


class PathSafetyTests(unittest.TestCase):
    def test_safe_upload_filename_removes_path_and_reserved_chars(self) -> None:
        self.assertEqual(
            safe_upload_filename('..\\docs\\a<b>:"c.pdf'),
            "a_b___c.pdf",
        )

    def test_display_filename_removes_content_prefix(self) -> None:
        self.assertEqual(
            display_filename("0123456789abcdef_报告.pdf"),
            "报告.pdf",
        )
        self.assertEqual(display_filename("报告.pdf"), "报告.pdf")

    def test_resolve_within_accepts_plain_filename(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            resolved = resolve_within(temp_dir, "docs.pdf")
            self.assertEqual(resolved, Path(temp_dir).resolve() / "docs.pdf")

    def test_resolve_within_rejects_traversal_and_nested_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            for filename in ("../secret.txt", r"..\secret.txt", "sub/docs.pdf"):
                with self.subTest(filename=filename):
                    with self.assertRaises(ValueError):
                        resolve_within(temp_dir, filename)


class RetrievalMetricTests(unittest.TestCase):
    def test_hit_rate_is_binary_and_recall_is_separate(self) -> None:
        metrics = retrieval_metrics(
            ["doc#2", "doc#3", "doc#1"],
            ["doc#1", "doc#2"],
            k=3,
        )
        self.assertEqual(metrics["hit_rate"], 1.0)
        self.assertEqual(metrics["recall_at_k"], 1.0)
        self.assertAlmostEqual(metrics["precision_at_k"], 2 / 3)
        self.assertEqual(metrics["mrr"], 1.0)

    def test_miss_has_zero_hit_rate(self) -> None:
        metrics = retrieval_metrics(["doc#2"], ["doc#1"], k=1)
        self.assertEqual(metrics["hit_rate"], 0.0)
        self.assertEqual(metrics["recall_at_k"], 0.0)
        self.assertEqual(metrics["mrr"], 0.0)


if __name__ == "__main__":
    unittest.main()
