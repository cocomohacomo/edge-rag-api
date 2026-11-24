import tempfile
import unittest
from pathlib import Path

from rag.core import Store
from rag.retrieval_v2 import expand_query, search_with_diagnostics


class RetrievalV2Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.tmp.name) / "retrieval.db")
        docs = [
            ("index.md", "登録文書をチャンク化してFTS5に登録する。チャンクには元文書名と位置情報を保持する。"),
            ("search.md", "日本語文書の全文検索にはSQLite FTS5を使用する。日本語テキストは2-gramに分割して索引化する。"),
            ("memory.md", "8GB RAM内でモデルと検索DBを共存させる。回答生成時のピークメモリを測定する。"),
            ("unrelated.md", "会議の担当者は佐藤。期限は12月15日。"),
        ]
        for name, text in docs:
            path = Path(self.tmp.name) / name
            self.store.add_document(path, text)

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_expansion_adds_indexing_terms_without_replacing_query(self):
        expanded = expand_query("検索インデックスの構築方法")
        self.assertTrue(set(expanded) >= set(("検索", "FTS5", "チャンク化")))

    def test_index_query_finds_index_document(self):
        hits, diag = search_with_diagnostics(self.store, "検索インデックスはどのように構築しますか", 5)
        self.assertTrue(hits)
        self.assertEqual(hits[0]["source"], "index.md")
        self.assertIn("fts5_bm25", diag["channels"])
        self.assertIn("bigram", diag["channels"])
        self.assertIn("exact_match", diag["channels"])

    def test_multi_evidence_query_keeps_memory_and_db_evidence(self):
        hits, _ = search_with_diagnostics(self.store, "8GBメモリ環境でモデルと検索DBを共存できますか", 5)
        sources = {h["source"] for h in hits}
        self.assertIn("memory.md", sources)

    def test_diagnostics_explain_empty_query(self):
        hits, diag = search_with_diagnostics(self.store, "   ", 5)
        self.assertEqual(hits, [])
        self.assertIn("QUERY_TOKENIZATION_FAILURE", diag["failure_classification"])

    def test_rrf_exposes_channel_consensus(self):
        hits, _ = search_with_diagnostics(self.store, "SQLite FTS5 検索", 5)
        self.assertTrue(hits)
        self.assertGreaterEqual(hits[0]["channel_count"], 2)


if __name__ == "__main__":
    unittest.main()
