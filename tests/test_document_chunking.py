import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.modules.setdefault("numpy", types.SimpleNamespace(array=lambda values: values, dot=lambda *args: []))

from services.document_library import split_text
from services.rag_service import DocumentRetriever, parse_rule_match


class FakeEmbedding(list):
    def tolist(self):
        return list(self)


class FakeEmbedder:
    def encode(self, texts, normalize_embeddings=True):
        return [FakeEmbedding([1.0, 0.0]) for _ in texts]


class FakeRetriever(DocumentRetriever):
    def get_embedder(self):
        return FakeEmbedder()


class DocumentChunkingTests(unittest.TestCase):
    def test_split_text_adds_section_metadata_and_offsets(self):
        text = "# 인사 규정\n제1조 목적\n첫 문장입니다. 둘째 문장입니다.\n## 휴가\n- 연차 휴가"

        chunks = split_text(text, max_chars=35, overlap_chars=0)

        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual(chunks[0].section_title, "인사 규정")
        self.assertTrue(any(chunk.section_title == "제1조 목적" for chunk in chunks))
        self.assertEqual(chunks[0].chunk_index, 0)
        self.assertEqual(chunks[0].char_start, 0)
        self.assertLessEqual(len(chunks[0].text), 35)
        self.assertEqual(chunks[-1].section_title, "- 연차 휴가")
        self.assertEqual(text[chunks[-1].char_start:chunks[-1].char_end], chunks[-1].text)

    def test_split_text_re_splits_oversized_paragraph(self):
        text = "가" * 95

        chunks = split_text(text, max_chars=40, overlap_chars=0)

        self.assertEqual([len(chunk.text) for chunk in chunks], [40, 40, 15])
        self.assertEqual([(chunk.char_start, chunk.char_end) for chunk in chunks], [(0, 40), (40, 80), (80, 95)])

    def test_build_records_stores_chunk_metadata(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "policy.md"
            path.write_text("# 총칙\n본문입니다.", encoding="utf-8")
            retriever = FakeRetriever(path.parent, "fake", [], chroma_store=None)

            records = retriever.build_records_for_document(path)

        self.assertEqual(records[0]["metadata"]["section_title"], "총칙")
        self.assertEqual(records[0]["metadata"]["chunk_index"], 0)
        self.assertEqual(records[0]["metadata"]["char_start"], 0)
        self.assertGreater(records[0]["metadata"]["char_end"], 0)

    def test_parse_rule_match_exposes_location(self):
        parsed = parse_rule_match("[policy.md] (휴가 · chunk 2 · chars 10-20)\n본문", 0.7)

        self.assertEqual(parsed["source"], "policy.md")
        self.assertEqual(parsed["location"], "휴가 · chunk 2 · chars 10-20")
        self.assertEqual(parsed["excerpt"], "본문")


if __name__ == "__main__":
    unittest.main()
