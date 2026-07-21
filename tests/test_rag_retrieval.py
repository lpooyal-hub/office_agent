import json
import unittest
from pathlib import Path

from services.rag_service import parse_rule_match, rank_matches


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "rag_cases.json"


class RagRankingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    def test_rank_matches_promotes_expected_source_with_keyword_evidence(self):
        for case in self.cases:
            with self.subTest(case=case["name"]):
                ranked = rank_matches(case["question"], case["semantic_matches"])
                top = parse_rule_match(ranked[0])

                self.assertEqual(top["source"], case["expected_document"])
                self.assertGreaterEqual(top["semantic_score"], case["min_similarity"])
                self.assertGreater(top["lexical_score"], 0)
                self.assertGreaterEqual(top["rank_score"], top["semantic_score"])
                self.assertIn("source_kind", top)
                self.assertIn("chunk_index", top)

                evidence = f"{top['source']}\n{top['excerpt']}"
                for keyword in case["expected_keywords"]:
                    self.assertIn(keyword, evidence)

    def test_rank_matches_returns_score_breakdown_without_mutating_input(self):
        original = {
            "chunk": "[휴가 규정]\n연차 휴가 안내",
            "semantic_score": 0.4,
            "source_kind": "document",
            "chunk_index": 7,
        }

        ranked = rank_matches("연차 휴가", [original])

        self.assertNotIn("lexical_score", original)
        self.assertEqual(ranked[0]["source_kind"], "document")
        self.assertEqual(ranked[0]["chunk_index"], 7)
        self.assertAlmostEqual(ranked[0]["semantic_score"], 0.4)
        self.assertGreater(ranked[0]["lexical_score"], 0)
        self.assertAlmostEqual(
            ranked[0]["rank_score"],
            ranked[0]["semantic_score"] + ranked[0]["lexical_score"],
        )


if __name__ == "__main__":
    unittest.main()
