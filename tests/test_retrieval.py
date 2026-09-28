import json
import unittest
from pathlib import Path
from retrieval import CampusRetriever

ROOT = Path(__file__).resolve().parents[1]

class RetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.r = CampusRetriever(ROOT / "data/corpus.json")

    def test_paraphrase_returns_grounded_excerpt(self):
        result = self.r.ask("My login passphrase slipped my mind; how can I recover my account?")
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["citations"][0]["doc_id"], "ITS-2026")
        self.assertIn(result["citations"][0]["text"], result["answer"])

    def test_missing_topic_abstains_without_citations(self):
        result = self.r.ask("What soup is served in the dining hall tomorrow?")
        self.assertEqual(result["status"], "abstained")
        self.assertEqual(result["citations"], [])

    def test_version_disagreement_is_explicit(self):
        result = self.r.ask("How many nights can an overnight housing guest stay?")
        self.assertEqual(result["status"], "conflict")
        self.assertEqual(result["citations"][0]["doc_id"], "HOU-2026")
        self.assertTrue(any(c["status"] == "archived" for c in result["citations"]))

    def test_eval_is_reproducible_and_partitioned(self):
        cases = json.loads((ROOT / "data/eval_questions.json").read_text())
        first, second = self.r.evaluate(cases), self.r.evaluate(cases)
        self.assertEqual(first, second)
        self.assertEqual(first["summary"]["total"], 30)
        self.assertEqual(set(first["summary"]["by_category"]), {"answerable","unanswerable","ambiguous_conflict"})

if __name__ == "__main__": unittest.main()
