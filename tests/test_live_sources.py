import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError

from live_sources import LiveKnowledge


ROBOTS = b"User-agent: *\nAllow: /\n"
URL_ONE = "https://www.glsuniversity.ac.in/fcait.html"
URL_TWO = "https://www.glsuniversity.ac.in/announcement.html"
ROBOTS_URL = "https://www.glsuniversity.ac.in/robots.txt"


def page(subject, detail):
    return ("<html><header>Repeated navigation that must disappear</header><main>"
            f"<h1>{subject}</h1><p>{detail}</p>"
            "<script>secret_script_noise()</script></main><footer>Footer links</footer></html>").encode()


class FakeFetcher:
    def __init__(self, pages):
        self.pages = pages
        self.fail = set()
        self.redirects = {}
        self.calls = []

    def __call__(self, url, timeout, max_bytes, allowed_redirects):
        self.calls.append((url, tuple(sorted(allowed_redirects))))
        if url in self.fail:
            raise URLError("temporary outage")
        if url == ROBOTS_URL:
            return {"body": ROBOTS, "content_type": "text/plain", "final_url": url}
        return {"body": self.pages[url], "content_type": "text/html",
                "final_url": self.redirects.get(url, url)}


class FakeEmbedder:
    def __init__(self):
        self.calls = []

    def __call__(self, texts, model):
        self.calls.append((list(texts), model))
        return [[1.0, 0.0] if "scholarship" in text.lower() else [0.0, 1.0] for text in texts]


class LiveKnowledgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = self.root / "sources.json"
        self.index = self.root / "live-index.json"

    def tearDown(self):
        self.temp.cleanup()

    def write_config(self, two=False, embedding_threshold=0.48):
        sources = [{"id": "FCAIT", "title": "FCAIT", "url": URL_ONE}]
        if two:
            sources.append({"id": "NEWS", "title": "Announcements", "url": URL_TWO})
        self.config.write_text(json.dumps({"timeout_seconds": 1, "max_bytes": 100000,
            "chunk_chars": 500, "lexical_threshold": 0.25,
            "embedding_threshold": embedding_threshold, "sources": sources}))

    def test_unchanged_refresh_reuses_chunk_and_embedding_cache(self):
        self.write_config()
        fetcher = FakeFetcher({URL_ONE: page("Scholarship notice", "Scholarship applications require a completed student form and faculty approval before the published closing date.")})
        embedder = FakeEmbedder()
        with patch.dict(os.environ, {"GLS_EMBED_MODEL": "embeddinggemma"}):
            knowledge = LiveKnowledge(self.config, self.index, fetcher, embedder)
            knowledge.refresh()
            first = knowledge.document("FCAIT")
            first_changed = knowledge.status()["sources"][0]["last_changed"]
            embedding_calls = len(embedder.calls)
            knowledge.refresh()
        self.assertEqual(knowledge.document("FCAIT")["sections"], first["sections"])
        self.assertEqual(knowledge.status()["sources"][0]["last_changed"], first_changed)
        self.assertEqual(len(embedder.calls), embedding_calls)
        self.assertNotIn("secret_script_noise", first["sections"][0]["text"])

    def test_changed_content_replaces_chunks(self):
        self.write_config()
        fetcher = FakeFetcher({URL_ONE: page("Academic calendar", "The autumn registration calendar is available to enrolled students through the academic services portal.")})
        with patch.dict(os.environ, {"GLS_EMBED_MODEL": ""}, clear=False):
            knowledge = LiveKnowledge(self.config, self.index, fetcher)
            knowledge.refresh()
            old_ids = [section["id"] for section in knowledge.document("FCAIT")["sections"]]
            fetcher.pages[URL_ONE] = page("Revised calendar", "The revised winter registration calendar lists a new closing date and updated verification instructions for students.")
            knowledge.refresh()
        new_doc = knowledge.document("FCAIT")
        self.assertNotEqual(old_ids, [section["id"] for section in new_doc["sections"]])
        self.assertIn("winter", new_doc["sections"][0]["text"].lower())

    def test_failed_refresh_retains_last_successful_snapshot(self):
        self.write_config()
        fetcher = FakeFetcher({URL_ONE: page("Programme notice", "Students can review programme notices and academic requirements through the approved faculty information page.")})
        with patch.dict(os.environ, {"GLS_EMBED_MODEL": ""}, clear=False):
            knowledge = LiveKnowledge(self.config, self.index, fetcher)
            knowledge.refresh()
            sections = knowledge.document("FCAIT")["sections"]
            fetcher.fail.add(URL_ONE)
            status = knowledge.refresh()["sources"][0]
        self.assertEqual(status["status"], "error")
        self.assertIn("URLError", status["error"])
        self.assertEqual(knowledge.document("FCAIT")["sections"], sections)

    def test_unsafe_redirect_is_rejected(self):
        self.write_config()
        fetcher = FakeFetcher({URL_ONE: page("Programme notice", "This page contains sufficient approved programme information for a readable source snapshot and retrieval test.")})
        fetcher.redirects[URL_ONE] = "https://evil.example/policy"
        with patch.dict(os.environ, {"GLS_EMBED_MODEL": ""}, clear=False):
            knowledge = LiveKnowledge(self.config, self.index, fetcher)
            source = knowledge.refresh()["sources"][0]
        self.assertEqual(source["status"], "error")
        self.assertIn("allowlist", source["error"])
        self.assertEqual(source["chunks"], 0)

    def test_real_provider_path_sets_embedding_mode_and_live_citation(self):
        self.write_config(two=True, embedding_threshold=0.4)
        fetcher = FakeFetcher({
            URL_ONE: page("Scholarship applications", "Scholarship applicants submit verified academic documents and the completed application before the announced closing date."),
            URL_TWO: page("Cultural programme", "The annual cultural programme includes student performances, club exhibitions, and an evening recognition ceremony."),
        })
        embedder = FakeEmbedder()
        with patch.dict(os.environ, {"GLS_EMBED_MODEL": "nomic-embed-text"}):
            knowledge = LiveKnowledge(self.config, self.index, fetcher, embedder)
            knowledge.refresh()
            result = knowledge.ask("What documents are needed for the scholarship?")
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["mode"], "live-embedding")
        self.assertEqual(result["citations"][0]["source_url"], URL_ONE)
        self.assertTrue(result["citations"][0]["fetched_at"])
        self.assertTrue(knowledge.status()["embedding"]["available"])

    def test_unsupported_question_abstains_in_lexical_mode(self):
        self.write_config()
        fetcher = FakeFetcher({URL_ONE: page("Programme notice", "Students can review programme notices and academic requirements through the approved faculty information page.")})
        with patch.dict(os.environ, {"GLS_EMBED_MODEL": ""}, clear=False):
            knowledge = LiveKnowledge(self.config, self.index, fetcher)
            knowledge.refresh()
            result = knowledge.ask("Which bus route reaches the railway station?")
        self.assertEqual(result["status"], "abstained")
        self.assertEqual(result["mode"], "live-lexical")
        self.assertEqual(result["citations"], [])

    def test_lexical_answer_skips_higher_single_term_candidate(self):
        self.write_config(two=True)
        fetcher = FakeFetcher({
            URL_ONE: page("Admissions", "Admissions information and unrelated general material is available for prospective learners considering different programmes."),
            URL_TWO: page("University announcements", "University admission announcements publish application dates and current document requirements for prospective students."),
        })
        with patch.dict(os.environ, {"GLS_EMBED_MODEL": ""}, clear=False):
            knowledge = LiveKnowledge(self.config, self.index, fetcher)
            knowledge.refresh()
            pairs = knowledge._all_chunks()
            first = next(pair for pair in pairs if pair[0]["id"] == "FCAIT")
            second = next(pair for pair in pairs if pair[0]["id"] == "NEWS")
            ranked = [(0.9, 1, first), (0.55, 2, second)]
            with patch.object(knowledge, "_lexical_rank", return_value=ranked):
                result = knowledge.ask("current admission announcements")
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["citations"][0]["doc_id"], "NEWS")
        self.assertEqual(len(result["citations"]), 1)

    def test_config_rejects_non_gls_hosts(self):
        self.config.write_text(json.dumps({"sources": [{"id": "BAD", "title": "Bad", "url": "https://evil.example/arbitrary.html"}]}))
        with self.assertRaisesRegex(ValueError, "approved GLS University"):
            LiveKnowledge(self.config, self.index)

    def test_config_accepts_additional_official_pdf_url(self):
        self.config.write_text(json.dumps({"sources": [{"id": "PDF", "title": "Official PDF", "url": "https://www.glsuniversity.ac.in/documents/latest-policy.pdf"}]}))
        knowledge = LiveKnowledge(self.config, self.index)
        self.assertEqual(knowledge.documents()[0]["url"], "https://www.glsuniversity.ac.in/documents/latest-policy.pdf")

    def test_bad_query_vector_falls_back_to_lexical(self):
        self.write_config()
        fetcher = FakeFetcher({URL_ONE: page("Scholarship applications", "Scholarship applicants submit verified academic documents and the completed application before the announced closing date.")})
        embedder = FakeEmbedder()
        with patch.dict(os.environ, {"GLS_EMBED_MODEL": "embeddinggemma"}):
            knowledge = LiveKnowledge(self.config, self.index, fetcher, embedder)
            knowledge.refresh()
            embedder.__call__ = lambda texts, model: [[1.0, 0.0, 0.0]]
            # Special methods are resolved on the class, so replace the provider directly.
            knowledge.embedder = lambda texts, model: [[1.0, 0.0, 0.0]]
            result = knowledge.ask("scholarship academic documents")
        self.assertEqual(result["mode"], "live-lexical")
        self.assertFalse(knowledge.status()["embedding"]["available"])


if __name__ == "__main__":
    unittest.main()
