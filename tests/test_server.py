import http.client
import json
import threading
import unittest
from unittest.mock import patch
from http.server import ThreadingHTTPServer

from app import Handler


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=2)
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        payload = json.loads(response.read())
        connection.close()
        return response.status, payload

    def test_health_contract(self):
        status, payload = self.request("GET", "/api/health")
        self.assertEqual(status, 200)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["mode"], "offline-extractive")

    def test_answer_contract_includes_exact_citations(self):
        body = json.dumps({"question": "Which wifi should my laptop connect to?"})
        status, payload = self.request("POST", "/api/ask", body, {"Content-Type": "application/json"})
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "answered")
        self.assertTrue(payload["citations"])
        self.assertIn(payload["citations"][0]["text"], payload["answer"])

    def test_non_object_json_is_rejected(self):
        status, payload = self.request("POST", "/api/ask", "[]", {"Content-Type": "application/json"})
        self.assertEqual(status, 400)
        self.assertIn("object", payload["error"])

    def test_negative_content_length_is_rejected_without_reading(self):
        status, payload = self.request("POST", "/api/ask", None, {"Content-Length": "-1", "Content-Type": "application/json"})
        self.assertEqual(status, 400)
        self.assertIn("non-empty", payload["error"])

    def test_collections_are_separate(self):
        with patch("app.LIVE") as live:
            live.documents.return_value = [{"id": "official-page"}]
            status, payload = self.request("GET", "/api/documents?collection=live")
            self.assertEqual(status, 200)
            self.assertEqual(payload["documents"], [{"id": "official-page"}])
            status, demo = self.request("GET", "/api/documents")
            self.assertEqual(len(demo["documents"]), 7)
            live.documents.assert_called_once()

    def test_unknown_collection_rejected(self):
        status, _ = self.request("POST", "/api/ask", json.dumps({"collection": "unknown", "question": "hello"}), {"Content-Type": "application/json"})
        self.assertEqual(status, 400)

    def test_refresh_starts_background_job(self):
        with patch("app.start_refresh") as refresh, patch("app.live_status", return_value={"refreshing": True}):
            status, payload = self.request("POST", "/api/live/refresh", "{}", {"Content-Type": "application/json"})
            self.assertEqual(status, 202)
            self.assertTrue(payload["refreshing"])
            refresh.assert_called_once()

    def test_external_site_cannot_trigger_refresh(self):
        with patch("app.start_refresh") as refresh:
            status, _ = self.request("POST", "/api/live/refresh", "{}", {"Content-Type": "application/json", "Origin": "https://example.org"})
            self.assertEqual(status, 403)
            refresh.assert_not_called()


if __name__ == "__main__":
    unittest.main()
