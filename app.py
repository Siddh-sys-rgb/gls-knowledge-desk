#!/usr/bin/env python3
"""GLS Knowledge Desk: zero-dependency local web server."""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen
from urllib.parse import parse_qs, urlsplit

from retrieval import CampusRetriever
from live_sources import LiveKnowledge

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
DATA = ROOT / "data"
RETRIEVER = CampusRetriever(DATA / "corpus.json")
LIVE = LiveKnowledge(DATA / "live_sources.json", DATA / "live_index.json")
REFRESH_LOCK = threading.Lock()
REFRESH_MINUTES = 0
REFRESH_ERROR = None


def live_status():
    result = LIVE.status()
    result.update(refreshing=REFRESH_LOCK.locked(), auto_refresh_minutes=REFRESH_MINUTES,
                  refresh_error=REFRESH_ERROR)
    return result


def start_refresh():
    """A single background job keeps the UI responsive during bounded remote fetches."""
    if not REFRESH_LOCK.acquire(blocking=False):
        return False
    def run():
        global REFRESH_ERROR
        try:
            LIVE.refresh()
            REFRESH_ERROR = None
        except Exception as exc:
            REFRESH_ERROR = f"Refresh failed ({type(exc).__name__}); retry after checking the server log."
            print(f"Live refresh failed: {exc}")
        finally:
            REFRESH_LOCK.release()
    threading.Thread(target=run, daemon=True).start()
    return True


def send_json(handler, payload, status=200):
    body = json.dumps(payload, ensure_ascii=False).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def ollama_generate(question, result, model):
    """Optional localhost-only grounded rewrite. Retrieval remains the authority."""
    allowed = [c["id"] for c in result.get("citations", [])]
    context = "\n\n".join(f'[{c["id"]}] {c["text"]}' for c in result.get("citations", []))
    prompt = (
        "You rewrite an extractive campus-policy answer. Use ONLY the supplied excerpts. "
        "Every factual sentence must end with one supplied source ID in brackets. "
        "Do not resolve conflicts or add facts. If the retrieval status is not answered, preserve that status.\n"
        f"Status: {result['status']}\nQuestion: {question}\nExcerpts:\n{context}\nAnswer:"
    )
    req = Request(
        "http://127.0.0.1:11434/api/generate",
        data=json.dumps({"model": model, "prompt": prompt, "stream": False}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=15) as response:
            text = json.loads(response.read())["response"].strip()
    except (URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
        return None, f"Ollama unavailable: {type(exc).__name__}"
    cited = set(re.findall(r"\[([A-Za-z0-9-]+)\]", text))
    if not cited or not cited.issubset(set(allowed)):
        return None, "Ollama response failed citation validation; showing extractive answer."
    return text, None


class Handler(SimpleHTTPRequestHandler):
    server_version = "GLSKnowledgeDesk/1.0"

    def log_message(self, fmt, *args):
        print(f"[campus-faq] {self.address_string()} {fmt % args}")

    def do_GET(self):
        url = urlsplit(self.path)
        path = url.path
        collection = parse_qs(url.query).get("collection", ["demo"])[0]
        if collection not in ("demo", "live"):
            return send_json(self, {"error": "Unknown collection."}, 400)
        retriever = LIVE if collection == "live" else RETRIEVER
        if path == "/api/health":
            return send_json(self, {"ok": True, "documents": RETRIEVER.document_count, "mode": "offline-extractive"})
        if path == "/api/live/status":
            return send_json(self, live_status())
        if path == "/api/documents":
            return send_json(self, {"documents": retriever.documents(), "collection": collection})
        if path.startswith("/api/documents/"):
            doc_id = path.rsplit("/", 1)[-1]
            doc = retriever.document(doc_id)
            return send_json(self, doc or {"error": "Document not found"}, 200 if doc else 404)
        if path == "/api/evaluate":
            questions = json.loads((DATA / "eval_questions.json").read_text())
            return send_json(self, RETRIEVER.evaluate(questions))
        path = "/index.html" if self.path == "/" else self.path.split("?", 1)[0]
        target = (WEB / path.lstrip("/")).resolve()
        if WEB.resolve() not in target.parents or not target.is_file():
            return self.send_error(404)
        body = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(target.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self' http://127.0.0.1:11434")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path not in ("/api/ask", "/api/live/refresh"):
            return send_json(self, {"error": "Not found"}, 404)
        origin = self.headers.get("Origin")
        if origin and origin != "http://" + self.headers.get("Host", ""):
            return send_json(self, {"error": "Cross-origin requests are not allowed."}, 403)
        if self.headers.get("Content-Type", "").split(";", 1)[0].strip() != "application/json":
            return send_json(self, {"error": "Use application/json."}, 415)
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size <= 0:
                return send_json(self, {"error": "A non-empty JSON request body is required."}, 400)
            if size > 10_000:
                return send_json(self, {"error": "Request too large"}, 413)
            data = json.loads(self.rfile.read(size))
            if not isinstance(data, dict):
                return send_json(self, {"error": "JSON body must be an object."}, 400)
            if self.path == "/api/live/refresh":
                start_refresh()
                return send_json(self, live_status(), 202)
            collection = data.get("collection", "demo")
            if collection not in ("demo", "live"):
                return send_json(self, {"error": "Unknown collection."}, 400)
            question = str(data.get("question", "")).strip()
            if not question or len(question) > 500:
                return send_json(self, {"error": "Question must be 1–500 characters."}, 400)
            result = (LIVE if collection == "live" else RETRIEVER).ask(question)
            result["collection"] = collection
            result.setdefault("mode", "offline-extractive")
            if data.get("mode") == "ollama" and result["status"] == "answered" and result["citations"]:
                generated, error = ollama_generate(question, result, str(data.get("model", "llama3.2")))
                if generated:
                    result["extractive_answer"] = result["answer"]
                    result["answer"] = generated
                    result["retrieval_mode"] = result["mode"]
                    result["mode"] += "+ollama-rewrite-unverified"
                    result["generation_warning"] = "Locally generated wording is unverified. Check every statement against the excerpts below."
                if error:
                    result["generation_warning"] = error
            return send_json(self, result)
        except (ValueError, json.JSONDecodeError):
            return send_json(self, {"error": "Invalid JSON request."}, 400)


def main():
    parser = argparse.ArgumentParser(description="Run the local Campus FAQ demo")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8101")))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--refresh-minutes", type=int, default=0,
                        help="Refresh configured public sources on startup and every N minutes (0 disables).")
    args = parser.parse_args()
    if args.refresh_minutes < 0 or 0 < args.refresh_minutes < 5:
        parser.error("--refresh-minutes must be 0 (off) or at least 5.")
    global REFRESH_MINUTES
    REFRESH_MINUTES = args.refresh_minutes
    stop = threading.Event()
    if REFRESH_MINUTES:
        start_refresh()
        def scheduled_refresh():
            while not stop.wait(REFRESH_MINUTES * 60):
                start_refresh()
        threading.Thread(target=scheduled_refresh, daemon=True).start()
    print(f"GLS Knowledge Desk → http://{args.host}:{args.port}")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        server.server_close()


if __name__ == "__main__":
    main()
