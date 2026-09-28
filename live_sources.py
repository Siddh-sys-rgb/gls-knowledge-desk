"""Safe, optional live-source retrieval for GLS Knowledge Desk.

The default demo corpus remains separate. This module refreshes only an explicit
allowlist, stores last-good snapshots, and uses Ollama embeddings only when the
operator explicitly enables a supported model.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import tempfile
import threading
from collections import Counter
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen
from urllib.robotparser import RobotFileParser

SUPPORTED_MODELS = {"embeddinggemma", "nomic-embed-text"}
TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {"a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "i", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "was", "what", "when", "where", "which", "who", "with"}


def _now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _hash(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _tokens(value):
    return [word for word in TOKEN_RE.findall(value.lower()) if word not in STOPWORDS and len(word) > 1]


class _TextExtractor(HTMLParser):
    SKIP = {"script", "style", "nav", "header", "footer", "form", "svg", "noscript", "aside"}
    BLOCKS = {"p", "li", "h1", "h2", "h3", "h4", "td", "th", "article", "section", "br"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.skip_depth = 0
        self.parts = []
        self.heading_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip_depth += 1
        elif not self.skip_depth and tag in self.BLOCKS:
            self.parts.append("\n§" if tag.startswith("h") and len(tag) == 2 else "\n")
            if tag.startswith("h") and len(tag) == 2:
                self.heading_depth += 1

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.skip_depth:
            self.skip_depth -= 1
        elif not self.skip_depth and tag in self.BLOCKS:
            self.parts.append("\n")
            if tag.startswith("h") and len(tag) == 2 and self.heading_depth:
                self.heading_depth -= 1

    def handle_data(self, data):
        if not self.skip_depth:
            self.parts.append(data)

    def text(self):
        lines, seen = [], set()
        for raw in "".join(self.parts).splitlines():
            line = re.sub(r"\s+", " ", raw).strip()
            heading = line.startswith("§")
            line = line.lstrip("§").strip()
            key = line.lower()
            short_label = bool(re.fullmatch(r"[A-Z][A-Za-z0-9&()./,+ -]{1,50}", line))
            boilerplate = key in {"home", "about us", "contact us", "read more", "click here", "menu", "search"}
            if (len(line) >= 25 or heading or short_label) and not boilerplate and key not in seen:
                seen.add(key)
                lines.append(line)
        return "\n".join(lines)


class _CheckedRedirects(HTTPRedirectHandler):
    def __init__(self, allowed):
        super().__init__()
        self.allowed = allowed

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        candidate = urljoin(req.full_url, newurl)
        if candidate not in self.allowed:
            raise URLError("redirect target is outside the approved URL set")
        return super().redirect_request(req, fp, code, msg, headers, candidate)


def _default_fetch(url, timeout, max_bytes, allowed_redirects):
    opener = build_opener(_CheckedRedirects(set(allowed_redirects)))
    request = Request(url, headers={"User-Agent": "GLSKnowledgeDeskPortfolio/1.0"})
    with opener.open(request, timeout=timeout) as response:
        length = response.headers.get("Content-Length")
        if length and int(length) > max_bytes:
            raise ValueError("source exceeds configured size limit")
        body = response.read(max_bytes + 1)
        if len(body) > max_bytes:
            raise ValueError("source exceeds configured size limit")
        return {"body": body, "content_type": response.headers.get_content_type(), "final_url": response.geturl()}


def _ollama_embeddings(texts, model):
    request = Request(
        "http://127.0.0.1:11434/api/embed",
        data=json.dumps({"model": model, "input": texts}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=12) as response:
        payload = json.loads(response.read())
    vectors = payload.get("embeddings")
    if not isinstance(vectors, list) or len(vectors) != len(texts):
        raise ValueError("Ollama returned an invalid embeddings payload")
    if any(not isinstance(vector, list) or not vector for vector in vectors):
        raise ValueError("Ollama returned an empty embedding")
    return vectors


class LiveKnowledge:
    def __init__(self, config_path, index_path, fetcher=None, embedder=None):
        self.config_path = Path(config_path)
        self.index_path = Path(index_path)
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        self._lock = threading.RLock()
        self._refresh_lock = threading.Lock()
        self.sources = self.config.get("sources", [])
        self.allowed_urls = {source["url"] for source in self.sources}
        self._validate_config()
        self.fetcher = fetcher or _default_fetch
        self.embedder = embedder or _ollama_embeddings
        requested = os.getenv("GLS_EMBED_MODEL", "").strip()
        self.model = requested if requested in SUPPORTED_MODELS else ""
        self.embedding_enabled = bool(self.model)
        self.embedding_available = False
        self.embedding_message = "Embeddings disabled; using lexical retrieval."
        if requested and not self.model:
            self.embedding_message = "Unsupported GLS_EMBED_MODEL; using lexical retrieval."
        elif self.model:
            self.embedding_message = "Enabled but not yet checked. Refresh sources to build embeddings."
        self.index = self._load_index()
        configured = {source["id"]: source["url"] for source in self.sources}
        self.index["sources"] = {
            source_id: snapshot for source_id, snapshot in self.index.get("sources", {}).items()
            if configured.get(source_id) == snapshot.get("url")
        }
        if self.index.get("embedding_model") != self.model:
            self.index["embedding_cache"] = {}
            for snapshot in self.index.get("sources", {}).values():
                for chunk in snapshot.get("chunks", []):
                    chunk.pop("embedding", None)
            self.index["embedding_model"] = self.model
        elif self.embedding_enabled:
            cached_chunks = [chunk for snapshot in self.index.get("sources", {}).values()
                             for chunk in snapshot.get("chunks", []) if chunk.get("embedding")]
            if cached_chunks:
                try:
                    cleaned = self._validated_vectors([chunk["embedding"] for chunk in cached_chunks], len(cached_chunks))
                    for chunk, vector in zip(cached_chunks, cleaned):
                        chunk["embedding"] = vector
                    self.embedding_available = True
                    self.embedding_message = "Cached Ollama embeddings ready."
                except (TypeError, ValueError, OverflowError):
                    for snapshot in self.index.get("sources", {}).values():
                        for chunk in snapshot.get("chunks", []):
                            chunk.pop("embedding", None)
                    self.index["embedding_cache"] = {}
                    self.embedding_message = "Cached embeddings were invalid and will be rebuilt on refresh."

    def _validate_config(self):
        if not self.sources:
            raise ValueError("live source config has no sources")
        ids = set()
        timeout = int(self.config.get("timeout_seconds", 10))
        max_bytes = int(self.config.get("max_bytes", 2_000_000))
        chunk_chars = int(self.config.get("chunk_chars", 1200))
        if not 1 <= timeout <= 30 or not 10_000 <= max_bytes <= 10_000_000 or not 200 <= chunk_chars <= 4_000:
            raise ValueError("live source limits are outside the safe range")
        for source in self.sources:
            parsed = urlparse(source.get("url", ""))
            if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.fragment:
                raise ValueError("every live source must be a public HTTPS URL")
            try:
                port = parsed.port
            except ValueError as exc:
                raise ValueError("live source URL has an invalid port") from exc
            if parsed.hostname.lower() != "www.glsuniversity.ac.in" or port not in (None, 443):
                raise ValueError("live source must use the approved GLS University HTTPS host")
            if source.get("id") in ids:
                raise ValueError("duplicate live source id")
            ids.add(source.get("id"))

    def _load_index(self):
        empty = {"version": 1, "updated_at": None, "embedding_model": self.model, "embedding_cache": {}, "sources": {}}
        if not self.index_path.exists():
            return empty
        try:
            saved = json.loads(self.index_path.read_text(encoding="utf-8"))
            return saved if saved.get("version") == 1 else empty
        except (OSError, ValueError):
            return empty

    def _save(self):
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            payload = json.dumps(self.index, ensure_ascii=False, indent=2)
        handle, name = tempfile.mkstemp(prefix=".live-index-", suffix=".json", dir=str(self.index_path.parent))
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                stream.write(payload)
            os.replace(name, self.index_path)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    def _call_fetcher(self, url, max_bytes, redirects):
        response = self.fetcher(url, int(self.config.get("timeout_seconds", 10)), max_bytes, redirects)
        if not isinstance(response, dict) or not isinstance(response.get("body"), (bytes, bytearray)):
            raise ValueError("fetcher returned an invalid response")
        return response

    def _robots_allowed(self, source_url):
        parsed = urlparse(source_url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        try:
            response = self._call_fetcher(robots_url, 512_000, {robots_url})
            if response.get("final_url", robots_url) != robots_url:
                return False, "robots redirect was not approved"
            parser = RobotFileParser()
            parser.set_url(robots_url)
            parser.parse(response["body"].decode("utf-8", errors="replace").splitlines())
            return parser.can_fetch("GLSKnowledgeDeskPortfolio/1.0", source_url), None
        except HTTPError as exc:
            if exc.code == 404:
                return True, None
            return False, f"robots check failed with HTTP {exc.code}"
        except Exception as exc:
            return False, f"robots check failed: {type(exc).__name__}"

    def _extract(self, response, source):
        content_type = str(response.get("content_type", "")).lower()
        body = bytes(response["body"])
        if "pdf" in content_type or source["url"].lower().endswith(".pdf"):
            try:
                from pypdf import PdfReader
            except ImportError as exc:
                raise RuntimeError("PDF source requires optional dependency: pip install -r requirements-live.txt") from exc
            reader = PdfReader(io.BytesIO(body))
            text = "\n".join((page.extract_text() or "") for page in reader.pages)
        else:
            extractor = _TextExtractor()
            extractor.feed(body.decode("utf-8", errors="replace"))
            text = extractor.text()
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if len(text) < 80:
            raise ValueError("source did not contain enough readable policy text")
        return text

    def _chunk(self, source_id, text):
        limit = int(self.config.get("chunk_chars", 1200))
        paragraphs = [part.strip() for part in text.splitlines() if part.strip()]
        chunks, current = [], ""
        for paragraph in paragraphs:
            if current and len(current) + len(paragraph) + 1 > limit:
                chunks.append(current)
                current = ""
            if len(paragraph) > limit:
                if current:
                    chunks.append(current); current = ""
                chunks.extend(paragraph[pos:pos + limit] for pos in range(0, len(paragraph), limit))
            else:
                current = f"{current}\n{paragraph}".strip()
        if current:
            chunks.append(current)
        output = []
        for number, chunk in enumerate(chunks, 1):
            digest = _hash(chunk)
            output.append({"id": f"{source_id}-L{number}-{digest[:8]}", "hash": digest, "text": chunk})
        return output

    def _ensure_embeddings(self):
        if not self.embedding_enabled:
            return
        with self._lock:
            cache = self.index.setdefault("embedding_cache", {})
            missing = []
            for snapshot in self.index["sources"].values():
                for chunk in snapshot.get("chunks", []):
                    cached = cache.get(chunk["hash"])
                    if cached:
                        chunk["embedding"] = cached
                    elif "embedding" not in chunk:
                        missing.append(chunk)
        if not missing:
            self.embedding_available = any(c.get("embedding") for s in self.index["sources"].values() for c in s.get("chunks", []))
            self.embedding_message = "Ollama embeddings ready." if self.embedding_available else "No live chunks available to embed."
            return
        try:
            vectors = self._validated_vectors(self.embedder([chunk["text"] for chunk in missing], self.model), len(missing))
            with self._lock:
                for chunk, values in zip(missing, vectors):
                    chunk["embedding"] = values
                    cache[chunk["hash"]] = values
            self.embedding_available = True
            self.embedding_message = "Ollama embeddings ready."
        except Exception as exc:
            self.embedding_available = False
            self.embedding_message = f"Ollama embeddings unavailable ({type(exc).__name__}); using lexical retrieval."

    @staticmethod
    def _validated_vectors(vectors, expected_count, expected_dimension=None):
        if not isinstance(vectors, list) or len(vectors) != expected_count:
            raise ValueError("embedding provider returned the wrong vector count")
        cleaned, dimension = [], expected_dimension
        for vector in vectors:
            if not isinstance(vector, list) or not vector:
                raise ValueError("embedding provider returned an empty vector")
            values = [float(value) for value in vector]
            if any(not math.isfinite(value) for value in values) or not any(value != 0 for value in values):
                raise ValueError("embedding provider returned an invalid vector")
            if dimension is None:
                dimension = len(values)
            if len(values) != dimension:
                raise ValueError("embedding provider returned inconsistent dimensions")
            cleaned.append(values)
        return cleaned

    def refresh(self):
        with self._refresh_lock:
            return self._refresh_locked()

    def _refresh_locked(self):
        checked = _now()
        max_bytes = int(self.config.get("max_bytes", 2_000_000))
        for source in self.sources:
            with self._lock:
                previous = dict(self.index["sources"].get(source["id"], {}))
            snapshot = dict(previous)
            snapshot.update({"id": source["id"], "title": source["title"], "url": source["url"], "last_checked": checked})
            try:
                allowed, robots_error = self._robots_allowed(source["url"])
                if not allowed:
                    raise PermissionError(robots_error or "robots.txt disallows this source")
                response = self._call_fetcher(source["url"], max_bytes, self.allowed_urls)
                final_url = response.get("final_url", source["url"])
                if final_url not in self.allowed_urls:
                    raise PermissionError("redirect target is outside the approved source allowlist")
                text = self._extract(response, source)
                digest = _hash(text)
                changed = digest != previous.get("content_hash")
                if changed:
                    snapshot.update({"content_hash": digest, "content": text, "chunks": self._chunk(source["id"], text), "last_changed": checked})
                snapshot.update({"status": "ready", "error": None, "fetched_at": checked if changed else previous.get("fetched_at", checked)})
            except Exception as exc:
                snapshot.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"[:300]})
                snapshot.setdefault("chunks", previous.get("chunks", []))
                snapshot.setdefault("last_changed", previous.get("last_changed"))
                snapshot.setdefault("fetched_at", previous.get("fetched_at"))
            with self._lock:
                self.index["sources"][source["id"]] = snapshot
        self._ensure_embeddings()
        with self._lock:
            self.index["updated_at"] = checked
            self.index["embedding_model"] = self.model
        self._save()
        return self.status()

    def status(self):
        rows = []
        for source in self.sources:
            with self._lock:
                snapshot = dict(self.index["sources"].get(source["id"], {}))
            rows.append({
                "id": source["id"], "title": source["title"], "url": source["url"],
                "last_checked": snapshot.get("last_checked"), "last_changed": snapshot.get("last_changed"),
                "status": snapshot.get("status", "never_refreshed"), "error": snapshot.get("error"),
                "chunks": len(snapshot.get("chunks", [])),
            })
        with self._lock:
            updated_at = self.index.get("updated_at")
        return {"sources": rows, "embedding": {"enabled": self.embedding_enabled, "model": self.model or None,
                "available": self.embedding_available, "message": self.embedding_message},
                "totals": {"sources": len(rows), "ready": sum(row["status"] == "ready" for row in rows),
                           "chunks": sum(row["chunks"] for row in rows)}, "updated_at": updated_at}

    def documents(self):
        return [{"id": row["id"], "title": row["title"], "url": row["url"], "status": row["status"],
                 "last_checked": row["last_checked"], "last_changed": row["last_changed"], "chunks": row["chunks"]}
                for row in self.status()["sources"]]

    def document(self, doc_id):
        with self._lock:
            stored = self.index["sources"].get(doc_id)
            snapshot = json.loads(json.dumps(stored)) if stored else None
        if not snapshot:
            return None
        return {"id": snapshot["id"], "title": snapshot["title"], "url": snapshot["url"],
                "status": snapshot.get("status"), "error": snapshot.get("error"),
                "fetched_at": snapshot.get("fetched_at"), "last_changed": snapshot.get("last_changed"),
                "last_checked": snapshot.get("last_checked"),
                "sections": [{"id": chunk["id"], "heading": f"Live extract {number}", "text": chunk["text"]}
                             for number, chunk in enumerate(snapshot.get("chunks", []), 1)]}

    @staticmethod
    def _cosine(left, right):
        if len(left) != len(right) or not left:
            return 0.0
        denominator = math.sqrt(sum(x*x for x in left)) * math.sqrt(sum(x*x for x in right))
        return sum(x*y for x, y in zip(left, right)) / denominator if denominator else 0.0

    def _all_chunks(self):
        with self._lock:
            pairs = []
            for source in self.sources:
                snapshot = self.index["sources"].get(source["id"])
                if not snapshot or snapshot.get("url") != source["url"]:
                    continue
                snapshot_copy = {key: value for key, value in snapshot.items() if key != "chunks"}
                for chunk in snapshot.get("chunks", []):
                    chunk_copy = dict(chunk)
                    if chunk.get("embedding"):
                        chunk_copy["embedding"] = list(chunk["embedding"])
                    pairs.append((snapshot_copy, chunk_copy))
            return pairs

    def _lexical_rank(self, question, pairs):
        query = Counter(_tokens(question))
        df = Counter()
        bags = []
        for _, chunk in pairs:
            bag = Counter(_tokens(chunk["text"])); bags.append(bag); df.update(set(bag))
        ranked, total = [], len(bags)
        for pair, bag in zip(pairs, bags):
            matched = set(query) & set(bag)
            score = sum((math.log((total + 1) / (df[word] + 1)) + 1) * min(bag[word], 3) for word in matched)
            score /= math.sqrt(sum(value * value for value in bag.values())) or 1
            ranked.append((score, len(matched), pair))
        return sorted(ranked, key=lambda row: row[0], reverse=True)

    def _citation(self, snapshot, chunk):
        return {"id": chunk["id"], "doc_id": snapshot["id"], "title": snapshot["title"], "heading": "Live source extract",
                "text": chunk["text"], "source_url": snapshot["url"], "fetched_at": snapshot.get("fetched_at"),
                "last_checked": snapshot.get("last_checked"), "status": snapshot.get("status"),
                "freshness": "stale_last_good" if snapshot.get("status") == "error" else "current_snapshot"}

    def ask(self, question):
        question = str(question).strip()
        pairs = self._all_chunks()
        if not question or not pairs:
            return {"status": "abstained", "answer": "No refreshed live source contains enough evidence to answer that question.",
                    "confidence": 0.0, "reason": "no_live_evidence", "citations": [], "alternatives": [], "mode": "live-lexical"}
        mode, ranked = "live-lexical", None
        if self.embedding_enabled and all(chunk.get("embedding") for _, chunk in pairs):
            try:
                dimension = len(pairs[0][1]["embedding"])
                query_vector = self._validated_vectors(self.embedder([question], self.model), 1, dimension)[0]
                ranked = [(self._cosine(query_vector, chunk["embedding"]), 0, (snapshot, chunk)) for snapshot, chunk in pairs]
                ranked.sort(key=lambda row: row[0], reverse=True)
                mode, self.embedding_available, self.embedding_message = "live-embedding", True, "Ollama embeddings ready."
            except Exception as exc:
                self.embedding_available = False
                self.embedding_message = f"Ollama query embedding unavailable ({type(exc).__name__}); using lexical retrieval."
        if ranked is None:
            ranked = self._lexical_rank(question, pairs)
        threshold = float(self.config.get("embedding_threshold", 0.48)) if mode == "live-embedding" else float(self.config.get("lexical_threshold", 0.32))
        eligible = [row for row in ranked if row[0] >= threshold and (mode == "live-embedding" or row[1] >= 2)]
        if not eligible:
            best_score = ranked[0][0]
            return {"status": "abstained", "answer": "I couldn’t find enough support in the refreshed GLS IT sources to answer that. Try more specific policy terms or check the official source pages.",
                    "confidence": round(max(0.0, min(1.0, best_score)), 3), "reason": "retrieval_below_threshold",
                    "citations": [], "alternatives": [], "mode": mode}
        best_score, _, best_pair = eligible[0]
        citations = [self._citation(*best_pair)]
        return {"status": "answered", "answer": " ".join(citation["text"] for citation in citations),
                "confidence": round(max(0.0, min(1.0, best_score)), 3), "reason": "supported_by_live_source",
                "citations": citations, "alternatives": [], "mode": mode}
