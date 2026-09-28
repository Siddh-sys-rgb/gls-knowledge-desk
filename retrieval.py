"""Transparent, deterministic retrieval and evaluation for synthetic policies."""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "could", "do", "does",
    "for", "from", "had", "has", "have", "how", "i", "if", "in", "is", "it", "its",
    "may", "my", "of", "on", "or", "s", "should", "that", "the", "their", "there",
    "this", "to", "was", "what", "when", "where", "which", "who", "will", "with", "would",
}
SYNONYMS = {
    "drop": ["withdraw", "remove", "deadline"], "withdraw": ["drop", "deadline"],
    "wifi": ["wireless", "network", "internet"], "internet": ["wifi", "network"],
    "password": ["passphrase", "account"], "money": ["refund", "aid", "payment"],
    "refund": ["reimbursement", "credit", "withdraw"], "guest": ["visitor", "overnight"],
    "visitor": ["guest", "overnight"], "room": ["housing", "residence"],
    "cheating": ["academic", "integrity", "misconduct"], "appeal": ["review", "challenge"],
    "transcript": ["record", "registrar"], "late": ["deadline", "after"],
    "scholarship": ["aid", "financial"], "gpa": ["academic", "standing"],
    "device": ["computer", "network"], "forgot": ["reset", "password"],
    "dropping": ["drop", "withdraw", "deadline"], "switch": ["move", "change"],
    "rooms": ["room", "housing", "residence"], "dorm": ["housing", "residence", "room"],
    "friend": ["guest", "visitor"], "stay": ["overnight", "host"], "nights": ["night", "overnight"],
}


def tokens(text):
    base = [word for word in TOKEN_RE.findall(text.lower()) if word not in STOPWORDS]
    expanded = list(base)
    for word in base:
        expanded.extend(SYNONYMS.get(word, []))
    return expanded


class CampusRetriever:
    def __init__(self, corpus_path):
        raw = json.loads(Path(corpus_path).read_text())
        self._docs = {d["id"]: d for d in raw["documents"]}
        self.chunks = []
        for doc in raw["documents"]:
            for section in doc["sections"]:
                self.chunks.append({**section, "doc_id": doc["id"], "title": doc["title"],
                                    "domain": doc["domain"], "published": doc["published"],
                                    "effective": doc["effective"], "status": doc["status"]})
        self.document_count = len(self._docs)
        self.df = Counter()
        self.chunk_tokens = []
        for chunk in self.chunks:
            bag = tokens(chunk["heading"] + " " + chunk["text"] + " " + " ".join(chunk.get("keywords", [])))
            self.chunk_tokens.append(Counter(bag))
            self.df.update(set(bag))

    def documents(self):
        return [{k: d[k] for k in ("id", "title", "domain", "published", "effective", "status", "summary")} for d in self._docs.values()]

    def document(self, doc_id):
        return self._docs.get(doc_id)

    def _rank(self, question):
        query = Counter(tokens(question))
        ranked = []
        n = len(self.chunks)
        for chunk, bag in zip(self.chunks, self.chunk_tokens):
            score = 0.0
            matched = set()
            for term, qtf in query.items():
                if term in bag:
                    idf = math.log((n + 1) / (self.df[term] + 1)) + 1
                    score += min(qtf, 2) * (1 + math.log(bag[term])) * idf
                    matched.add(term)
            norm = math.sqrt(sum(v * v for v in bag.values())) or 1
            score /= norm
            ranked.append((score, len(matched), chunk))
        return sorted(ranked, key=lambda row: (row[0], row[2]["effective"]), reverse=True)

    def ask(self, question):
        ranked = self._rank(question)
        best_score, overlap, best = ranked[0]
        confidence = min(0.99, best_score / 2.4)
        if best_score < 0.62 or overlap < 2:
            return {"status": "abstained", "answer": "I couldn’t find enough support in the synthetic GLS IT policy collection to answer that. Try asking about registration, financial aid, housing, campus technology, or student conduct.",
                    "confidence": round(confidence, 2), "reason": "retrieval_below_threshold", "citations": [], "alternatives": self._suggest(ranked)}

        same_topic = [(s, c) for s, _, c in ranked if c.get("conflict_key") and c.get("conflict_key") == best.get("conflict_key") and s >= best_score * .42]
        variants = {c.get("variant") for _, c in same_topic}
        if len(variants) > 1:
            selected = sorted((c for _, c in same_topic), key=lambda c: c["effective"], reverse=True)[:3]
            citations = [self._citation(c) for c in selected]
            latest = selected[0]
            answer = ("The collection contains conflicting versions. The most recent excerpt says: " + latest["text"] +
                      " Verify the controlling version with the listed campus office before relying on it.")
            return {"status": "conflict", "answer": answer, "confidence": round(confidence, 2),
                    "reason": "dated_versions_disagree", "citations": citations, "alternatives": []}

        selected = [best]
        for score, _, chunk in ranked[1:]:
            if score >= best_score * .62 and chunk["doc_id"] == best["doc_id"] and chunk["id"] != best["id"]:
                selected.append(chunk)
                break
        return {"status": "answered", "answer": " ".join(c["text"] for c in selected),
                "confidence": round(confidence, 2), "reason": "supported_by_retrieved_excerpt",
                "citations": [self._citation(c) for c in selected], "alternatives": []}

    def _citation(self, chunk):
        return {k: chunk[k] for k in ("id", "doc_id", "title", "heading", "text", "published", "effective", "status")}

    def _suggest(self, ranked):
        out = []
        for score, _, chunk in ranked:
            if score > .25 and chunk["heading"] not in out:
                out.append(chunk["heading"])
            if len(out) == 3: break
        return out

    def evaluate(self, cases):
        rows, by_category = [], defaultdict(lambda: {"correct": 0, "total": 0})
        for case in cases:
            result = self.ask(case["question"])
            correct = result["status"] == case["expected_status"]
            if correct and case.get("expected_source"):
                correct = any(c["doc_id"] == case["expected_source"] for c in result["citations"])
            rows.append({"id": case["id"], "category": case["category"], "question": case["question"],
                         "expected": case["expected_status"], "actual": result["status"], "correct": correct})
            stats = by_category[case["category"]]; stats["total"] += 1; stats["correct"] += int(correct)
        for stats in by_category.values(): stats["accuracy"] = round(stats["correct"] / stats["total"], 3)
        return {"summary": {"correct": sum(r["correct"] for r in rows), "total": len(rows),
                            "accuracy": round(sum(r["correct"] for r in rows) / len(rows), 3), "by_category": by_category}, "results": rows}
