#!/usr/bin/env python3
import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from retrieval import CampusRetriever

retriever = CampusRetriever(root / "data/corpus.json")
report = retriever.evaluate(json.loads((root / "data/eval_questions.json").read_text()))
print(json.dumps(report["summary"], indent=2))
failed = [r for r in report["results"] if not r["correct"]]
if failed:
    print("\nMismatches:")
    for row in failed: print(f"{row['id']}: expected {row['expected']}, got {row['actual']} — {row['question']}")
    raise SystemExit(1)
