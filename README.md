# GLS Knowledge Desk

GLS Knowledge Desk is an independent portfolio project for document-grounded questions. It combines a deterministic synthetic-policy demo with a separate live collection of selected public GLS University pages. Answers quote their evidence, expose source metadata, flag known demo-policy conflicts, and abstain when retrieval is weak.

> **Unofficial portfolio demo:** This project is not affiliated with or endorsed by GLS University Institute of Technology. Demo policies are entirely synthetic. Live answers use cached snapshots of selected public pages; an excerpt does not guarantee that information is current or applies to a particular student.

## Clone and run

Prerequisites:

- Git
- Python 3.9 or newer
- A modern browser

Clone the repository:

```bash
git clone https://github.com/Siddh-sys-rgb/gls-knowledge-desk.git
cd gls-knowledge-desk
```

A virtual environment is recommended even though the default application uses only the Python standard library.

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can skip activation and run `.\.venv\Scripts\python.exe app.py` directly. Windows Command Prompt users can activate with `.venv\Scripts\activate.bat`.

No `pip install` is required for the default demo or live HTML sources. Start the app:

```bash
python3 app.py
```

On Windows, `python app.py` also works after activating the environment. Open [http://127.0.0.1:8101](http://127.0.0.1:8101). The server binds to the local machine by default.

Check its health in a browser at [http://127.0.0.1:8101/api/health](http://127.0.0.1:8101/api/health), or run:

```bash
curl -s http://127.0.0.1:8101/api/health
```

Use another port with `python3 app.py --port 9000`.

## What to try

The interface keeps two retrieval collections separate.

**Demo policies** use seven synthetic, versioned documents. Useful questions include:

- “When is the deadline to drop a full-term course?” — shows a dated conflict.
- “How do I reset a forgotten campus password?” — shows paraphrase retrieval.
- “What is the dining hall menu?” — demonstrates abstention.
- “Can my friend stay in my dorm for three nights?” — shows a housing conflict.

**Live sources** use only refreshed snapshots from `data/live_sources.json`. Open **Live sources**, select **Refresh sources now**, and wait for each source to show its status. Try:

- “What programmes does the Faculty of Computer Applications and Information Technology offer?”
- “What are the admission announcements for GLS University?”
- “Which BCA MCA syllabus documents are listed?”

The initial configuration includes the public [FCAIT page](https://www.glsuniversity.ac.in/fcait.html), [admission announcements](https://www.glsuniversity.ac.in/announcement.html), and [current syllabus index](https://www.glsuniversity.ac.in/syllabus-for-the-current-academic-year.html). The index page is ingested as a page; linked PDFs are not crawled automatically.

## Live refresh and local cache

Manual refresh is the default. Automatic polling can refresh once at startup and then on an interval:

```bash
python3 app.py --refresh-minutes 15
```

The minimum interval is five minutes. Polling is not a real-time notification service, and stopping the process stops polling.

Downloaded text, chunks, hashes, and optional vectors are saved to `data/live_index.json`, which Git ignores. Content hashes prevent unchanged pages from being rechunked, and cached embeddings are reused. If a fetch fails, the last successful snapshot remains available and is marked stale with the new error.

For a completely fresh live cache, stop the server first, delete `data/live_index.json`, restart, and refresh the sources. On PowerShell:

```powershell
Remove-Item .\data\live_index.json -ErrorAction SilentlyContinue
```

On macOS or Linux:

```bash
rm -f data/live_index.json
```

Restart the app after changing Python code, `data/live_sources.json`, the port, or an embedding environment variable.

## Optional local models

### Semantic retrieval with Ollama embeddings

Keyword retrieval is the truthful default. To use semantic retrieval, install [Ollama](https://ollama.com/), start it locally, and pull one supported embedding model:

```bash
ollama pull nomic-embed-text
GLS_EMBED_MODEL=nomic-embed-text python3 app.py --refresh-minutes 15
```

`embeddinggemma` is also supported. On Windows PowerShell:

```powershell
ollama pull nomic-embed-text
$env:GLS_EMBED_MODEL="nomic-embed-text"
python app.py --refresh-minutes 15
```

Refresh live sources after enabling or changing the model. The engine calls Ollama at `127.0.0.1:11434/api/embed`, validates vector shape and values, and labels answers `live-embedding` only after a real embedding query succeeds. Missing or invalid model output produces an explicit lexical fallback.

Embeddings retrieve relevant excerpts; they do not write prose.

### Optional Ollama answer rewrite

To rewrite a supported extractive answer locally:

```bash
ollama pull llama3.2
```

Enable **Local Ollama rewrite** in the interface. This is separate from embedding retrieval. Rewrites are labeled unverified, must cite only retrieved source IDs, and fall back to the extractive answer on timeout or citation-validation failure. Conflict and abstention responses are not rewritten.

### Optional PDF parsing

HTML ingestion needs no package. Text-based PDF sources require:

```bash
python3 -m pip install -r requirements-live.txt
```

Add each intended HTTPS PDF URL explicitly to `data/live_sources.json`. Sources must use the approved GLS University host. Scanned PDFs need OCR, which is not implemented.

## Tests

Run the repository tests:

```bash
python3 -m unittest discover -s tests -v
```

Run the deterministic 30-question demo-policy evaluation:

```bash
python3 scripts/evaluate.py
```

The evaluation contains 10 answerable, 10 unanswerable, and 10 conflicting cases. Its current 30/30 result is an authored regression baseline, not an independent measure of real-world accuracy. GitHub Actions runs tests and evaluation on Python 3.9 and 3.13.

## Architecture

```text
Browser
  ├── Demo policies ──> deterministic lexical ranking + conflict metadata
  └── Live sources ───> refresh/cache ──> lexical or optional embeddings
            │
            └── exact excerpt + source URL + fetched/check time
                              └── optional unverified Ollama rewrite
```

- `app.py`: local HTTP server, API validation, background refresh scheduling, and optional rewrite adapter.
- `retrieval.py`: synthetic demo retrieval, abstention, conflict handling, and evaluation.
- `live_sources.py`: safe GLS-host fetching, robots checks, extraction, chunking, cache persistence, lexical ranking, and optional Ollama embeddings.
- `data/corpus.json`: synthetic demo policies.
- `data/live_sources.json`: trusted operator configuration for public sources.
- `data/live_index.json`: generated local cache, excluded from Git.
- `web/`: responsive HTML, CSS, and JavaScript interface.
- `tests/`: retrieval, ingestion, cache, safety, embedding, and HTTP behavior tests.

The demo and live indexes never mix. Retrieval scores are heuristics, not calibrated probabilities.

## API

| Endpoint | Purpose |
| --- | --- |
| `GET /api/health` | Local server health |
| `GET /api/documents` | Demo documents |
| `GET /api/documents?collection=live` | Cached live documents |
| `GET /api/documents/:id?collection=live` | One live snapshot and chunks |
| `GET /api/live/status` | Refresh, source, cache, and embedding status |
| `POST /api/live/refresh` | Start one background refresh |
| `POST /api/ask` | Query the selected collection |
| `GET /api/evaluate` | Run the demo regression set |

Example:

```bash
curl -s -X POST http://127.0.0.1:8101/api/ask \
  -H 'Content-Type: application/json' \
  -d '{"collection":"live","question":"What computing programmes are offered?"}'
```

## Troubleshooting

- **Port already in use:** stop the older process with Ctrl+C or choose another port.
- **Live collection is empty:** refresh from the interface and inspect per-source errors.
- **Result says lexical fallback:** unset `GLS_EMBED_MODEL` if lexical mode is intended; otherwise confirm Ollama is running, the model is pulled, then refresh.
- **PDF dependency error:** install `requirements-live.txt` inside the active virtual environment.
- **A page looks stale:** check fetched/check times. Refresh manually; if needed, stop the server and rebuild the cache.
- **Refresh does not notice a newly linked document:** add that exact document URL to the trusted source configuration. The connector does not crawl links.
- **PowerShell uses the wrong Python:** activate `.venv` and run `python --version`.

Stop the app with Ctrl+C.

## Security and honest limitations

The app is a local portfolio server, not a production university service. It has no accounts, authorization, audit log, rate limiting, or student-record controls. It stores no questions, but it caches downloaded public content locally.

Live ingestion enforces HTTPS, the GLS University hostname, configured redirect targets, robots rules, request timeouts, and size limits. It strips common HTML boilerplate and scripts, but extraction can still include irrelevant text or miss JavaScript-rendered content. Failed refreshes preserve stale evidence deliberately. Lexical retrieval can miss paraphrases; semantic retrieval depends on a separately running local model. Neither mode proves that a source is authoritative, current, complete, or applicable. Generated wording can hallucinate despite citation checks.

Production use would require policy-owner review, authenticated connectors, source governance, accessibility work, monitoring, calibrated evaluation, retention rules, and a production web server.

## Contributing with small commits

After cloning, use a short-lived branch and keep each commit focused:

```bash
git switch main
git pull --ff-only
git switch -c docs/improve-setup
# edit and review one coherent change
git add README.md
git commit -m "docs: clarify local setup"
git push -u origin docs/improve-setup
```

Open a pull request, let GitHub Actions run, and merge after review. For code changes, keep implementation and its focused tests in the same commit when practical. Avoid committing `data/live_index.json`, virtual environments, logs, downloaded models, or secrets.

## License

MIT for project code and the synthetic dataset. Downloaded university content remains subject to its original ownership and terms; the project license does not relicense it, and cached live content is excluded from Git.

