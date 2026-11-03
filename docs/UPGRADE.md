# Incremental upgrade checklist

The actual application root is the directory containing `app.py`, `src/`, and
`requirements.txt`. The GitHub repository places this directory under
`PythonProject_ai/`; this local workspace is already that directory and has no
Git metadata. No branch, remote, commit, or push was created. A source-only ZIP
backup is in the ignored `.upgrade-backups/` directory, excluding secrets/data.

Existing architecture: Streamlit orchestrates three modes; an OpenAI-compatible
DeepSeek client serves chat, RAG, paper summaries and data analysis. Normalized
Sentence Transformer vectors use FAISS inner-product search. The agent already
routes, grades, rewrites, searches the web and verifies. Chat JSON persists text
and web memory, while indices and dataframes live in session state.

Implementation order:

- Native vision and bounded local media, compatible chat history.
- Structured PDF blocks, page metadata, tables, captions and selective crops.
- Dense baseline, BM25 + reciprocal rank fusion, optional reranking adapters.
- Extend the existing agent with decomposition, evidence accumulation and limits.
- Offline evaluation corpus and metrics, mocked tests, Streamlit dashboards.
- Dependencies, installation, migration and reproducible benchmark instructions.

Keep existing text chat, web memory, paper summaries, four languages and data
analysis. Heavy model/parser imports remain lazy. Paid integration tests require
an explicit environment switch. See README for validation and limitations.

## Validation performed

- Full offline suite: `python -m unittest discover -v` — 35 tests run;
  34 passed, one paid native-vision integration test skipped by default.
- Real Streamlit AppTest: all four views and all four languages; text chat;
  image submission and image follow-up; paper-figure vision request with page
  and filename evidence. APIs and chat persistence were mocked for these tests.
- PDF parser tests use real generated PDFs and real PyMuPDF crop rendering.
  Docling failure/fallback was tested; Docling itself is not installed here.
- Real local retrieval benchmark: MiniLM + FAISS and MiniLM + BM25 + RRF,
  fixed 10-chunk/8-question synthetic corpus, K=3, zero API calls. JSON/CSV
  outputs are in `reports/local-baseline/`. Neural reranking was not measured
  because no weights were cached. Its scoring contract and graceful fallback
  were tested with mocks; the model adapters are implemented and lazy-loaded.
- Streamlit development server: `http://127.0.0.1:8502`, root and health endpoints
  returned HTTP 200. No live DeepSeek request was made.
- Python syntax and centralized four-language strings were checked. A source
  scan found no API-key-like literals in app/source/tests/docs examples.

## Remaining limits

- Fresh Python 3.12 installation, real Docling/OCR and real Qwen/neural-reranker
  execution were not validated in this preserved Python 3.9 environment.
- Live provider behavior and optional paid answer judging need explicit opt-in.
- Native fallback parsing is heuristic. Textless PDFs require OCR for reliable
  semantic retrieval; explicit page crops still support visual inspection.
- Citations originate in actual extraction metadata, but LLM interpretation and
  verification do not guarantee factual correctness.
- Storage and browser-session workspace scope remain suitable for a trusted
  local user, not an authenticated multi-user service.
