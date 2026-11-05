# Enterprise AI Assistant local validation

Date: 2026-11-05 (Europe/Berlin)

## Verified environment

- Repository initially had no FastAPI application, API route, Docker Compose file, or n8n workflow.
- Local n8n container: `n8n`, image `docker.n8n.io/n8nio/n8n`, version **2.46.6**, published as `127.0.0.1:5678`.
- `GET /healthz` and `GET /healthz/readiness` returned HTTP 200.
- `GET /rest/settings` returned HTTP 200; `/api/v1/workflows` and `/rest/workflows` returned HTTP 401 without credentials.
- No workflow was imported, overwritten, or activated. No public Webhook was enabled.
- FastAPI was not running on port 8000 before this implementation.

## Execution mode

Validation used `ENTERPRISE_ASSISTANT_MODE=mock`. Retrieval was real local inference through the existing `all-MiniLM-L6-v2`, FAISS, BM25 and Reciprocal Rank Fusion implementation over `evaluation/example.json`. Answer generation was deterministic and explicitly labelled `[MOCK GENERATION — RETRIEVED EVIDENCE]`. No DeepSeek or other paid LLM request was made.

## Results

| Scenario | Execution | Result | Evidence |
|---|---|---|---|
| Normal query | Real FastAPI process and real local hybrid retrieval | PASS | HTTP 200, `status=ok`, `mode=mock`, structured sources returned. |
| Empty input | Real FastAPI process | PASS | HTTP 400, `status=invalid_request`, empty sources. |
| No related material | Real FastAPI process | PASS | HTTP 200, `status=no_evidence`, fixed insufficiency answer, empty sources. |
| Service unavailable | Request from the running n8n Docker container after FastAPI stopped | PASS | `host.docker.internal:8000` returned `ECONNREFUSED`, as expected. The workflow's error-format node is configured but was not executed because the workflow was not imported. |
| Request timeout | Real FastAPI process with 50 ms request limit and controlled mock delay | PASS | HTTP 504, `status=timeout`, response in approximately 0.057 seconds. |
| Docker host routing | Node.js `fetch` from the n8n container to Mac FastAPI | PASS | HTTP 200 from `http://host.docker.internal:8000/api/ask` with answer and sources. |
| Workflow import syntax | Ephemeral n8n 2.46.6 container with an isolated temporary database | PASS | `Successfully imported 1 workflow.` The container was removed afterward and the user's n8n database was untouched. |

Targeted offline pytest result: **6 passed**. Tests inject fake backends and make no network or paid API calls. They cover the five required scenarios plus workflow JSON compatibility checks.

Full repository pytest result after implementation: **74 passed, 1 skipped**. The skipped test is the explicitly opted-in paid API integration test. FastAPI was stopped after validation; no demo server was left running.

## n8n workflow compatibility

`n8n/enterprise_ai_assistant.workflow.json` uses node versions observed in the installed n8n 2.46.6 container:

- Webhook `2.2`
- Code `2`
- HTTP Request `4.5`
- Respond to Webhook `1.5`

The workflow is saved with `active=false`. It accepts `{ "question": "..." }`, calls the Mac host through `host.docker.internal`, formats success and error payloads, and returns structured JSON. Import parsing was validated in an ephemeral isolated n8n container. Import into the user's running n8n instance and actual workflow execution remain **not run** because the local workflow APIs require credentials and no authorization was provided to create or activate a workflow.

## Known limitations

- Mock generation returns retrieved excerpts rather than a synthesized answer; it is intended only for integration validation.
- The lightweight no-evidence gate uses lexical overlap before hybrid retrieval and may reject synonym-only questions.
- `asyncio.wait_for` returns a timeout response but cannot forcibly terminate an already-running synchronous model or API thread.
- The FastAPI demo has no authentication, rate limiting, tenant separation, durable job queue, or production observability.
- Real DeepSeek generation was not executed in this phase.
