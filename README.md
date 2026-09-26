# Agent Harness — Operations Assistant

An agent harness that runs an LLM ↔ tool execution loop for an operations
assistant, with input/output validation, retries, timeouts, budget limits,
server-anchored human approval, and a full execution trace.

Everything below runs with **zero API keys** using a deterministic scripted
LLM — this is the default for every run (`--llm scripted` / `"llm": "scripted"`).
A real LLM can be swapped in per-run instead (`--llm real` / `"llm": "real"`);
see [Using a real LLM](#using-a-real-llm).

See [`docs/report.md`](docs/report.md) for the design write-up (architecture,
database schema, state machine, error handling, approval design,
observability, limitations, future work).

## Quickstart

Requires [`uv`](https://docs.astral.sh/uv/) and Python 3.11+ (uv will fetch a
matching interpreter automatically if you don't have one: `uv python install 3.12`).

```bash
cd agent-harness
uv sync --extra dev          # installs everything, including test deps

# Run the full test suite (no API key needed, no network calls)
uv run pytest -q

# Start the API (docs at http://127.0.0.1:8000/docs)
uv run uvicorn harness.api.app:app --reload

# ...or drive it from the CLI instead of the API
uv run harness run "Check payment-api and open an incident if needed" --scenario happy_path
uv run harness scenarios     # list available scripted scenarios
uv run harness tools         # list registered tools + JSON schemas
```

### Docker

```bash
docker compose up --build
# API:            http://localhost:8000/docs
# Jaeger traces:  http://localhost:16686   (OTEL_EXPORTER_OTLP_ENDPOINT is preset in compose)
```

## Using the CLI

```bash
# Happy path — read-only, no approval needed
uv run harness run "Check payment-api" --scenario happy_path

# Approval flow — the CLI prompts y/N when the agent wants to call create_incident
uv run harness run "Handle the auth-service outage" --scenario approval_create_incident

# Inspect a run afterwards
uv run harness show <run_id>
uv run harness trace <run_id>

# Approve/reject an approval directly by id (e.g. from another terminal/session)
uv run harness approve <run_id> <approval_id> --approver alice
uv run harness approve <run_id> <approval_id> --reject --reason "duplicate incident"

# Real LLM instead of the scripted one (needs an API key — see below); no
# --scenario, since a real model decides its own next step
uv run harness run "Check payment-api and open an incident if it's degraded" --llm real
```

## Using the API

Start the server, then either open `/docs` (Swagger UI) or import the
Postman collection:

- `postman/collection.json` + `postman/environment.json` — import both into
  Postman, select the "Agent Harness — Local" environment, and run folders
  top-to-bottom: **0. Health & Tools → 1. Happy path → 2. Approval flow →
  3. Tool failure → 4. Limits & malformed LLM output**. Each request's test
  script saves `run_id`/`approval_id` into the environment for the next
  request, so a whole folder can be run with Postman's Collection Runner
  without manual edits.

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/runs` | `{objective, llm: "scripted"\|"real", scenario?, max_steps?, max_run_seconds?}`. Runs until completion or approval is needed. |
| `GET` | `/runs/{id}` | Status, final answer, pending approval (if any). |
| `GET` | `/runs/{id}/trace` | Full ordered execution trace. |
| `POST` | `/runs/{id}/approvals/{approval_id}` | `{decision: "approve"\|"reject", approver, reason?}`. |
| `GET` | `/runs?status=waiting_approval` | List runs, optionally filtered by status. |
| `GET` | `/tools` | Registered tools + JSON schemas. |
| `GET` | `/health` | Health check. |

Available scripted scenarios (pass as `"scenario"` when `"llm": "scripted"`):
`happy_path`, `approval_create_incident`, `tool_failure_retry_success`,
`tool_failure_timeout`, `tool_failure_schema`, `malformed_then_repair`,
`malformed_exhausted`, `unknown_tool`, `invalid_severity`, `infinite_loop`,
`max_steps_probe`. Each exercises one specific behavior described in the
error-handling table in `docs/report.md`.

## Using a real LLM

The scripted LLM is a fixed script per scenario — useful for tests and demos,
but it doesn't actually "think". To let a real model drive the loop:

```bash
cp .env.example .env
```

Edit `.env` and set **one** of these (only one key is needed):

```
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-...
```
```
LLM_MODEL=claude-3-5-sonnet-20241022
ANTHROPIC_API_KEY=sk-ant-...
```

`LLM_MODEL` accepts any [litellm](https://docs.litellm.ai/docs/providers)-supported
model id — this is the only line that changes to swap providers.

Then run without `--scenario` (a real model picks its own next step):

```bash
uv run harness run "Check payment-api and open an incident if it's degraded" --llm real
```

or via the API, `POST /runs` with `{"objective": "...", "llm": "real"}` (no
`scenario` field).

Notes:
- This calls a real, billed API — usually a few cents per run.
- `.env` is git-ignored; your key is never committed.
- A real model won't always return valid JSON on the first try — that's the
  `LLM_MAX_REPAIR` repair loop kicking in for real; worth watching via
  `harness trace <run_id>` afterward.

## Environment variables

All optional — see [`.env.example`](.env.example) for the full list and
defaults. The defaults require no API key at all.

```
LLM_PROVIDER=scripted        # informational default; actual provider is chosen per run via
                              # --llm/"llm" (CLI/API), not read from this variable
LLM_MODEL=gpt-4o-mini        # any litellm-supported model id, e.g. claude-3-5-sonnet-20241022
OPENAI_API_KEY=              # only needed when calling with --llm real / "llm":"real" on an OpenAI model
ANTHROPIC_API_KEY=           # only needed when calling with --llm real / "llm":"real" on a Claude model
MAX_STEPS=10
MAX_RUN_SECONDS=60
TOOL_TIMEOUT_SECONDS=5
TOOL_MAX_RETRIES=2
LLM_MAX_RETRIES=3
LLM_MAX_REPAIR=2
DATABASE_URL=sqlite:///./harness.db
LOG_LEVEL=INFO
OTEL_EXPORTER_OTLP_ENDPOINT=  # empty disables tracing export; point at an OTLP/HTTP collector to enable
DATA_DIR=./data
```

## Mock tools & dataset

- `get_service_status(service_name)` — reads `data/services.json` (9 services;
  three have injected, deterministic faults: `slow-service` always times out,
  `flaky-service` fails twice then succeeds, `broken-service` returns a
  schema-invalid payload).
- `search_knowledge_base(query)` — keyword search over `data/kb.json` (13
  operational runbooks).
- `create_incident(title, description, severity)` — requires approval; writes
  to the `incidents` table with an idempotency key so a replayed/duplicate
  approval never creates two incidents.

The same `data/` directory is the mock dataset referenced in the submission
(also uploaded separately per the assignment's sharing requirement).

## Project layout

```
agent-harness/
├── src/harness/
│   ├── api/            # FastAPI app, routes, request/response schemas
│   ├── cli.py           # Typer CLI — calls the same service layer as the API
│   ├── core/            # runner.py (agent loop), state.py, budget.py, approval.py, backoff.py, service.py
│   ├── llm/             # LLM interface + ScriptedLLM + LiteLLMClient + parser
│   ├── tools/            # tool registry, executor (timeout/retry), mock tool implementations
│   ├── storage/          # SQLModel models + repository (DB access)
│   └── observability/    # structlog JSON logging, OpenTelemetry tracing
├── data/                 # mock dataset: services.json, kb.json, scenarios.json
├── tests/                # pytest suite (deterministic, no network calls)
├── postman/              # collection.json + environment.json
├── docs/report.md        # design write-up
├── Dockerfile / docker-compose.yml
└── pyproject.toml
```

## Testing

```bash
uv run pytest -q                       # full suite
uv run pytest -q --cov=harness         # with coverage
```

The suite (58 tests, ~95% coverage) covers: happy-path completion, tool
retry/timeout/schema failures, the full approval lifecycle (pause →
approve/reject → resume, double-decide 409, stale-run-version conflict,
unknown-id 404), step/deadline/loop limits (using a fake/ticking clock — no
real `sleep`), malformed-LLM-output repair and repair-budget exhaustion,
LLM-API-error retry and retry-exhaustion, parser edge cases, the state
machine's illegal-transition guard, input validation, idempotent incident
creation, the real-LLM client (with `litellm` stubbed out — no network), and
end-to-end API tests via `httpx.AsyncClient`.
