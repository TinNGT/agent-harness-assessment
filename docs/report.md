# Agent Harness for an Operations Assistant — Design Report

## 1. Problem framing and where this design sits

The assessment asks for a harness that manages **agent execution, tool
calling, state, failures, and safety controls** for an operations assistant
with three tools: `search_knowledge_base`, `get_service_status`, and the
sensitive `create_incident`.

Using the orchestration-pattern vocabulary from recent agent-architecture
literature, this implementation is a:

- **Single-agent inner loop** — one LLM repeatedly chooses between calling a
  tool or answering. No planner/sub-agent hierarchy; the assessment's tools
  and scope don't warrant one, and a single, well-instrumented loop is easier
  to reason about and grade.
- **Hybrid execution model** — the loop runs **in-process, in memory** for
  speed (no queue/worker hop per step), but **checkpoints to SQLite after
  every step** (`runs.messages`, `steps` rows). This gives the "stateful,
  resumable" property of a durable-execution system without its
  infrastructure cost: if the process restarts while a run is
  `WAITING_APPROVAL`, the run resumes correctly from the DB once a decision
  comes in, because the entire conversation state lives in `runs.messages`,
  not in a Python stack frame.

## 2. Architecture and tech stack

```
        Postman / CLI
             │
      ┌──────▼──────┐
      │  FastAPI    │  POST /runs, GET /runs/{id}, /trace, /approvals
      └──────┬──────┘
             │
      ┌──────▼───────────────────────────────┐
      │           AgentRunner (core loop)     │
      │  ┌─────────┐  ┌──────────┐  ┌──────┐ │
      │  │ LLM     │  │ Tool     │  │Budget│ │
      │  │ Client  │  │ Executor │  │Guard │ │
      │  └────┬────┘  └────┬─────┘  └──────┘ │
      │       │       ┌────▼─────┐           │
      │       │       │ Approval │           │
      │       │       │  Gate    │           │
      │       │       └──────────┘           │
      └───────┼──────────────┬───────────────┘
              │              │
   ┌──────────▼───┐   ┌──────▼──────┐   ┌───────────────┐
   │ Scripted LLM │   │ Mock Tools  │   │ SQLite (state │
   │ / LiteLLM    │   │ + dataset   │   │ + history)    │
   └──────────────┘   └─────────────┘   └───────────────┘
              ↓ structlog JSON logs + OpenTelemetry spans
```

| Layer | Choice | Why |
|---|---|---|
| Language / project mgmt | Python 3.12 + **uv** | Fast, reproducible installs; `uv run` needs no manual venv activation |
| API | **FastAPI** | Free OpenAPI/Swagger docs, async-native |
| Validation | **Pydantic v2** | One schema definition serves tool-input validation, tool-output validation, LLM-decision parsing, and `/tools` JSON-schema output |
| DB | **SQLite + SQLModel** | Zero setup for a reviewer; SQLModel gives typed models over SQLAlchemy Core |
| Real LLM | **LiteLLM** | Swapping OpenAI/Claude/Gemini is one env var (`LLM_MODEL`), not a new client class |
| Retry | Hand-rolled attempt loop (`core/backoff.py`: exponential + full jitter) | Simple enough not to need `tenacity`; keeps attempt-numbering visible to fault-injected mocks, and the same `compute_delay()` backs both tool retries and LLM-API retries |
| Logging | **structlog** → JSON lines | Every event carries `run_id`/`tool`/`attempt` for grep-able logs |
| Tracing | **OpenTelemetry**, optional OTLP export | `run` → `llm_call` / `tool_call.<name>` spans; no-op by default, so it costs nothing when `OTEL_EXPORTER_OTLP_ENDPOINT` is unset |
| CLI | **Typer** | Thin wrapper calling the exact same `HarnessService` the API uses |
| Tests | **pytest + pytest-asyncio**, `httpx.AsyncClient` for API tests | |

### Why not an agent framework (LangGraph, OpenAI Agents SDK, Pydantic AI) for the core loop

The assessment explicitly grades *agent-loop and state-management design,
error handling, and safety controls*. Delegating the loop, retries, or the
approval gate to a framework would hide exactly the design decisions being
evaluated. The implementation does borrow one idea from **Pydantic AI**'s
public API — a per-tool `requires_approval` flag that pauses execution and
resumes via a stored "deferred tool call" — because it is a clean, minimal
shape for the same problem, not because the library itself is used.

## 3. State machine

```
PENDING → RUNNING ─┬─→ COMPLETED
                   ├─→ WAITING_APPROVAL ──(approve/reject)──→ RUNNING
                   ├─→ FAILED            (unrecoverable error)
                   └─→ LIMIT_EXCEEDED    (steps, deadline, or loop)
```

All status changes go through one function, `core/state.transition()`, which
looks up an explicit adjacency table and raises `InvalidTransitionError` on
anything not listed there — e.g. it is structurally impossible to "approve" a
run that is already `COMPLETED`, because `COMPLETED` has no outgoing edges.

**Rejecting an approval does not fail the run.** The harness appends an
observation ("User rejected create_incident: `<reason>`") to the
conversation and returns the run to `RUNNING`; the LLM decides what to do
next (typically: give a final answer acknowledging the human's decision).

## 4. Database design

```sql
runs
  id, objective, status, llm_provider, scenario
  step_count, max_steps, started_at, deadline_at   -- deadline_at = started_at + max_run_seconds
  consecutive_llm_errors                            -- independent from step_count, bounds repairs
  messages        JSON   -- full conversation, replayable after a restart
  final_answer, error JSON
  version         INT    -- optimistic lock
  created_at, updated_at

steps                     -- append-only execution trace
  id, run_id, idx
  type            -- llm_call | tool_call | tool_result | approval_requested
                  -- | approval_decided | error | final
  payload JSON, attempt INT, latency_ms, created_at

approvals
  id, run_id, tool_call_id, tool_name
  args JSON, args_hash    -- sha256 of canonical JSON, for audit/integrity
  status, decided_by, reason, decided_at

incidents                 -- mock create_incident's backing table
  id (INC-1001, ...), idempotency_key UNIQUE
  title, description, severity, created_at
```

Three points worth calling out:

- **`messages` lives on `runs`, not in a process-local variable.** After an
  approval decision, `HarnessService.decide_approval` re-fetches the `Run`
  row and continues the loop from exactly that state — including across a
  server restart. This is the "hybrid execution" property from section 1.
  One implementation subtlety: SQLAlchemy's change tracking does not notice
  in-place mutation of a plain Python list stored in a JSON column, so
  `messages` is typed with SQLAlchemy's `MutableList.as_mutable(JSON)` —
  without it, `messages.append(...)` would silently fail to persist.
- **`version` is an optimistic lock**, checked with a single atomic
  `UPDATE approvals SET status=... WHERE id=? AND status='pending'` followed
  by `UPDATE runs SET version=version+1 WHERE id=? AND version=?`. Two
  concurrent "approve" requests race on the same guarded `UPDATE`; exactly
  one succeeds, and the loser's `rowcount == 0` becomes a `409 Conflict` —
  no read-modify-write race window.
- **`args_hash`** records a SHA-256 of the approved tool arguments for audit.
  The approve/reject API call itself never carries arguments — see §6.

## 5. Agent loop and error handling

Simplified loop (see `core/runner.py::AgentRunner._drive` for the real thing):

```python
while True:
    budget.check(run)                      # steps, deadline, identical-call loop → LIMIT_EXCEEDED
    decision = await llm_turn(run)         # call LLM, parse; malformed → repair message, retry
    if decision is None: continue          # still within repair budget
    if decision.kind == "final": return complete(run, decision.answer)

    tool = registry.get(decision.tool_call.name)     # unknown → observation, counts as a "mistake"
    args = tool.validate_input(decision.tool_call.arguments)  # invalid → observation, counts too

    if tool.requires_approval:
        create_approval(run, tool, args)
        return pause(run)                  # WAITING_APPROVAL, checkpointed, function returns

    result = await executor.execute(tool, args)      # timeout + bounded retry + output validation
    append_observation(run, result)
    checkpoint(run)
```

Two counters, independent of each other:

- `step_count` (bounded by `MAX_STEPS`) increments only on **meaningful agent
  actions**: an executed tool call, an approval request, or a final answer.
- `consecutive_llm_errors` (bounded by `LLM_MAX_REPAIR`) increments on
  **LLM mistakes**: malformed JSON, an unknown tool name, or arguments that
  fail the tool's schema. It resets to zero the moment the LLM makes a valid
  tool call. This separation means a confused LLM that eventually
  self-corrects doesn't burn through the step budget meant for genuine
  agent work, while a *persistently* confused LLM still fails fast instead
  of spinning forever.

| Situation | Handling | Limit |
|---|---|---|
| Tool timeout | `asyncio.wait_for` cancels it; retried with the next attempt | `TOOL_MAX_RETRIES=2` (3 attempts total) |
| Tool transient error (`TransientToolError`) | Retried | same |
| Tool permanent error (bad input, not found, schema-invalid output) | **No retry**; returned to the LLM as an observation | — |
| Retries exhausted | Observation `{"error": {"retryable": false/true, ...}}`; LLM decides what to do next | — |
| Tool output fails its schema | Treated as a permanent tool error, logged at `error` level | — |
| Malformed / incomplete LLM JSON | Repair message appended, LLM asked to retry | `LLM_MAX_REPAIR=2` consecutive, then `FAILED` |
| Unknown tool name | Observation lists the valid tool names | counts toward the repair budget above |
| Invalid tool arguments | Observation with the validation error | counts toward the repair budget above |
| LLM API error/timeout (real LLM) | Retried with backoff at the call site | `LLM_MAX_RETRIES=3`, then `FAILED` |
| Step budget exceeded | `LIMIT_EXCEEDED` | `MAX_STEPS=10` |
| Deadline exceeded | `LIMIT_EXCEEDED` | `MAX_RUN_SECONDS=60` |
| Same tool + same arguments 3× in a row | `LIMIT_EXCEEDED` (loop detector) | 3 identical calls |

**Idempotency for `create_incident`.** This tool is not safe to blindly
retry: if a request times out mid-flight, the incident may already have been
created, and a naive retry would create a duplicate. The fix is an
**idempotency key** — the harness uses the `approval_id` itself, since it is
unique per approved intent — passed into the mock tool, which looks the key
up in the `incidents` table before inserting and replays the existing row on
a repeat. Read tools (`search_knowledge_base`, `get_service_status`) have no
such restriction and retry freely; each `ToolSpec` declares an `idempotent`
flag that the executor uses to decide whether retries are attempted at all.

## 6. Approval design (server-anchored)

The client's decision request is intentionally minimal:
`{"decision": "approve"|"reject", "approver": "...", "reason": "..."}` — it
never includes the tool name or arguments. Those were captured server-side
the moment the LLM proposed the call and are the only ones ever executed;
a client cannot smuggle in different arguments at decision time. `args_hash`
records what was approved for audit purposes. Concurrency safety comes from
the atomic guarded `UPDATE` described in §4, not from application-level
locking.

## 7. Observability

- **Structured logs**: every tool retry, timeout, and permanent failure is
  one JSON line via `structlog`, e.g.
  `{"event": "tool.retry", "run_id": "...", "tool": "get_service_status", "attempt": 2, ...}`.
- **Traces**: `run` → `llm_call` / `tool_call.<name>` spans via
  OpenTelemetry. With `OTEL_EXPORTER_OTLP_ENDPOINT` unset (the default), the
  OpenTelemetry API falls back to a no-op tracer — zero overhead, zero
  configuration for a reviewer. Setting it (`docker compose up` wires this to
  a local Jaeger) exports real traces.
- **Trace endpoint** (`GET /runs/{id}/trace`) reads the `steps` table
  directly — a full, ordered execution history is available without Jaeger
  running at all.

## 8. Testing strategy

58 tests, ~95% line coverage, all deterministic and network-free (real network
calls — `litellm`/`LiteLLMClient` — are exercised with a stubbed-in module,
never a live provider): happy path; flaky/timeout/schema-invalid tool
failures; the full approval lifecycle (pause → approve/reject → resume,
double-decide → 409, unknown id → 404, a stale-run-version conflict distinct
from an already-decided approval); step/deadline/loop limits;
malformed-LLM repair and repair-exhaustion; LLM-API-error retry and
retry-exhaustion; parser edge cases (markdown-fenced JSON, non-object JSON,
missing fields); the state-machine's illegal-transition guard; input
validation; idempotent incident creation; and end-to-end API tests via
`httpx.AsyncClient`. Two techniques keep this both fast and deterministic:

- **`ScriptedLLM`** plays back a fixed script per scenario (`data/scenarios.json`),
  indexed by how many LLM turns have happened so far — so a run resumed after
  an approval decision picks up exactly where it left off, even reloaded from
  the DB. It never touches the network.
- **A fake/ticking clock** is injected into `BudgetGuard` so deadline tests
  don't need a real `sleep()` — see `tests/test_limits.py::TickingClock`,
  which advances simulated time on every call to `.now()`.

Fault injection (services.json's `fault` field: `timeout` / `flaky` /
`broken`) is keyed off `(service, attempt)`, never randomness, so the same
scenario always produces the same trace — required for the Postman
collection's assertions to be stable.

Run instructions, environment variables, and the Postman collection layout
are documented in [`README.md`](../README.md).

## 9. Limitations

- The loop runs synchronously inside the HTTP request; a long-running real
  LLM call blocks that worker for the duration of `POST /runs`.
- SQLite is fine for a single-process demo; it will not hold up under
  concurrent write load.
- Knowledge-base search is keyword overlap, not semantic (no embeddings/BM25).
- Approval has no real authentication/authorization — `approver` is a free-text
  field, not a verified identity.
- The loop calls exactly one tool per turn; there's no parallel tool-calling.

## 10. Future improvements

- Move run execution to a worker + queue (or a durable-execution engine like
  Temporal) so `POST /runs` returns immediately and long runs survive a
  process restart mid-tool-call, not just mid-approval-wait.
- Swap SQLite for Postgres for concurrent-write safety.
- Vector search (embeddings) over the knowledge base instead of keyword match.
- RBAC for who is allowed to approve which tools/severities.
- Parallel tool calls within a single LLM turn.
- Cross-session memory (e.g. `mem0`) so the agent recalls prior incidents for
  the same service instead of starting cold every run.
- A lightweight risk-classifier model in front of the approval policy,
  instead of a static `requires_approval` flag per tool.
- Extend to multi-agent (hierarchical or fan-out) once a single tool-calling
  loop stops being sufficient for the operations surface being covered.
