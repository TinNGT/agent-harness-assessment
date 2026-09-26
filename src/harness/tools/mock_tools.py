from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from harness.tools.errors import PermanentToolError, TransientToolError
from harness.tools.registry import ToolContext, ToolRegistry, ToolSpec

# --------------------------------------------------------------------------- schemas


class ServiceStatusInput(BaseModel):
    service_name: str = Field(min_length=1, description="Name of the service to check")


class ServiceStatusOutput(BaseModel):
    service_name: str
    status: str
    detail: str


class KBSearchInput(BaseModel):
    query: str = Field(min_length=1, description="Free-text search query")


class KBHit(BaseModel):
    id: str
    title: str
    snippet: str


class KBSearchOutput(BaseModel):
    query: str
    results: list[KBHit]


class CreateIncidentInput(BaseModel):
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    severity: Literal["low", "medium", "high", "critical"]


class CreateIncidentOutput(BaseModel):
    incident_id: str
    title: str
    severity: str
    status: str
    created: bool


# ----------------------------------------------------------------------------- data


class ServiceData:
    def __init__(self, records: dict[str, dict[str, Any]]) -> None:
        self._records = records

    @classmethod
    def load(cls, path: Path) -> "ServiceData":
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls({item["name"]: item for item in raw})

    def get(self, name: str) -> Optional[dict[str, Any]]:
        return self._records.get(name)


class KnowledgeBase:
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self._records = records

    @classmethod
    def load(cls, path: Path) -> "KnowledgeBase":
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(raw)

    def search(self, query: str, top_k: int = 5) -> list[KBHit]:
        tokens = [t for t in re.findall(r"[a-z0-9]+", query.lower()) if t]
        scored: list[tuple[int, dict[str, Any]]] = []
        for record in self._records:
            haystack = " ".join(
                [record.get("title", ""), " ".join(record.get("tags", [])), record.get("content", "")]
            ).lower()
            score = sum(haystack.count(tok) for tok in tokens)
            if score > 0:
                scored.append((score, record))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        hits: list[KBHit] = []
        for _, record in scored[:top_k]:
            content = record.get("content", "")
            snippet = content.strip().replace("\n", " ")[:200]
            hits.append(KBHit(id=record["id"], title=record["title"], snippet=snippet))
        return hits


# -------------------------------------------------------------------------- handlers

# Faults are keyed purely off deterministic (service, attempt) pairs, never
# random, so scripted-LLM demos and tests are 100% reproducible.


def make_get_service_status_handler(services: ServiceData):
    async def handler(args: ServiceStatusInput, ctx: ToolContext) -> dict[str, Any]:
        record = services.get(args.service_name)
        if record is None:
            raise PermanentToolError(f"service '{args.service_name}' is not registered")

        fault = record.get("fault")
        if fault == "timeout":
            await asyncio.sleep(999)  # always cancelled by the executor's timeout
        if fault == "flaky" and ctx.attempt < 3:
            raise TransientToolError(
                f"service '{args.service_name}' is temporarily unreachable (attempt {ctx.attempt})"
            )
        if fault == "broken":
            # Deliberately violates ServiceStatusOutput's schema to exercise
            # output-validation handling (missing field + wrong type).
            return {"service_name": args.service_name, "status": 12345}

        return {
            "service_name": args.service_name,
            "status": record["status"],
            "detail": record.get("detail", ""),
        }

    return handler


def make_search_knowledge_base_handler(kb: KnowledgeBase):
    async def handler(args: KBSearchInput, ctx: ToolContext) -> dict[str, Any]:
        hits = kb.search(args.query)
        return {"query": args.query, "results": [hit.model_dump() for hit in hits]}

    return handler


async def create_incident_handler(args: CreateIncidentInput, ctx: ToolContext) -> dict[str, Any]:
    assert ctx.repo is not None, "create_incident requires a repository in its context"
    assert ctx.idempotency_key is not None, "create_incident requires an idempotency key"
    incident, created = ctx.repo.create_incident_if_not_exists(
        idempotency_key=ctx.idempotency_key,
        title=args.title,
        description=args.description,
        severity=args.severity,
    )
    return {
        "incident_id": incident.id,
        "title": incident.title,
        "severity": incident.severity,
        "status": "open",
        "created": created,
    }


def build_registry(data_dir: Path) -> ToolRegistry:
    services = ServiceData.load(data_dir / "services.json")
    kb = KnowledgeBase.load(data_dir / "kb.json")

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="get_service_status",
            description="Retrieve the current status of a service by name.",
            input_model=ServiceStatusInput,
            output_model=ServiceStatusOutput,
            handler=make_get_service_status_handler(services),
            requires_approval=False,
            idempotent=True,
        )
    )
    registry.register(
        ToolSpec(
            name="search_knowledge_base",
            description="Search internal documentation / runbooks.",
            input_model=KBSearchInput,
            output_model=KBSearchOutput,
            handler=make_search_knowledge_base_handler(kb),
            requires_approval=False,
            idempotent=True,
        )
    )
    registry.register(
        ToolSpec(
            name="create_incident",
            description="Create an incident in the external incident-management system.",
            input_model=CreateIncidentInput,
            output_model=CreateIncidentOutput,
            handler=create_incident_handler,
            requires_approval=True,
            idempotent=False,
        )
    )
    return registry
