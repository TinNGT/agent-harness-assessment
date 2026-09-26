from __future__ import annotations

import httpx
import pytest

from conftest import make_settings
from harness.api.app import create_app


@pytest.fixture
def client(tmp_path):
    app = create_app(make_settings(tmp_path))
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


@pytest.mark.asyncio
async def test_health(client):
    async with client as c:
        resp = await c.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_list_tools(client):
    async with client as c:
        resp = await c.get("/tools")
    assert resp.status_code == 200
    names = {t["name"] for t in resp.json()}
    assert names == {"search_knowledge_base", "get_service_status", "create_incident"}


@pytest.mark.asyncio
async def test_happy_path_via_api(client):
    async with client as c:
        resp = await c.post(
            "/runs", json={"objective": "Check payment-api", "llm": "scripted", "scenario": "happy_path"}
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["status"] == "completed"
        assert body["final_answer"]

        trace_resp = await c.get(f"/runs/{body['id']}/trace")
        assert trace_resp.status_code == 200
        assert len(trace_resp.json()["steps"]) > 0


@pytest.mark.asyncio
async def test_approval_flow_via_api(client):
    async with client as c:
        resp = await c.post(
            "/runs",
            json={
                "objective": "Handle auth-service outage",
                "llm": "scripted",
                "scenario": "approval_create_incident",
            },
        )
        body = resp.json()
        assert body["status"] == "waiting_approval"
        approval = body["pending_approval"]
        assert approval["tool_name"] == "create_incident"

        decide_resp = await c.post(
            f"/runs/{body['id']}/approvals/{approval['id']}",
            json={"decision": "approve", "approver": "alice"},
        )
        assert decide_resp.status_code == 200
        assert decide_resp.json()["status"] == "completed"

        # Deciding again must conflict.
        conflict_resp = await c.post(
            f"/runs/{body['id']}/approvals/{approval['id']}",
            json={"decision": "approve", "approver": "alice"},
        )
        assert conflict_resp.status_code == 409


@pytest.mark.asyncio
async def test_get_run_404_for_unknown_id(client):
    async with client as c:
        resp = await c.get("/runs/does-not-exist")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_approving_unknown_run_404(client):
    async with client as c:
        resp = await c.post(
            "/runs/does-not-exist/approvals/also-missing",
            json={"decision": "approve", "approver": "alice"},
        )
    assert resp.status_code == 404
