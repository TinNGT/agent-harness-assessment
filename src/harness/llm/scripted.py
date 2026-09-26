from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from harness.llm.base import LLMClient
from harness.tools.registry import ToolSpec


class ScriptedLLM(LLMClient):
    """Deterministic stand-in for a real LLM, driven by data/scenarios.json.

    Each scenario is a list of "turns". A turn that is a JSON object is
    serialized as the model's raw output. A turn that is a plain string is
    returned verbatim — used to simulate malformed / non-JSON LLM output for
    the repair-loop tests. The turn index is derived from how many assistant
    messages already exist in the conversation, so a run resumed from the DB
    (e.g. after an approval decision) picks up exactly where it left off.
    """

    def __init__(self, scenario: str, scripts: dict[str, list[Any]]) -> None:
        if scenario not in scripts:
            available = ", ".join(sorted(scripts))
            raise ValueError(f"unknown scripted scenario '{scenario}'. Available: {available}")
        self.scenario = scenario
        self.script = scripts[scenario]

    async def step(self, messages: list[dict[str, Any]], tools: list[ToolSpec]) -> str:
        call_index = sum(1 for m in messages if m.get("role") == "assistant")
        if call_index >= len(self.script):
            return json.dumps(
                {
                    "type": "final",
                    "answer": (
                        f"Scenario '{self.scenario}' script exhausted after {call_index} calls "
                        "without reaching a final answer."
                    ),
                }
            )
        turn = self.script[call_index]
        return turn if isinstance(turn, str) else json.dumps(turn)

    @staticmethod
    def load_scripts(path: Path) -> dict[str, list[Any]]:
        return json.loads(path.read_text(encoding="utf-8"))
