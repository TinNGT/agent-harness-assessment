from __future__ import annotations

import pytest

from harness.llm.errors import LLMParseError
from harness.llm.parser import parse_decision


def test_parses_valid_tool_call():
    raw = '{"type": "tool_call", "tool_call": {"id": "c1", "name": "x", "arguments": {"a": 1}}}'
    decision = parse_decision(raw)
    assert decision.type == "tool_call"
    assert decision.tool_call.name == "x"
    assert decision.tool_call.arguments == {"a": 1}


def test_parses_valid_final():
    decision = parse_decision('{"type": "final", "answer": "done"}')
    assert decision.type == "final"
    assert decision.answer == "done"


def test_strips_markdown_code_fence():
    raw = '```json\n{"type": "final", "answer": "done"}\n```'
    decision = parse_decision(raw)
    assert decision.type == "final"
    assert decision.answer == "done"


def test_rejects_non_json_text():
    with pytest.raises(LLMParseError):
        parse_decision("this is not json at all")


def test_rejects_json_array_instead_of_object():
    with pytest.raises(LLMParseError):
        parse_decision("[1, 2, 3]")


def test_rejects_tool_call_missing_tool_call_field():
    with pytest.raises(LLMParseError):
        parse_decision('{"type": "tool_call"}')


def test_rejects_final_with_empty_answer():
    with pytest.raises(LLMParseError):
        parse_decision('{"type": "final", "answer": "   "}')


def test_rejects_unknown_type_value():
    with pytest.raises(LLMParseError):
        parse_decision('{"type": "maybe", "answer": "done"}')
