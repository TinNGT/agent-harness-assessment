from __future__ import annotations


class LLMError(Exception):
    """Base class for LLM-related failures."""


class LLMParseError(LLMError):
    """The LLM's raw output could not be parsed into a valid decision."""


class LLMAPIError(LLMError):
    """The LLM API call itself failed (network, rate limit, provider error)."""
