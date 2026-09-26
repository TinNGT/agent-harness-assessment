from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration, sourced from environment variables / .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_provider: str = "scripted"  # scripted | litellm
    llm_model: str = "gpt-4o-mini"
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    max_steps: int = 10
    max_run_seconds: int = 60
    tool_timeout_seconds: float = 5.0
    tool_max_retries: int = 2
    llm_max_retries: int = 3
    llm_max_repair: int = 2

    database_url: str = "sqlite:///./harness.db"

    log_level: str = "INFO"
    otel_exporter_otlp_endpoint: str = ""

    data_dir: str = "./data"

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)


@lru_cache
def get_settings() -> Settings:
    return Settings()
