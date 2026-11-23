"""
Centralized configuration for the Multi-Agent Content Pipeline.

All tunable values live here and are sourced from environment variables
(loaded from a .env file via python-dotenv). Nothing in the rest of the
codebase should read os.environ directly -- import `settings` from this
module instead.
"""
from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


def _get_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


class Settings(BaseModel):
    # --- LLM provider ---
    openai_api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_base_url: str | None = Field(
        default_factory=lambda: os.getenv("OPENAI_BASE_URL") or None
    )
    model_name: str = Field(default_factory=lambda: os.getenv("MODEL_NAME", "gpt-4o-mini"))
    temperature: float = Field(
        default_factory=lambda: float(os.getenv("TEMPERATURE", "0.4"))
    )
    api_timeout: int = Field(default_factory=lambda: int(os.getenv("API_TIMEOUT", "60")))
    # For reasoning-capable models (Groq's gpt-oss-*, Qwen3 family, etc.).
    # Values like "none"/"low" tell the provider to skip/minimize chain-of-
    # thought so the answer lands in the normal `content` field instead of
    # a separate reasoning field. Leave blank for models that don't support it
    # (e.g. plain OpenAI gpt-4o-mini) -- it's omitted from the request entirely.
    reasoning_effort: str | None = Field(
        default_factory=lambda: os.getenv("REASONING_EFFORT") or None
    )

    # --- Pipeline behavior ---
    max_iterations: int = Field(
        default_factory=lambda: int(os.getenv("MAX_ITERATIONS", "3"))
    )
    quality_threshold: float = Field(
        default_factory=lambda: float(os.getenv("QUALITY_THRESHOLD", "80"))
    )
    max_words: int = Field(default_factory=lambda: int(os.getenv("MAX_WORDS", "3000")))

    # --- Research layer ---
    research_provider: str = Field(
        default_factory=lambda: os.getenv("RESEARCH_PROVIDER", "mock")
    )  # "mock" | "web" | "local"
    web_search_api_key: str = Field(
        default_factory=lambda: os.getenv("WEB_SEARCH_API_KEY", "")
    )
    local_documents_dir: str = Field(
        default_factory=lambda: os.getenv("LOCAL_DOCUMENTS_DIR", "./data/documents")
    )
    use_vector_store: bool = Field(default_factory=lambda: _get_bool("USE_VECTOR_STORE", False))

    # --- Logging ---
    log_level: str = Field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    log_dir: str = Field(default_factory=lambda: os.getenv("LOG_DIR", "./logs"))

    # --- Retry behavior ---
    llm_max_retries: int = Field(default_factory=lambda: int(os.getenv("LLM_MAX_RETRIES", "2")))


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (singleton for the process)."""
    return Settings()


settings = get_settings()
