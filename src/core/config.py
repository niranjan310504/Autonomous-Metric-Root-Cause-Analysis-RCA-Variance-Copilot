"""Application configuration loaded from environment variables."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the RCA copilot."""

    model_config = SettingsConfigDict(
        env_prefix="RCA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Autonomous Metric RCA Copilot"
    environment: str = "development"
    openai_api_key: str | None = Field(default=None, repr=False)
    duckdb_path: str = ":memory:"
    max_query_rows: int = Field(default=500, ge=1, le=500)
    query_timeout_seconds: int = Field(default=30, ge=1, le=300)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings."""

    return Settings()
