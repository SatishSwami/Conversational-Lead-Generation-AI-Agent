"""
Centralized application configuration.
"""

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "AutoStream AI Agent"

    app_version: str = "1.0.0"

    environment: str = Field(
        default="development",
        alias="ENV",
    )

    port: int = Field(
        default=8000,
        alias="PORT",
    )

    llm_provider: str = Field(
        default="anthropic",
        alias="LLM_PROVIDER",
    )

    llm_model: Optional[str] = Field(
        default=None,
        alias="LLM_MODEL",
    )

    database_url: str = Field(
        default="sqlite:///./data/autostream.db",
        alias="DATABASE_URL",
    )

    whatsapp_verify_token: Optional[str] = Field(
        default=None,
        alias="WHATSAPP_VERIFY_TOKEN",
    )

    whatsapp_app_secret: Optional[str] = Field(
        default=None,
        alias="WHATSAPP_APP_SECRET",
    )

    whatsapp_access_token: Optional[str] = Field(
        default=None,
        alias="WHATSAPP_ACCESS_TOKEN",
    )

    whatsapp_phone_number_id: Optional[str] = Field(
        default=None,
        alias="WHATSAPP_PHONE_NUMBER_ID",
    )

    api_key: Optional[str] = Field(
        default=None,
        alias="API_KEY",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()