"""
Centralized Configuration System
Environment-aware settings for all agents and services.
"""
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from typing import Literal, Optional


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    """
    Production-grade configuration management.
    Loads from environment variables with sensible defaults.
    """

    # ============================================
    # OPENAI CONFIGURATION
    # ============================================
    openai_api_key: str = Field(..., validation_alias="OPENAI_API_KEY")

    # ============================================
    # MODEL SELECTION (by agent role)
    # ============================================
    classifier_model: str = "openai:gpt-4o-mini"
    director_model: str = "openai:gpt-4o"
    executor_model: str = "openai:gpt-4o-mini"

    # ============================================
    # BUSINESS RULES
    # ============================================
    enterprise_threshold: int = 30  # Seats needed for enterprise classification
    history_window_size: int = 20   # Max messages in working memory

    # ============================================
    # COST CONTROLS & SAFETY
    # ============================================
    daily_cost_limit_usd: float = 100.0
    hourly_cost_limit_usd: float = 20.0
    max_retries: int = 3
    retry_min_wait_seconds: int = 2
    retry_max_wait_seconds: int = 10

    # ============================================
    # COMPLIANCE & SECURITY
    # ============================================
    enable_pii_filtering: bool = True
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    enable_structured_logging: bool = False  # Set to True for production JSON logs

    # Security Validation
    enable_security_validation: bool = True
    max_message_length: int = 5000  # Maximum message length in characters
    block_prompt_injection: bool = True
    block_context_flooding: bool = True
    block_profanity: bool = False  # Set to True to block profane messages
    block_injection_attempts: bool = True  # SQL, XSS, command injection

    # ============================================
    # ENVIRONMENT
    # ============================================
    environment: Literal["development", "staging", "production", "test"] = "development"

    # ============================================
    # MONGODB CONFIGURATION
    # ============================================
    mongodb_uri: str = Field(
        default="mongodb://localhost:27017",
        validation_alias="MONGODB_URI"
    )
    mongodb_database: str = Field(
        default="gp-data-v4",
        validation_alias="MONGODB_DATABASE"
    )
    mongodb_max_pool_size: int = 10
    mongodb_min_pool_size: int = 1
    mongodb_server_selection_timeout_ms: int = 5000  # 5 seconds

    # ============================================
    # DATA RETENTION & ARCHIVAL
    # ============================================
    message_retention_days: int = 365
    enable_message_archival: bool = True

    # ============================================
    # TWILIO CONFIGURATION
    # ============================================
    twilio_account_sid: Optional[str] = Field(
        default=None,
        validation_alias="TWILIO_ACCOUNT_SID"
    )
    twilio_auth_token: Optional[str] = Field(
        default=None,
        validation_alias="TWILIO_AUTH_TOKEN"
    )
    twilio_whatsapp_from: str = Field(
        default="whatsapp:+16205828564",
        validation_alias="TWILIO_WHATSAPP_FROM",
        description="Your Twilio WhatsApp number (format: whatsapp:+16205828564)"
    )


@lru_cache()
def get_settings() -> Settings:
    """
    Singleton pattern for settings.
    Uses LRU cache to ensure only one Settings instance exists.
    """
    return Settings()


# Convenience accessor for common use
settings = get_settings()
