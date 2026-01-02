"""
Centralized Configuration System
Environment-aware settings for all agents and services.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from typing import Literal


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow"
    )
    """
    Production-grade configuration management.
    Loads from environment variables with sensible defaults.
    """

    # ============================================
    # OPENAI CONFIGURATION
    # ============================================
    openai_api_key: str

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

    # ============================================
    # ENVIRONMENT
    # ============================================
    environment: Literal["development", "staging", "production"] = "development"

    # ============================================
    # MONGODB CONFIGURATION
    # ============================================
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_database: str = "gp_data_v4"
    mongodb_max_pool_size: int = 10
    mongodb_min_pool_size: int = 1
    mongodb_server_selection_timeout_ms: int = 5000  # 5 seconds

    # ============================================
    # DATA RETENTION & ARCHIVAL
    # ============================================
    message_retention_days: int = 365
    enable_message_archival: bool = True


@lru_cache()
def get_settings() -> Settings:
    """
    Singleton pattern for settings.
    Uses LRU cache to ensure only one Settings instance exists.
    """
    return Settings()


# Convenience accessor for common use
settings = get_settings()
