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
    max_context_chars: int = Field(
        default=6000,
        description="Maximum characters in formatted conversation history to prevent context overflow"
    )
    min_recent_messages: int = Field(
        default=5,
        description="Minimum number of recent messages to always include in full"
    )

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
    twilio_validate_signature: bool = Field(
        default=True,
        description="Validate Twilio webhook signatures for security"
    )

    # ============================================
    # RATE LIMITING & ABUSE DETECTION
    # ============================================
    rate_limit_max_requests: int = Field(
        default=10,
        description="Maximum messages per lead per time window"
    )
    rate_limit_window_seconds: int = Field(
        default=3600,
        description="Rate limit time window in seconds (default: 1 hour)"
    )
    rate_limit_spike_threshold: int = Field(
        default=5,
        description="Number of messages that trigger spike detection"
    )
    rate_limit_spike_window_seconds: int = Field(
        default=60,
        description="Time window for spike detection in seconds (default: 1 minute)"
    )
    rate_limit_ban_duration_seconds: int = Field(
        default=3600,
        description="Default ban duration in seconds (default: 1 hour)"
    )
    rate_limit_auto_ban_on_spike: bool = Field(
        default=True,
        description="Automatically ban leads that trigger spike detection"
    )

    # ============================================
    # MESSAGE BUFFERING
    # ============================================
    message_buffer_seconds: float = Field(
        default=10.0,
        description="Seconds to wait for additional messages before processing"
    )
    message_buffer_max_messages: int = Field(
        default=20,
        description="Maximum messages to buffer per lead before force-flush"
    )
    message_buffer_separator: str = Field(
        default="\n",
        description="Separator used when concatenating buffered messages"
    )

    # ============================================
    # CIRCUIT BREAKER (LLM Degradation)
    # ============================================
    circuit_breaker_failure_threshold: int = Field(
        default=5,
        description="Consecutive failures before circuit opens"
    )
    circuit_breaker_recovery_timeout: float = Field(
        default=60.0,
        description="Seconds before attempting recovery probe"
    )
    circuit_breaker_half_open_max_calls: int = Field(
        default=1,
        description="Max probe calls in half-open state"
    )

    # ============================================
    # FOLLOW-UP PROMPTS (Lead Re-engagement)
    # ============================================
    followup_initial_delay_hours: int = Field(
        default=24,
        description="Hours after last interaction before first follow-up"
    )
    followup_max_attempts: int = Field(
        default=3,
        description="Maximum follow-up attempts before marking lead cold"
    )
    followup_escalation_hours: list[int] = Field(
        default=[24, 48, 72],
        description="Hours between follow-up attempts (escalating)"
    )

    # ============================================
    # HUMAN HANDOFF
    # ============================================
    slack_handoff_webhook_url: Optional[str] = Field(
        default=None,
        validation_alias="SLACK_HANDOFF_WEBHOOK_URL",
        description="Slack webhook URL for handoff notifications"
    )
    handoff_message_template: str = Field(
        default="A lead requires human attention",
        description="Default message when AI triggers handoff"
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
