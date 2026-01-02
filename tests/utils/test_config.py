"""
Tests for centralized configuration system.
Verifies environment variable loading and default values.
"""
import pytest
import os
from src.config import Settings, get_settings


class TestConfigurationSystem:
    """Test suite for configuration management."""

    def test_default_values(self):
        """Verify all configuration fields have sensible defaults."""
        # Clear the lru_cache to get fresh settings
        get_settings.cache_clear()

        settings = Settings(OPENAI_API_KEY="test-key")

        # Model defaults
        assert settings.classifier_model == "openai:gpt-4o-mini"
        assert settings.director_model == "openai:gpt-4o"
        assert settings.executor_model == "openai:gpt-4o-mini"

        # Business rules
        assert settings.enterprise_threshold == 30
        assert settings.history_window_size == 20

        # Cost controls
        assert settings.daily_cost_limit_usd == 100.0
        assert settings.hourly_cost_limit_usd == 20.0
        assert settings.max_retries == 3
        assert settings.retry_min_wait_seconds == 2
        assert settings.retry_max_wait_seconds == 10

        # Compliance
        assert settings.enable_pii_filtering is True
        assert settings.log_level == "INFO"
        assert settings.enable_structured_logging is False

        # Environment (when not explicitly set in conftest.py for tests)
        assert settings.environment in ["development", "test"]

    def test_environment_variable_override(self, monkeypatch):
        """Verify environment variables override defaults."""
        get_settings.cache_clear()

        # Set environment variables
        monkeypatch.setenv("OPENAI_API_KEY", "custom-key")
        monkeypatch.setenv("CLASSIFIER_MODEL", "openai:gpt-4o")
        monkeypatch.setenv("ENTERPRISE_THRESHOLD", "50")
        monkeypatch.setenv("DAILY_COST_LIMIT_USD", "200.0")
        monkeypatch.setenv("MAX_RETRIES", "5")
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("ENVIRONMENT", "production")

        settings = get_settings()

        assert settings.openai_api_key == "custom-key"
        assert settings.classifier_model == "openai:gpt-4o"
        assert settings.enterprise_threshold == 50
        assert settings.daily_cost_limit_usd == 200.0
        assert settings.max_retries == 5
        assert settings.log_level == "DEBUG"
        assert settings.environment == "production"

    def test_boolean_environment_variables(self, monkeypatch):
        """Verify boolean environment variables parse correctly."""
        get_settings.cache_clear()

        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("ENABLE_PII_FILTERING", "false")
        monkeypatch.setenv("ENABLE_STRUCTURED_LOGGING", "true")

        settings = get_settings()

        assert settings.enable_pii_filtering is False
        assert settings.enable_structured_logging is True

    def test_singleton_pattern(self):
        """Verify get_settings() returns same instance."""
        get_settings.cache_clear()

        settings1 = get_settings()
        settings2 = get_settings()

        assert settings1 is settings2

    def test_production_configuration(self, monkeypatch):
        """Verify production environment configuration."""
        get_settings.cache_clear()

        monkeypatch.setenv("OPENAI_API_KEY", "prod-key")
        monkeypatch.setenv("ENVIRONMENT", "production")
        monkeypatch.setenv("ENABLE_STRUCTURED_LOGGING", "true")
        monkeypatch.setenv("LOG_LEVEL", "WARNING")
        monkeypatch.setenv("DAILY_COST_LIMIT_USD", "1000.0")

        settings = get_settings()

        assert settings.environment == "production"
        assert settings.enable_structured_logging is True
        assert settings.log_level == "WARNING"
        assert settings.daily_cost_limit_usd == 1000.0

    def test_development_configuration(self, monkeypatch):
        """Verify development environment configuration."""
        get_settings.cache_clear()

        monkeypatch.setenv("OPENAI_API_KEY", "dev-key")
        monkeypatch.setenv("ENVIRONMENT", "development")
        monkeypatch.setenv("CLASSIFIER_MODEL", "openai:gpt-4o-mini")
        monkeypatch.setenv("DIRECTOR_MODEL", "openai:gpt-4o-mini")
        monkeypatch.setenv("EXECUTOR_MODEL", "openai:gpt-4o-mini")
        monkeypatch.setenv("DAILY_COST_LIMIT_USD", "10.0")

        settings = get_settings()

        assert settings.environment == "development"
        assert settings.classifier_model == "openai:gpt-4o-mini"
        assert settings.director_model == "openai:gpt-4o-mini"
        assert settings.executor_model == "openai:gpt-4o-mini"
        assert settings.daily_cost_limit_usd == 10.0

    def test_retry_configuration_bounds(self, monkeypatch):
        """Verify retry configuration accepts valid values."""
        get_settings.cache_clear()

        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("MAX_RETRIES", "10")
        monkeypatch.setenv("RETRY_MIN_WAIT_SECONDS", "1")
        monkeypatch.setenv("RETRY_MAX_WAIT_SECONDS", "30")

        settings = get_settings()

        assert settings.max_retries == 10
        assert settings.retry_min_wait_seconds == 1
        assert settings.retry_max_wait_seconds == 30

    def test_cost_limit_configuration(self, monkeypatch):
        """Verify cost limits can be configured independently."""
        get_settings.cache_clear()

        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("HOURLY_COST_LIMIT_USD", "50.0")
        monkeypatch.setenv("DAILY_COST_LIMIT_USD", "500.0")

        settings = get_settings()

        assert settings.hourly_cost_limit_usd == 50.0
        assert settings.daily_cost_limit_usd == 500.0

    def test_model_configuration_flexibility(self, monkeypatch):
        """Verify each agent can use different models."""
        get_settings.cache_clear()

        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("CLASSIFIER_MODEL", "openai:gpt-4o-mini")
        monkeypatch.setenv("DIRECTOR_MODEL", "openai:gpt-4o")
        monkeypatch.setenv("EXECUTOR_MODEL", "openai:gpt-3.5-turbo")

        settings = get_settings()

        # Verify each agent can have independent model config
        assert settings.classifier_model != settings.director_model
        assert settings.director_model != settings.executor_model
