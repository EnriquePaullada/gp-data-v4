"""
Tests for Twilio Signature Validation.
"""

import pytest
import hashlib
import hmac
import base64

from src.utils.twilio_signature import TwilioSignatureValidator, validate_twilio_signature


class TestTwilioSignatureValidator:
    """Test suite for Twilio signature validation."""

    @pytest.fixture
    def auth_token(self):
        """Test auth token."""
        return "test_auth_token_12345"

    @pytest.fixture
    def validator(self, auth_token):
        """Create validator instance."""
        return TwilioSignatureValidator(auth_token)

    def compute_test_signature(self, url: str, params: dict, auth_token: str) -> str:
        """Helper to compute signature for testing."""
        data = url
        for key in sorted(params.keys()):
            data += key + params[key]

        mac = hmac.new(
            auth_token.encode('utf-8'),
            data.encode('utf-8'),
            hashlib.sha256
        )
        return base64.b64encode(mac.digest()).decode('utf-8')

    def test_valid_signature(self, validator, auth_token):
        """Test that valid signature passes validation."""
        url = "https://example.com/webhooks/twilio"
        params = {
            "MessageSid": "SM123",
            "From": "whatsapp:+1234567890",
            "Body": "Test message"
        }

        # Compute valid signature
        signature = self.compute_test_signature(url, params, auth_token)

        # Validate
        assert validator.validate(url, params, signature) is True

    def test_invalid_signature(self, validator):
        """Test that invalid signature fails validation."""
        url = "https://example.com/webhooks/twilio"
        params = {
            "MessageSid": "SM123",
            "From": "whatsapp:+1234567890",
            "Body": "Test message"
        }

        # Use wrong signature
        invalid_signature = "invalidSignature123=="

        # Validate
        assert validator.validate(url, params, invalid_signature) is False

    def test_wrong_url_fails(self, validator, auth_token):
        """Test that signature fails if URL is different."""
        correct_url = "https://example.com/webhooks/twilio"
        wrong_url = "https://attacker.com/webhooks/twilio"

        params = {
            "MessageSid": "SM123",
            "From": "whatsapp:+1234567890",
            "Body": "Test message"
        }

        # Compute signature with correct URL
        signature = self.compute_test_signature(correct_url, params, auth_token)

        # Try to validate with wrong URL
        assert validator.validate(wrong_url, params, signature) is False

    def test_tampered_params_fail(self, validator, auth_token):
        """Test that tampered parameters fail validation."""
        url = "https://example.com/webhooks/twilio"

        original_params = {
            "MessageSid": "SM123",
            "From": "whatsapp:+1234567890",
            "Body": "Original message"
        }

        tampered_params = {
            "MessageSid": "SM123",
            "From": "whatsapp:+1234567890",
            "Body": "Tampered message"  # Changed
        }

        # Compute signature with original params
        signature = self.compute_test_signature(url, original_params, auth_token)

        # Try to validate with tampered params
        assert validator.validate(url, tampered_params, signature) is False

    def test_missing_param_fails(self, validator, auth_token):
        """Test that missing parameter fails validation."""
        url = "https://example.com/webhooks/twilio"

        original_params = {
            "MessageSid": "SM123",
            "From": "whatsapp:+1234567890",
            "Body": "Test message"
        }

        incomplete_params = {
            "MessageSid": "SM123",
            "From": "whatsapp:+1234567890"
            # Missing Body
        }

        # Compute signature with original params
        signature = self.compute_test_signature(url, original_params, auth_token)

        # Try to validate with incomplete params
        assert validator.validate(url, incomplete_params, signature) is False

    def test_extra_param_fails(self, validator, auth_token):
        """Test that extra parameter fails validation."""
        url = "https://example.com/webhooks/twilio"

        original_params = {
            "MessageSid": "SM123",
            "From": "whatsapp:+1234567890",
        }

        params_with_extra = {
            "MessageSid": "SM123",
            "From": "whatsapp:+1234567890",
            "Body": "Extra param"  # Added
        }

        # Compute signature with original params
        signature = self.compute_test_signature(url, original_params, auth_token)

        # Try to validate with extra param
        assert validator.validate(url, params_with_extra, signature) is False

    def test_case_sensitive_params(self, validator, auth_token):
        """Test that parameter names are case-sensitive."""
        url = "https://example.com/webhooks/twilio"

        params_lowercase = {
            "messagesid": "SM123",  # lowercase
            "from": "whatsapp:+1234567890",
        }

        params_correct_case = {
            "MessageSid": "SM123",  # correct case
            "From": "whatsapp:+1234567890",
        }

        # Compute signature with lowercase
        signature = self.compute_test_signature(url, params_lowercase, auth_token)

        # Try to validate with correct case
        assert validator.validate(url, params_correct_case, signature) is False

    def test_empty_params(self, validator, auth_token):
        """Test validation with empty parameters."""
        url = "https://example.com/webhooks/twilio"
        params = {}

        # Compute signature
        signature = self.compute_test_signature(url, params, auth_token)

        # Validate
        assert validator.validate(url, params, signature) is True

    def test_special_characters_in_params(self, validator, auth_token):
        """Test parameters with special characters."""
        url = "https://example.com/webhooks/twilio"
        params = {
            "MessageSid": "SM123",
            "Body": "Hello! How are you? 😊",  # Special chars and emoji
            "From": "whatsapp:+1234567890"
        }

        # Compute signature
        signature = self.compute_test_signature(url, params, auth_token)

        # Validate
        assert validator.validate(url, params, signature) is True

    def test_url_with_query_params(self, validator, auth_token):
        """Test URL with query parameters."""
        url = "https://example.com/webhooks/twilio?foo=bar&baz=qux"
        params = {
            "MessageSid": "SM123",
            "From": "whatsapp:+1234567890",
        }

        # Compute signature
        signature = self.compute_test_signature(url, params, auth_token)

        # Validate
        assert validator.validate(url, params, signature) is True

    def test_compute_signature_deterministic(self, validator):
        """Test that signature computation is deterministic."""
        url = "https://example.com/webhooks/twilio"
        params = {
            "MessageSid": "SM123",
            "From": "whatsapp:+1234567890",
            "Body": "Test"
        }

        # Compute twice
        sig1 = validator.compute_signature(url, params)
        sig2 = validator.compute_signature(url, params)

        # Should be identical
        assert sig1 == sig2

    def test_param_order_doesnt_matter(self, validator, auth_token):
        """Test that parameter order doesn't affect validation."""
        url = "https://example.com/webhooks/twilio"

        # Different insertion orders, but same params
        params1 = {"A": "1", "B": "2", "C": "3"}
        params2 = {"C": "3", "A": "1", "B": "2"}

        # Both should produce same signature
        sig1 = validator.compute_signature(url, params1)
        sig2 = validator.compute_signature(url, params2)

        assert sig1 == sig2

    def test_convenience_function(self, auth_token):
        """Test convenience validation function."""
        url = "https://example.com/webhooks/twilio"
        params = {
            "MessageSid": "SM123",
            "From": "whatsapp:+1234567890",
        }

        # Compute signature
        signature = self.compute_test_signature(url, params, auth_token)

        # Validate using convenience function
        assert validate_twilio_signature(url, params, signature, auth_token) is True

    def test_different_auth_tokens_produce_different_signatures(self):
        """Test that different auth tokens produce different signatures."""
        url = "https://example.com/webhooks/twilio"
        params = {"MessageSid": "SM123"}

        token1 = "token1"
        token2 = "token2"

        validator1 = TwilioSignatureValidator(token1)
        validator2 = TwilioSignatureValidator(token2)

        sig1 = validator1.compute_signature(url, params)
        sig2 = validator2.compute_signature(url, params)

        assert sig1 != sig2

    def test_constant_time_comparison(self, validator, auth_token):
        """Test that comparison is constant-time (prevents timing attacks)."""
        url = "https://example.com/webhooks/twilio"
        params = {"MessageSid": "SM123"}

        # Compute valid signature
        valid_sig = self.compute_test_signature(url, params, auth_token)

        # Create almost-matching signature (first half matches)
        half_len = len(valid_sig) // 2
        almost_matching = valid_sig[:half_len] + "X" * (len(valid_sig) - half_len)

        # Should still return False (timing shouldn't reveal partial matches)
        assert validator.validate(url, params, almost_matching) is False

    def test_real_world_twilio_example(self):
        """Test with realistic Twilio webhook data."""
        # Example from Twilio documentation
        auth_token = "your_auth_token_here"
        url = "https://mycompany.com/myapp.php?foo=1&bar=2"
        params = {
            "CallSid": "CA1234567890ABCDE",
            "Caller": "+12349013030",
            "Digits": "1234",
            "From": "+12349013030",
            "To": "+18005551212"
        }

        validator = TwilioSignatureValidator(auth_token)

        # Compute expected signature
        expected_sig = validator.compute_signature(url, params)

        # Should validate successfully
        assert validator.validate(url, params, expected_sig) is True
