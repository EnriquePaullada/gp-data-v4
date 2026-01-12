"""
Tests for Phone Number Normalization

Validates E.164 normalization and Mexico edge cases.
"""

import pytest
from src.utils.phone_normalizer import (
    PhoneNormalizer,
    PhoneNormalizationError,
    normalize_phone,
    get_phone_normalizer,
)


class TestPhoneNormalizer:
    """Tests for PhoneNormalizer class."""

    @pytest.fixture
    def normalizer(self):
        return PhoneNormalizer()

    # ===========================================
    # Basic E.164 Normalization
    # ===========================================

    def test_already_e164(self, normalizer):
        """Already E.164 formatted number should pass through."""
        result = normalizer.normalize("+525512345678")
        assert result.e164 == "+525512345678"

    def test_strips_formatting(self, normalizer):
        """Should strip spaces, dashes, parentheses."""
        result = normalizer.normalize("+52 (55) 1234-5678")
        assert result.e164 == "+525512345678"

    def test_adds_plus_sign(self, normalizer):
        """Should add + if missing."""
        result = normalizer.normalize("525512345678")
        assert result.e164 == "+525512345678"

    # ===========================================
    # Mexico Mobile Edge Case (+52 1 XX...)
    # ===========================================

    def test_mexico_mobile_with_1_prefix(self, normalizer):
        """Mexico mobile with '1' prefix should normalize correctly."""
        # Old format: +52 1 55 1234 5678
        result = normalizer.normalize("+5215512345678")
        assert result.e164 == "+525512345678"
        assert result.national_number == "5512345678"

    def test_mexico_mobile_without_1_prefix(self, normalizer):
        """Mexico mobile without '1' prefix should stay the same."""
        result = normalizer.normalize("+525512345678")
        assert result.e164 == "+525512345678"
        assert result.national_number == "5512345678"

    def test_mexico_mobile_formats_equivalent(self, normalizer):
        """Both Mexico mobile formats should normalize to same number."""
        with_1 = normalizer.normalize("+52 1 55 1234 5678")
        without_1 = normalizer.normalize("+52 55 1234 5678")
        assert with_1.e164 == without_1.e164

    def test_mexico_landline_preserved(self, normalizer):
        """Mexico landline numbers should not be modified."""
        # Mexico City landline (8 digits, area code 55)
        result = normalizer.normalize("+52 55 1234 5678")
        assert result.e164 == "+525512345678"

    def test_mexico_with_spaces(self, normalizer):
        """Mexico number with various spacing."""
        result = normalizer.normalize("+52 1 55 1234 5678")
        assert result.e164 == "+525512345678"

    # ===========================================
    # International Numbers
    # ===========================================

    def test_us_number(self, normalizer):
        """US numbers should normalize correctly."""
        result = normalizer.normalize("+1 (212) 555-1234")
        assert result.e164 == "+12125551234"
        assert result.country_code == "1"
        assert result.region == "US"

    def test_uk_number(self, normalizer):
        """UK numbers should normalize correctly."""
        result = normalizer.normalize("+44 20 7123 4567")
        assert result.e164 == "+442071234567"
        assert result.country_code == "44"

    def test_spain_number(self, normalizer):
        """Spain numbers should normalize correctly."""
        result = normalizer.normalize("+34 612 345 678")
        assert result.e164 == "+34612345678"
        assert result.country_code == "34"

    # ===========================================
    # Default Region Handling
    # ===========================================

    def test_default_region_mexico(self, normalizer):
        """Without country code, should assume Mexico."""
        result = normalizer.normalize("5512345678")
        assert result.e164 == "+525512345678"
        assert result.country_code == "52"

    def test_override_default_region(self, normalizer):
        """Should use provided default region."""
        result = normalizer.normalize("2071234567", default_region="GB")
        assert result.country_code == "44"

    # ===========================================
    # Validation
    # ===========================================

    def test_invalid_number_raises(self, normalizer):
        """Invalid number should raise error."""
        with pytest.raises(PhoneNormalizationError):
            normalizer.normalize("+52 123")  # Too short

    def test_non_numeric_raises(self, normalizer):
        """Non-numeric input should raise error."""
        with pytest.raises(PhoneNormalizationError):
            normalizer.normalize("not a phone")

    def test_empty_raises(self, normalizer):
        """Empty input should raise error."""
        with pytest.raises(PhoneNormalizationError):
            normalizer.normalize("")

    def test_is_valid_true(self, normalizer):
        """is_valid should return True for valid numbers."""
        assert normalizer.is_valid("+525512345678") is True

    def test_is_valid_false(self, normalizer):
        """is_valid should return False for invalid numbers."""
        assert normalizer.is_valid("invalid") is False

    # ===========================================
    # Equivalence Checking
    # ===========================================

    def test_are_equivalent_same_number(self, normalizer):
        """Same number in different formats should be equivalent."""
        assert normalizer.are_equivalent(
            "+52 1 55 1234 5678",
            "+52 55 1234 5678"
        ) is True

    def test_are_equivalent_different_numbers(self, normalizer):
        """Different numbers should not be equivalent."""
        assert normalizer.are_equivalent(
            "+525512345678",
            "+525598765432"
        ) is False

    def test_are_equivalent_invalid_returns_false(self, normalizer):
        """Invalid numbers should return False."""
        assert normalizer.are_equivalent("invalid", "+525512345678") is False

    # ===========================================
    # Metadata
    # ===========================================

    def test_mobile_detection(self, normalizer):
        """Should detect mobile numbers."""
        result = normalizer.normalize("+525512345678")
        assert result.is_mobile is True

    def test_region_detection(self, normalizer):
        """Should detect region."""
        result = normalizer.normalize("+525512345678")
        assert result.region == "MX"

    def test_preserves_original(self, normalizer):
        """Should preserve original input."""
        original = "+52 (55) 1234-5678"
        result = normalizer.normalize(original)
        assert result.original == original

    # ===========================================
    # Display Formatting
    # ===========================================

    def test_format_display(self, normalizer):
        """Should format for display."""
        formatted = normalizer.format_display("+525512345678")
        assert " " in formatted  # Has spaces
        assert formatted.startswith("+52")


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_normalize_phone(self):
        """normalize_phone should return E.164 string."""
        result = normalize_phone("+52 1 55 1234 5678")
        assert result == "+525512345678"

    def test_get_phone_normalizer_singleton(self):
        """Should return same instance."""
        n1 = get_phone_normalizer()
        n2 = get_phone_normalizer()
        assert n1 is n2


class TestMexicoEdgeCases:
    """Focused tests on Mexico-specific edge cases."""

    @pytest.fixture
    def normalizer(self):
        return PhoneNormalizer()

    @pytest.mark.parametrize("input_phone,expected", [
        # With "1" prefix (old mobile format)
        ("+5215512345678", "+525512345678"),
        ("+52 1 55 1234 5678", "+525512345678"),
        ("+521 55 12345678", "+525512345678"),
        # Without "1" prefix (new format)
        ("+525512345678", "+525512345678"),
        ("+52 55 1234 5678", "+525512345678"),
        # Different area codes with "1"
        ("+5213312345678", "+523312345678"),  # Guadalajara
        ("+5218112345678", "+528112345678"),  # Monterrey
        # Without country code (defaults to Mexico)
        ("5512345678", "+525512345678"),
    ])
    def test_mexico_normalization_variants(self, normalizer, input_phone, expected):
        """Test various Mexico phone formats normalize correctly."""
        result = normalizer.normalize(input_phone)
        assert result.e164 == expected
