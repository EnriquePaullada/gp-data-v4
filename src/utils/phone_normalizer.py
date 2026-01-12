"""
Phone Number Normalization Service

Handles E.164 validation, normalization, and Mexico-specific edge cases.
Mexican mobile numbers historically had an extra "1" after country code
that must be handled for proper deduplication.
"""

import phonenumbers
from phonenumbers import NumberParseException, PhoneNumberFormat
from dataclasses import dataclass
from typing import Optional
from src.utils.observability import logger


@dataclass
class NormalizedPhone:
    """Result of phone normalization."""
    original: str
    e164: str  # Normalized E.164 format
    country_code: str  # e.g., "52" for Mexico
    national_number: str  # Number without country code
    is_mobile: bool
    is_valid: bool
    region: str  # e.g., "MX" for Mexico


class PhoneNormalizationError(Exception):
    """Raised when phone number cannot be normalized."""
    pass


class PhoneNormalizer:
    """
    Normalizes phone numbers to E.164 format with special handling
    for Mexican mobile numbers.

    Mexico Edge Case:
    - Old format: +52 1 55 1234 5678 (mobile with "1" prefix)
    - New format: +52 55 1234 5678 (mobile without "1" prefix)
    - Both should normalize to: +525512345678

    Usage:
        normalizer = PhoneNormalizer()
        result = normalizer.normalize("+52 1 55 1234 5678")
        print(result.e164)  # "+525512345678"
    """

    # Mexico country code
    MEXICO_COUNTRY_CODE = "52"

    # Default region for parsing ambiguous numbers
    DEFAULT_REGION = "MX"

    def normalize(
        self,
        phone: str,
        default_region: Optional[str] = None,
    ) -> NormalizedPhone:
        """
        Normalize a phone number to E.164 format.

        Args:
            phone: Phone number in any format
            default_region: ISO country code for parsing (default: MX)

        Returns:
            NormalizedPhone with normalized data

        Raises:
            PhoneNormalizationError: If number is invalid
        """
        original = phone
        region = default_region or self.DEFAULT_REGION

        # Clean input
        phone = self._clean_input(phone)

        # Pre-process Mexico mobile "1" prefix BEFORE parsing
        # phonenumbers library doesn't recognize old +52 1 format
        phone = self._preprocess_mexico_mobile(phone)

        try:
            # Parse the phone number
            parsed = phonenumbers.parse(phone, region)

            # Validate
            if not phonenumbers.is_valid_number(parsed):
                raise PhoneNormalizationError(
                    f"Invalid phone number: {original}"
                )

            # Get country code and national number
            country_code = str(parsed.country_code)
            national = str(parsed.national_number)

            # Handle Mexico mobile "1" prefix edge case
            if country_code == self.MEXICO_COUNTRY_CODE:
                national = self._normalize_mexico_number(national)

            # Reconstruct E.164 format
            e164 = f"+{country_code}{national}"

            # Determine if mobile
            number_type = phonenumbers.number_type(parsed)
            is_mobile = number_type in (
                phonenumbers.PhoneNumberType.MOBILE,
                phonenumbers.PhoneNumberType.FIXED_LINE_OR_MOBILE,
            )

            # Get region
            region_code = phonenumbers.region_code_for_number(parsed)

            result = NormalizedPhone(
                original=original,
                e164=e164,
                country_code=country_code,
                national_number=national,
                is_mobile=is_mobile,
                is_valid=True,
                region=region_code or region,
            )

            logger.debug(
                f"Normalized phone: {original} -> {e164}",
                extra={"region": region_code, "is_mobile": is_mobile}
            )

            return result

        except NumberParseException as e:
            raise PhoneNormalizationError(
                f"Cannot parse phone number '{original}': {e}"
            )

    def _clean_input(self, phone: str) -> str:
        """Remove common formatting characters."""
        # Keep + at start if present
        if phone.startswith("+"):
            return "+" + "".join(c for c in phone[1:] if c.isdigit())
        return "".join(c for c in phone if c.isdigit())

    def _preprocess_mexico_mobile(self, phone: str) -> str:
        """
        Remove Mexico mobile '1' prefix before parsing.

        The old format +52 1 XX XXXX XXXX is not recognized by
        phonenumbers library, so we convert it to +52 XX XXXX XXXX.

        Args:
            phone: Cleaned phone number

        Returns:
            Phone with Mexico '1' prefix removed if applicable
        """
        # Check for +521 followed by 10 digits (old Mexico mobile format)
        if phone.startswith("+521") and len(phone) == 14:
            normalized = "+52" + phone[4:]
            logger.debug(f"Pre-processed Mexico mobile: {phone} -> {normalized}")
            return normalized

        # Check for 521 followed by 10 digits (without +)
        if phone.startswith("521") and len(phone) == 13:
            normalized = "52" + phone[3:]
            logger.debug(f"Pre-processed Mexico mobile: {phone} -> {normalized}")
            return normalized

        return phone

    def _normalize_mexico_number(self, national: str) -> str:
        """
        Ensure Mexico national number is 10 digits.

        After preprocessing removes the "1" prefix, this method
        handles any edge cases that might slip through.

        Args:
            national: National number (without country code)

        Returns:
            Normalized national number (10 digits for Mexico)
        """
        # Mexican numbers should be 10 digits after preprocessing
        # This handles any edge cases that slip through
        if len(national) == 11 and national.startswith("1"):
            normalized = national[1:]
            logger.debug(f"Removed Mexico mobile '1' prefix: {national} -> {normalized}")
            return normalized
        return national

    def are_equivalent(self, phone1: str, phone2: str) -> bool:
        """
        Check if two phone numbers are equivalent after normalization.

        Useful for deduplication.

        Args:
            phone1: First phone number
            phone2: Second phone number

        Returns:
            True if numbers are equivalent
        """
        try:
            norm1 = self.normalize(phone1)
            norm2 = self.normalize(phone2)
            return norm1.e164 == norm2.e164
        except PhoneNormalizationError:
            return False

    def is_valid(self, phone: str, default_region: Optional[str] = None) -> bool:
        """
        Check if a phone number is valid.

        Args:
            phone: Phone number to validate
            default_region: ISO country code for parsing

        Returns:
            True if valid
        """
        try:
            self.normalize(phone, default_region)
            return True
        except PhoneNormalizationError:
            return False

    def format_display(self, phone: str, default_region: Optional[str] = None) -> str:
        """
        Format phone number for display (international format).

        Args:
            phone: Phone number
            default_region: ISO country code for parsing

        Returns:
            Formatted string like "+52 55 1234 5678"
        """
        try:
            region = default_region or self.DEFAULT_REGION
            parsed = phonenumbers.parse(phone, region)
            return phonenumbers.format_number(parsed, PhoneNumberFormat.INTERNATIONAL)
        except NumberParseException:
            return phone  # Return original if can't format


# Singleton instance
_normalizer: Optional[PhoneNormalizer] = None


def get_phone_normalizer() -> PhoneNormalizer:
    """Get or create the phone normalizer singleton."""
    global _normalizer
    if _normalizer is None:
        _normalizer = PhoneNormalizer()
    return _normalizer


def normalize_phone(phone: str, default_region: str = "MX") -> str:
    """
    Convenience function to normalize a phone number.

    Args:
        phone: Phone number in any format
        default_region: ISO country code for parsing

    Returns:
        E.164 formatted phone number

    Raises:
        PhoneNormalizationError: If number is invalid
    """
    return get_phone_normalizer().normalize(phone, default_region).e164
