"""
Security Validation Layer
Pre-pipeline defense against malicious inputs, prompt injection, and abuse.
"""
import re
from dataclasses import dataclass
from typing import Optional, List
from enum import StrEnum
from loguru import logger


class ThreatType(StrEnum):
    """Types of security threats detected."""
    PROMPT_INJECTION = "prompt_injection"
    CONTEXT_FLOODING = "context_flooding"
    PII_LEAKAGE = "pii_leakage"
    PROFANITY = "profanity"
    INJECTION_ATTEMPT = "injection_attempt"
    SUSPICIOUS_PATTERN = "suspicious_pattern"


@dataclass
class SecurityThreat:
    """Represents a detected security threat."""
    threat_type: ThreatType
    severity: str  # "low", "medium", "high", "critical"
    description: str
    matched_pattern: Optional[str] = None
    recommended_action: str = "block"  # "block", "sanitize", "warn"


@dataclass
class ValidationResult:
    """Result of security validation."""
    is_safe: bool
    sanitized_message: str
    threats: List[SecurityThreat]

    @property
    def has_critical_threats(self) -> bool:
        """Check if any critical threats were detected."""
        return any(t.severity == "critical" for t in self.threats)

    @property
    def should_block(self) -> bool:
        """Determine if message should be blocked."""
        return not self.is_safe or self.has_critical_threats


class SecurityValidator:
    """
    Comprehensive security validation for incoming messages.

    Protects against:
    - Prompt injection and jailbreak attempts
    - Context window flooding
    - PII leakage (credit cards, SSNs, emails)
    - Profanity and toxicity
    - SQL injection, XSS, command injection

    Usage:
        >>> validator = SecurityValidator()
        >>> result = validator.validate_message("Hello, I need help", "+1234567890")
        >>> if result.should_block:
        ...     # Reject message
    """

    # Prompt injection patterns
    JAILBREAK_PATTERNS = [
        r"ignore\s+(all\s+)?previous\s+instructions?",
        r"ignore\s+above",
        r"disregard\s+(all\s+)?previous",
        r"forget\s+(all\s+)?previous",
        r"you\s+are\s+now\s+(a|an)",
        r"new\s+instructions?:",
        r"system\s*:\s*",
        r"admin\s+mode",
        r"developer\s+mode",
        r"DAN\s+mode",
        r"jailbreak",
        r"pretend\s+you\s+are",
        r"act\s+as\s+(if\s+)?(a|an)",
        r"roleplay\s+as",
        r"<\s*system\s*>",
        r"</\s*system\s*>",
        r"\[\s*system\s*\]",
        r"override\s+instructions?",
    ]

    # Delimiter injection attempts
    DELIMITER_PATTERNS = [
        r"```python",
        r"```javascript",
        r"```bash",
        r"<\|im_start\|>",
        r"<\|im_end\|>",
        r"###\s*Instruction",
        r"###\s*Response",
    ]

    # SQL injection patterns
    SQL_INJECTION_PATTERNS = [
        r"'\s*OR\s+'?1'?\s*=\s*'?1",
        r";\s*DROP\s+TABLE",
        r";\s*DELETE\s+FROM",
        r"UNION\s+SELECT",
        r"exec\s*\(",
        r"execute\s*\(",
        r"--\s*$",
        r"xp_cmdshell",
    ]

    # XSS patterns
    XSS_PATTERNS = [
        r"<script[^>]*>",
        r"javascript\s*:",
        r"onerror\s*=",
        r"onload\s*=",
        r"onclick\s*=",
        r"<iframe",
        r"eval\s*\(",
    ]

    # Command injection patterns
    COMMAND_INJECTION_PATTERNS = [
        r";\s*rm\s+-rf",
        r";\s*cat\s+/etc/passwd",
        r"\|\s*bash",
        r"&\s*curl",
        r"`.*`",  # Backtick command substitution
        r"\$\(.*\)",  # Command substitution
    ]

    # Basic profanity list
    PROFANITY_WORDS = [
        "fuck", "shit", "bitch", "asshole", "cunt", "damn",
        "piss", "bastard", "slut", "whore", "dick", "cock"
    ]

    # Hate speech indicators
    HATE_SPEECH_PATTERNS = [
        r"\bn[i1]gg[ea]r",
        r"\bf[a@]gg[o0]t",
        r"\bk[i1]ke",
        r"\bretard",
    ]

    # PII patterns
    CREDIT_CARD_PATTERN = re.compile(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b')
    SSN_PATTERN = re.compile(r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b')
    EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')

    # Context flooding thresholds
    MAX_MESSAGE_LENGTH = 5000  # characters
    MAX_REPEATED_CHARS = 50  # consecutive repeated characters
    MAX_WORD_REPETITION = 10  # same word repeated

    def __init__(self, max_message_length: int = MAX_MESSAGE_LENGTH):
        """
        Initialize security validator.

        Args:
            max_message_length: Maximum allowed message length
        """
        self.max_message_length = max_message_length
        logger.debug("SecurityValidator initialized")

    def validate_message(self, message: str, lead_id: str) -> ValidationResult:
        """
        Validate incoming message against all security threats.

        Args:
            message: The incoming message content
            lead_id: Lead identifier (for PII context)

        Returns:
            ValidationResult with safety assessment and sanitized message
        """
        threats: List[SecurityThreat] = []
        sanitized = message

        # Check for prompt injection
        if threat := self.detect_prompt_injection(message):
            threats.append(threat)

        # Check for context flooding
        if threat := self.detect_context_flooding(message):
            threats.append(threat)

        # Check for PII and redact
        has_pii, sanitized = self.detect_and_redact_pii(message, lead_id)
        if has_pii:
            threats.append(SecurityThreat(
                threat_type=ThreatType.PII_LEAKAGE,
                severity="medium",
                description="Message contains PII (redacted)",
                recommended_action="sanitize"
            ))

        # Check for profanity
        if threat := self.detect_profanity(message):
            threats.append(threat)

        # Check for injection attempts
        if threat := self.detect_injection_attempts(message):
            threats.append(threat)

        # Determine if message is safe
        # Block if any threat has recommended_action="block"
        is_safe = not any(t.recommended_action == "block" for t in threats)

        if threats:
            logger.warning(
                f"Security threats detected in message from {lead_id}",
                extra={
                    "lead_id": lead_id,
                    "threat_count": len(threats),
                    "threats": [t.threat_type for t in threats]
                }
            )

        return ValidationResult(
            is_safe=is_safe,
            sanitized_message=sanitized,
            threats=threats
        )

    def detect_prompt_injection(self, message: str) -> Optional[SecurityThreat]:
        """
        Detect prompt injection and jailbreak attempts.

        Args:
            message: Message content to analyze

        Returns:
            SecurityThreat if injection detected, None otherwise
        """
        message_lower = message.lower()

        # Check jailbreak patterns
        for pattern in self.JAILBREAK_PATTERNS:
            if re.search(pattern, message_lower, re.IGNORECASE):
                return SecurityThreat(
                    threat_type=ThreatType.PROMPT_INJECTION,
                    severity="critical",
                    description="Jailbreak attempt detected",
                    matched_pattern=pattern,
                    recommended_action="block"
                )

        # Check delimiter injection
        for pattern in self.DELIMITER_PATTERNS:
            if re.search(pattern, message, re.IGNORECASE):
                return SecurityThreat(
                    threat_type=ThreatType.PROMPT_INJECTION,
                    severity="high",
                    description="Delimiter injection attempt",
                    matched_pattern=pattern,
                    recommended_action="block"
                )

        return None

    def detect_context_flooding(self, message: str) -> Optional[SecurityThreat]:
        """
        Detect context window flooding attacks.

        Args:
            message: Message content to analyze

        Returns:
            SecurityThreat if flooding detected, None otherwise
        """
        # Check message length
        if len(message) > self.max_message_length:
            return SecurityThreat(
                threat_type=ThreatType.CONTEXT_FLOODING,
                severity="high",
                description=f"Message exceeds max length ({len(message)} > {self.max_message_length})",
                recommended_action="block"
            )

        # Check for repeated characters (e.g., "aaaaaaaaaa...")
        for char in set(message):
            if char * self.MAX_REPEATED_CHARS in message:
                return SecurityThreat(
                    threat_type=ThreatType.CONTEXT_FLOODING,
                    severity="medium",
                    description="Repeated character spam detected",
                    matched_pattern=char * self.MAX_REPEATED_CHARS,
                    recommended_action="block"
                )

        # Check for word repetition
        words = message.lower().split()
        if words:
            from collections import Counter
            word_counts = Counter(words)
            max_count = max(word_counts.values())
            if max_count > self.MAX_WORD_REPETITION:
                return SecurityThreat(
                    threat_type=ThreatType.CONTEXT_FLOODING,
                    severity="medium",
                    description="Excessive word repetition detected",
                    recommended_action="block"
                )

        return None

    def detect_and_redact_pii(self, message: str, lead_id: str) -> tuple[bool, str]:
        """
        Detect and redact PII from message.

        Args:
            message: Message content to analyze
            lead_id: Lead identifier (phone number not redacted)

        Returns:
            Tuple of (has_pii: bool, sanitized_message: str)
        """
        has_pii = False
        sanitized = message

        # Redact credit card numbers
        if self.CREDIT_CARD_PATTERN.search(sanitized):
            # Validate with Luhn algorithm
            potential_cards = self.CREDIT_CARD_PATTERN.findall(sanitized)
            for card in potential_cards:
                card_digits = re.sub(r'[\s-]', '', card)
                if self._is_valid_credit_card(card_digits):
                    sanitized = sanitized.replace(card, "[CREDIT_CARD_REDACTED]")
                    has_pii = True

        # Redact SSNs
        if self.SSN_PATTERN.search(sanitized):
            sanitized = self.SSN_PATTERN.sub("[SSN_REDACTED]", sanitized)
            has_pii = True

        # Redact emails
        if self.EMAIL_PATTERN.search(sanitized):
            sanitized = self.EMAIL_PATTERN.sub("[EMAIL_REDACTED]", sanitized)
            has_pii = True

        return has_pii, sanitized

    def detect_profanity(self, message: str) -> Optional[SecurityThreat]:
        """
        Detect profanity and hate speech.

        Args:
            message: Message content to analyze

        Returns:
            SecurityThreat if profanity detected, None otherwise
        """
        message_lower = message.lower()

        # Check hate speech patterns first (higher severity)
        for pattern in self.HATE_SPEECH_PATTERNS:
            if re.search(pattern, message_lower, re.IGNORECASE):
                return SecurityThreat(
                    threat_type=ThreatType.PROFANITY,
                    severity="critical",
                    description="Hate speech detected",
                    recommended_action="block"
                )

        # Check basic profanity
        words = re.findall(r'\b\w+\b', message_lower)
        for word in words:
            if word in self.PROFANITY_WORDS:
                return SecurityThreat(
                    threat_type=ThreatType.PROFANITY,
                    severity="low",
                    description="Profanity detected",
                    matched_pattern=word,
                    recommended_action="warn"
                )

        return None

    def detect_injection_attempts(self, message: str) -> Optional[SecurityThreat]:
        """
        Detect SQL injection, XSS, and command injection attempts.

        Args:
            message: Message content to analyze

        Returns:
            SecurityThreat if injection attempt detected, None otherwise
        """
        # Check SQL injection
        for pattern in self.SQL_INJECTION_PATTERNS:
            if re.search(pattern, message, re.IGNORECASE):
                return SecurityThreat(
                    threat_type=ThreatType.INJECTION_ATTEMPT,
                    severity="high",
                    description="SQL injection attempt detected",
                    matched_pattern=pattern,
                    recommended_action="block"
                )

        # Check XSS
        for pattern in self.XSS_PATTERNS:
            if re.search(pattern, message, re.IGNORECASE):
                return SecurityThreat(
                    threat_type=ThreatType.INJECTION_ATTEMPT,
                    severity="high",
                    description="XSS attempt detected",
                    matched_pattern=pattern,
                    recommended_action="block"
                )

        # Check command injection
        for pattern in self.COMMAND_INJECTION_PATTERNS:
            if re.search(pattern, message):
                return SecurityThreat(
                    threat_type=ThreatType.INJECTION_ATTEMPT,
                    severity="critical",
                    description="Command injection attempt detected",
                    matched_pattern=pattern,
                    recommended_action="block"
                )

        return None

    @staticmethod
    def _is_valid_credit_card(card_number: str) -> bool:
        """
        Validate credit card using Luhn algorithm.

        Args:
            card_number: Card number digits only

        Returns:
            True if valid credit card number
        """
        if not card_number.isdigit() or len(card_number) not in [13, 14, 15, 16, 19]:
            return False

        # Luhn algorithm
        def luhn_checksum(card):
            def digits_of(n):
                return [int(d) for d in str(n)]

            digits = digits_of(card)
            odd_digits = digits[-1::-2]
            even_digits = digits[-2::-2]
            checksum = sum(odd_digits)
            for d in even_digits:
                checksum += sum(digits_of(d * 2))
            return checksum % 10

        return luhn_checksum(card_number) == 0
