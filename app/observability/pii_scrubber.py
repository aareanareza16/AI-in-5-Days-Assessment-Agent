# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Active PII scrubbing utility for logs, traces, and session persistence.

Implements Requirement 4 of Category 4: Observability & Tracing:
- Uses regex filters to redact credit card numbers, CVVs/security codes,
  phone numbers, and email addresses before data is written to logs, trace
  attributes, or persistent session history.
"""

from __future__ import annotations

import re
from typing import Any

REDACTED_EMAIL = "[REDACTED_EMAIL]"
REDACTED_PHONE = "[REDACTED_PHONE]"
REDACTED_CREDIT_CARD = "[REDACTED_CREDIT_CARD]"
REDACTED_CVV = "[REDACTED_CVV]"

# Matches standard email addresses
EMAIL_REGEX = re.compile(r"\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\b")

# Matches 13-to-19 digit credit card numbers with optional spaces or hyphens
CREDIT_CARD_REGEX = re.compile(r"\b(?:\d[ -]*?){13,19}\b")

# Matches explicit CVV / CVC / security code patterns in free text
CVV_TEXT_REGEX = re.compile(
    r"\b(cvv2?|cvc2?|cid|security\s+code|card\s+verification(?:\s+value)?)(\s*[:=]\s*|\s+is\s+|\s+)(\d{3,4})\b",
    re.IGNORECASE,
)

# Matches US/international 10-11 digit phone numbers and 7-digit local numbers (e.g., 555-0101)
# Avoids matching ISO dates (YYYY-MM-DD) or Order IDs (ORD-1001).
PHONE_10_DIGIT_REGEX = re.compile(
    r"(?<![\w-])(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]\d{3}[-.\s]\d{4}(?![\w-])"
)
PHONE_7_DIGIT_REGEX = re.compile(r"(?<![\w\d-])\d{3}-\d{4}(?![\w\d-])")

SENSITIVE_CVV_KEYS = frozenset(
    {"cvv", "cvv2", "cvc", "cvc2", "cid", "security_code", "card_cvv"}
)
SENSITIVE_CARD_KEYS = frozenset(
    {"card_number", "credit_card", "credit_card_number", "cc_number", "pan"}
)
SENSITIVE_PHONE_KEYS = frozenset(
    {"phone", "customer_phone", "phone_number", "mobile_phone"}
)
SENSITIVE_EMAIL_KEYS = frozenset({"email", "customer_email", "email_address"})


def scrub_text(text: str) -> str:
    """Scrubs credit card numbers, CVVs, phone numbers, and emails from `text`.

    Order of operations:
    1. Credit card numbers (13-19 digits) so they are not partially matched as phones.
    2. CVV / CVC security codes in text (`CVV: 123` -> `CVV: [REDACTED_CVV]`).
    3. Email addresses.
    4. Phone numbers (10/11-digit and 7-digit formats).

    Args:
        text: Input string potentially containing PII.

    Returns:
        Sanitized string with all detected PII replaced by redaction tokens.
    """
    if not text:
        return text

    scrubbed = CREDIT_CARD_REGEX.sub(REDACTED_CREDIT_CARD, text)
    scrubbed = CVV_TEXT_REGEX.sub(rf"\1\2{REDACTED_CVV}", scrubbed)
    scrubbed = EMAIL_REGEX.sub(REDACTED_EMAIL, scrubbed)
    scrubbed = PHONE_10_DIGIT_REGEX.sub(REDACTED_PHONE, scrubbed)
    scrubbed = PHONE_7_DIGIT_REGEX.sub(REDACTED_PHONE, scrubbed)
    return scrubbed


def scrub_pii(data: Any, *, _key_hint: str | None = None) -> Any:
    """Recursively scrubs PII from strings, dicts, lists, tuples, sets, and models.

    Args:
        data: Arbitrary Python data structure to sanitize before logging or saving.
        _key_hint: Optional enclosing dictionary key name for context-aware scrubbing.

    Returns:
        A sanitized copy of `data` with emails, phones, credit cards, and CVVs redacted.
    """
    key_lower = _key_hint.lower() if _key_hint else ""

    if data is None or isinstance(data, (bool, int, float)):
        if (
            key_lower in SENSITIVE_CVV_KEYS
            and data is not None
            and not isinstance(data, bool)
        ):
            return REDACTED_CVV
        if (
            key_lower in SENSITIVE_CARD_KEYS
            and data is not None
            and not isinstance(data, bool)
        ):
            return REDACTED_CREDIT_CARD
        return data

    if isinstance(data, str):
        if key_lower in SENSITIVE_CVV_KEYS and data.strip():
            return REDACTED_CVV
        if key_lower in SENSITIVE_CARD_KEYS and data.strip():
            return REDACTED_CREDIT_CARD
        if key_lower in SENSITIVE_PHONE_KEYS and data.strip():
            return REDACTED_PHONE
        if key_lower in SENSITIVE_EMAIL_KEYS and data.strip():
            return REDACTED_EMAIL
        return scrub_text(data)

    if hasattr(data, "model_dump"):
        return scrub_pii(data.model_dump())

    if isinstance(data, dict):
        return {str(k): scrub_pii(v, _key_hint=str(k)) for k, v in data.items()}

    if isinstance(data, list):
        return [scrub_pii(item, _key_hint=_key_hint) for item in data]

    if isinstance(data, tuple):
        return tuple(scrub_pii(item, _key_hint=_key_hint) for item in data)

    if isinstance(data, set):
        return {scrub_pii(item, _key_hint=_key_hint) for item in data}

    return scrub_text(str(data))


class PIIScrubber:
    """Configurable PII scrubber helper class for observability pipelines."""

    def scrub(self, data: Any) -> Any:
        """Recursively scrubs PII from `data`."""
        return scrub_pii(data)

    def scrub_string(self, text: str) -> str:
        """Scrubs PII from a single string."""
        return scrub_text(text)
