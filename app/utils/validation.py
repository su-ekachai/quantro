"""
Input validation and sanitization utilities
"""

from __future__ import annotations

import html
import re
from typing import Any

from pydantic import BaseModel, field_validator


class InputSanitizer:
    """Utility class for input sanitization and validation"""

    # Common XSS patterns
    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"vbscript:",
        r"onload\s*=",
        r"onerror\s*=",
        r"onclick\s*=",
        r"onmouseover\s*=",
        r"<iframe[^>]*>.*?</iframe>",
        r"<object[^>]*>.*?</object>",
        r"<embed[^>]*>.*?</embed>",
        r"<link[^>]*>",
        r"<meta[^>]*>",
    ]

    # SQL injection patterns
    SQL_PATTERNS = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)",
        r"(\b(OR|AND)\s+\d+\s*=\s*\d+)",
        r"(\b(OR|AND)\s+['\"].*['\"])",
        r"(--|#|/\*|\*/)",
        r"(\bUNION\s+SELECT\b)",
        r"(\bINTO\s+OUTFILE\b)",
    ]

    @classmethod
    def sanitize_html(cls, text: str) -> str:
        """Sanitize HTML content to prevent XSS"""
        if not isinstance(text, str):
            return str(text)

        # First remove potentially dangerous patterns before encoding
        sanitized = text
        for pattern in cls.XSS_PATTERNS:
            sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE)

        # Then HTML encode special characters
        sanitized = html.escape(sanitized)

        return sanitized.strip()

    @classmethod
    def validate_sql_input(cls, text: str) -> bool:
        """Check if input contains potential SQL injection patterns"""
        if not isinstance(text, str):
            return True

        for pattern in cls.SQL_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return False

        return True

    @classmethod
    def sanitize_filename(cls, filename: str) -> str:
        """Sanitize filename to prevent directory traversal"""
        if not isinstance(filename, str):
            return "unknown"

        # Remove path separators and dangerous characters
        sanitized = re.sub(r'[<>:"/\\|?*]', "", filename)
        sanitized = re.sub(r"\.\.+", ".", sanitized)
        sanitized = sanitized.strip(". ")

        # Ensure filename is not empty
        if not sanitized:
            return "unknown"

        return sanitized[:255]  # Limit length

    @classmethod
    def validate_email(cls, email: str) -> bool:
        """Validate email format"""
        if not isinstance(email, str):
            return False

        pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        return bool(re.match(pattern, email)) and len(email) <= 255

    @classmethod
    def validate_username(cls, username: str) -> bool:
        """Validate username format"""
        if not isinstance(username, str):
            return False

        # Allow alphanumeric, underscore, hyphen
        pattern = r"^[a-zA-Z0-9_-]{3,50}$"
        return bool(re.match(pattern, username))

    @classmethod
    def validate_password_strength(cls, password: str) -> dict[str, Any]:
        """Validate password strength and return detailed feedback"""
        if not isinstance(password, str):
            return {"valid": False, "errors": ["Password must be a string"]}

        errors = []

        # Length check
        if len(password) < 8:
            errors.append("Password must be at least 8 characters long")

        if len(password) > 128:
            errors.append("Password must be less than 128 characters")

        # Character requirements
        if not re.search(r"[a-z]", password):
            errors.append("Password must contain at least one lowercase letter")

        if not re.search(r"[A-Z]", password):
            errors.append("Password must contain at least one uppercase letter")

        if not re.search(r"\d", password):
            errors.append("Password must contain at least one digit")

        if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
            errors.append("Password must contain at least one special character")

        # Common password patterns
        common_patterns = [
            r"123456",
            r"password",
            r"qwerty",
            r"abc123",
            r"admin",
        ]

        for pattern in common_patterns:
            if re.search(pattern, password, re.IGNORECASE):
                errors.append("Password contains common patterns")
                break

        return {"valid": len(errors) == 0, "errors": errors}


class SecureBaseModel(BaseModel):
    """Base Pydantic model with built-in sanitization"""

    @field_validator("*", mode="before")
    @classmethod
    def sanitize_strings(cls, value: Any) -> Any:
        """Automatically sanitize string inputs"""
        if isinstance(value, str):
            # Basic sanitization for all string fields
            return InputSanitizer.sanitize_html(value)
        return value

    model_config = {
        # Validate assignment to prevent injection after creation
        "validate_assignment": True,
        # Use enum values instead of names
        "use_enum_values": True,
        # Allow population by field name or alias
        "populate_by_name": True,
    }


def validate_request_size(
    content_length: int, max_size: int = 10 * 1024 * 1024
) -> bool:
    """Validate request content length (default 10MB)"""
    return content_length <= max_size


def validate_content_type(content_type: str, allowed_types: list[str]) -> bool:
    """Validate request content type"""
    if not content_type:
        return False

    # Extract main content type (ignore charset, boundary, etc.)
    main_type = content_type.split(";")[0].strip().lower()
    return main_type in [t.lower() for t in allowed_types]
