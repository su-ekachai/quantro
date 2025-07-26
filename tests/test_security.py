"""
Security tests for authentication and protection features
"""

from __future__ import annotations

import time
from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from app.core.middleware import LoginFailureTracker
from app.core.security import SecurityError, SecurityManager
from app.utils.validation import InputSanitizer


class TestSecurityManager:
    """Test security manager functionality"""

    def test_password_hashing(self) -> None:
        """Test password hashing and verification"""
        security_manager = SecurityManager()
        password = "TestPassword123!"

        # Hash password
        hashed = security_manager.hash_password(password)
        assert hashed != password
        assert len(hashed) > 50  # bcrypt hashes are long

        # Verify correct password
        assert security_manager.verify_password(password, hashed)

        # Verify incorrect password
        assert not security_manager.verify_password("WrongPassword", hashed)

    def test_jwt_token_creation_and_verification(self) -> None:
        """Test JWT token creation and verification"""
        security_manager = SecurityManager()
        data = {"user_id": 1, "username": "testuser"}

        # Create token
        token = security_manager.create_access_token(data)
        assert isinstance(token, str)
        assert len(token) > 50

        # Verify token
        payload = security_manager.verify_token(token)
        assert payload["user_id"] == 1
        assert payload["username"] == "testuser"
        assert "exp" in payload
        assert "iat" in payload

    def test_invalid_token_verification(self) -> None:
        """Test verification of invalid tokens"""
        security_manager = SecurityManager()

        # Test invalid token
        with pytest.raises(SecurityError, match="Invalid token"):
            security_manager.verify_token("invalid.token.here")

        # Test expired token (mock time)
        with patch("app.core.security.datetime") as mock_datetime:
            # Create token
            mock_datetime.now.return_value = datetime(2023, 1, 1)
            token = security_manager.create_access_token({"user_id": 1})

            # Verify with future time (expired)
            mock_datetime.now.return_value = datetime(2023, 1, 2)
            with pytest.raises(SecurityError, match="Token has expired"):
                security_manager.verify_token(token)

    def test_secure_token_generation(self) -> None:
        """Test secure token generation"""
        security_manager = SecurityManager()

        token1 = security_manager.generate_secure_token()
        token2 = security_manager.generate_secure_token()

        assert token1 != token2
        assert len(token1) > 30
        assert isinstance(token1, str)

    def test_ip_address_hashing(self) -> None:
        """Test IP address hashing for privacy"""
        security_manager = SecurityManager()

        ip1 = "192.168.1.1"
        ip2 = "192.168.1.2"

        hash1 = security_manager.hash_ip_address(ip1)
        hash2 = security_manager.hash_ip_address(ip2)

        assert hash1 != hash2
        assert len(hash1) == 16  # Truncated SHA256
        assert hash1 == security_manager.hash_ip_address(ip1)  # Consistent


class TestInputSanitizer:
    """Test input sanitization and validation"""

    def test_html_sanitization(self) -> None:
        """Test HTML sanitization for XSS prevention"""
        # Basic HTML encoding - script tags should be removed after encoding
        result = InputSanitizer.sanitize_html("<script>alert('xss')</script>")
        assert "<script>" not in result
        assert "alert" not in result

        # Regular HTML should be encoded
        assert (
            InputSanitizer.sanitize_html("Hello <b>World</b>")
            == "Hello &lt;b&gt;World&lt;/b&gt;"
        )

        # JavaScript injection should be removed
        result = InputSanitizer.sanitize_html("javascript:alert('xss')")
        assert "javascript:" not in result

        result = InputSanitizer.sanitize_html("<img onload='alert()'>")
        assert "onload" not in result

        # Safe content should remain unchanged
        assert InputSanitizer.sanitize_html("Hello World") == "Hello World"
        assert InputSanitizer.sanitize_html("Price: $100") == "Price: $100"

    def test_sql_injection_validation(self) -> None:
        """Test SQL injection pattern detection"""
        # Malicious patterns
        assert not InputSanitizer.validate_sql_input("'; DROP TABLE users; --")
        assert not InputSanitizer.validate_sql_input("1 OR 1=1")
        assert not InputSanitizer.validate_sql_input("UNION SELECT * FROM users")

        # Safe content
        assert InputSanitizer.validate_sql_input("normal text")
        assert InputSanitizer.validate_sql_input("user@example.com")
        assert InputSanitizer.validate_sql_input("Product Name 123")

    def test_filename_sanitization(self) -> None:
        """Test filename sanitization"""
        # Dangerous filenames
        assert InputSanitizer.sanitize_filename("../../../etc/passwd") == "etcpasswd"
        assert InputSanitizer.sanitize_filename("file<script>.txt") == "filescript.txt"
        assert InputSanitizer.sanitize_filename("") == "unknown"

        # Safe filenames
        assert InputSanitizer.sanitize_filename("document.pdf") == "document.pdf"
        assert InputSanitizer.sanitize_filename("my-file_123.txt") == "my-file_123.txt"

    def test_email_validation(self) -> None:
        """Test email format validation"""
        # Valid emails
        assert InputSanitizer.validate_email("user@example.com")
        assert InputSanitizer.validate_email("test.email+tag@domain.co.uk")

        # Invalid emails
        assert not InputSanitizer.validate_email("invalid-email")
        assert not InputSanitizer.validate_email("@domain.com")
        assert not InputSanitizer.validate_email("user@")
        assert not InputSanitizer.validate_email("")

    def test_username_validation(self) -> None:
        """Test username format validation"""
        # Valid usernames
        assert InputSanitizer.validate_username("user123")
        assert InputSanitizer.validate_username("test_user")
        assert InputSanitizer.validate_username("user-name")

        # Invalid usernames
        assert not InputSanitizer.validate_username("us")  # Too short
        assert not InputSanitizer.validate_username("user@domain")  # Invalid chars
        assert not InputSanitizer.validate_username("user space")  # Space
        assert not InputSanitizer.validate_username("")

    def test_password_strength_validation(self) -> None:
        """Test password strength validation"""
        # Strong password
        result = InputSanitizer.validate_password_strength("StrongPass123!")
        assert result["valid"]
        assert len(result["errors"]) == 0

        # Weak passwords
        weak_tests = [
            ("short", "at least 8 characters"),
            ("lowercase", "uppercase letter"),
            ("UPPERCASE", "lowercase letter"),
            ("NoNumbers!", "digit"),
            ("NoSpecial123", "special character"),
            ("password123!", "common patterns"),
        ]

        for password, expected_error in weak_tests:
            result = InputSanitizer.validate_password_strength(password)
            assert not result["valid"]
            assert any(expected_error in error for error in result["errors"])


class TestLoginFailureTracker:
    """Test login failure tracking and account locking"""

    def test_failure_tracking(self) -> None:
        """Test failure count tracking"""
        tracker = LoginFailureTracker(max_attempts=3, lockout_duration=60)
        identifier = "test_user"

        # Record failures
        assert not tracker.is_locked(identifier)

        tracker.record_failure(identifier)
        tracker.record_failure(identifier)
        assert not tracker.is_locked(identifier)

        tracker.record_failure(identifier)
        assert tracker.is_locked(identifier)

    def test_lockout_expiration(self) -> None:
        """Test lockout expiration"""
        tracker = LoginFailureTracker(max_attempts=2, lockout_duration=1)
        identifier = "test_user"

        # Lock account
        tracker.record_failure(identifier)
        tracker.record_failure(identifier)
        assert tracker.is_locked(identifier)

        # Wait for expiration
        time.sleep(1.1)
        assert not tracker.is_locked(identifier)

    def test_reset_attempts(self) -> None:
        """Test resetting failure attempts"""
        tracker = LoginFailureTracker(max_attempts=3)
        identifier = "test_user"

        # Record failures
        tracker.record_failure(identifier)
        tracker.record_failure(identifier)

        # Reset
        tracker.reset_attempts(identifier)
        assert not tracker.is_locked(identifier)

        # Should be able to fail again without immediate lock
        tracker.record_failure(identifier)
        tracker.record_failure(identifier)
        assert not tracker.is_locked(identifier)

    def test_lockout_time_remaining(self) -> None:
        """Test lockout time remaining calculation"""
        tracker = LoginFailureTracker(max_attempts=1, lockout_duration=60)
        identifier = "test_user"

        # Lock account
        tracker.record_failure(identifier)
        assert tracker.is_locked(identifier)

        # Check remaining time
        remaining = tracker.get_lockout_time_remaining(identifier)
        assert 50 < remaining <= 60  # Should be close to 60 seconds


class TestSecurityMiddleware:
    """Test security middleware functionality"""

    def test_ip_extraction(self) -> None:
        """Test IP address extraction from request headers"""
        from app.core.middleware import SecurityMiddleware

        middleware = SecurityMiddleware(None)

        # Mock request with forwarded headers
        request = Mock()
        request.headers = {"x-forwarded-for": "192.168.1.1, 10.0.0.1"}
        request.client = Mock()
        request.client.host = "127.0.0.1"

        ip = middleware._get_client_ip(request)
        assert ip == "192.168.1.1"

        # Test with real IP header
        request.headers = {"x-real-ip": "203.0.113.1"}
        ip = middleware._get_client_ip(request)
        assert ip == "203.0.113.1"

        # Test fallback to client host
        request.headers = {}
        ip = middleware._get_client_ip(request)
        assert ip == "127.0.0.1"

    def test_rate_limit_logic(self) -> None:
        """Test rate limiting logic"""
        from app.core.middleware import SecurityMiddleware

        middleware = SecurityMiddleware(None, rate_limit=3, rate_window=60)

        # Test normal requests
        assert not middleware._check_general_rate_limit("test_ip", time.time())
        assert not middleware._check_general_rate_limit("test_ip", time.time())
        assert not middleware._check_general_rate_limit("test_ip", time.time())

        # Fourth request should be rate limited
        assert middleware._check_general_rate_limit("test_ip", time.time())

    def test_login_rate_limit_logic(self) -> None:
        """Test login-specific rate limiting"""
        from app.core.middleware import SecurityMiddleware

        middleware = SecurityMiddleware(None, login_rate_limit=2, login_rate_window=60)

        # Test login attempts
        assert not middleware._check_login_rate_limit("test_ip", time.time())
        assert not middleware._check_login_rate_limit("test_ip", time.time())

        # Third attempt should be rate limited
        assert middleware._check_login_rate_limit("test_ip", time.time())
