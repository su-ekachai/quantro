"""
Security middleware for rate limiting and protection
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

from fastapi import Request, Response, status
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.security import security_manager


class SecurityMiddleware(BaseHTTPMiddleware):
    """Security middleware with rate limiting and headers"""

    def __init__(
        self,
        app: Any,
        rate_limit: int = 100,
        rate_window: int = 60,
        login_rate_limit: int = 5,
        login_rate_window: int = 300,
    ) -> None:
        super().__init__(app)
        self.rate_limit = rate_limit
        self.rate_window = rate_window
        self.login_rate_limit = login_rate_limit
        self.login_rate_window = login_rate_window

        # In-memory rate limiting storage (use Redis in production)
        self.request_counts: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"count": 0, "start_time": time.time()}
        )
        self.login_attempts: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"count": 0, "start_time": time.time()}
        )

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        """Process request with security checks"""
        client_ip = self._get_client_ip(request)
        hashed_ip = security_manager.hash_ip_address(client_ip)

        # Check rate limits
        if self._is_rate_limited(hashed_ip, request.url.path):
            logger.warning(
                "Rate limit exceeded",
                extra={
                    "ip_hash": hashed_ip,
                    "path": request.url.path,
                    "user_agent": request.headers.get("user-agent", "unknown"),
                },
            )
            return Response(
                content="Rate limit exceeded. Please try again later.",
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            )

        # Process request
        response = await call_next(request)

        # Add security headers
        self._add_security_headers(response)

        # Log security events
        if request.url.path.startswith("/api/auth/"):
            self._log_auth_event(request, response, hashed_ip)

        return response

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request"""
        # Check for forwarded headers (reverse proxy)
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip

        return request.client.host if request.client else "unknown"

    def _is_rate_limited(self, hashed_ip: str, path: str) -> bool:
        """Check if request should be rate limited"""
        current_time = time.time()

        # Special handling for login endpoints
        if "/auth/login" in path:
            return self._check_login_rate_limit(hashed_ip, current_time)

        # General rate limiting
        return self._check_general_rate_limit(hashed_ip, current_time)

    def _check_general_rate_limit(self, hashed_ip: str, current_time: float) -> bool:
        """Check general API rate limit"""
        request_data = self.request_counts[hashed_ip]

        # Reset counter if window expired
        if current_time - request_data["start_time"] > self.rate_window:
            request_data["count"] = 1
            request_data["start_time"] = current_time
            return False

        # Check if limit exceeded
        if request_data["count"] >= self.rate_limit:
            return True

        request_data["count"] += 1
        return False

    def _check_login_rate_limit(self, hashed_ip: str, current_time: float) -> bool:
        """Check login-specific rate limit"""
        login_data = self.login_attempts[hashed_ip]

        # Reset counter if window expired
        if current_time - login_data["start_time"] > self.login_rate_window:
            login_data["count"] = 1
            login_data["start_time"] = current_time
            return False

        # Check if limit exceeded
        if login_data["count"] >= self.login_rate_limit:
            return True

        login_data["count"] += 1
        return False

    def _add_security_headers(self, response: Response) -> None:
        """Add security headers to response"""
        # Prevent clickjacking
        response.headers["X-Frame-Options"] = "DENY"

        # Prevent MIME type sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"

        # XSS protection
        response.headers["X-XSS-Protection"] = "1; mode=block"

        # HSTS (HTTP Strict Transport Security)
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains; preload"
        )

        # Content Security Policy
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' "
            "https://cdn.jsdelivr.net https://cdn.tailwindcss.com; "
            "style-src 'self' 'unsafe-inline' https://cdn.tailwindcss.com; "
            "img-src 'self' data: https:; "
            "font-src 'self' https:; "
            "connect-src 'self'; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "form-action 'self'"
        )

        # Referrer policy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Permissions policy
        response.headers["Permissions-Policy"] = (
            "geolocation=(), microphone=(), camera=(), "
            "payment=(), usb=(), magnetometer=(), gyroscope=()"
        )

    def _log_auth_event(
        self, request: Request, response: Response, hashed_ip: str
    ) -> None:
        """Log authentication-related events"""
        logger.info(
            "Authentication event",
            extra={
                "ip_hash": hashed_ip,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "user_agent": request.headers.get("user-agent", "unknown"),
                "timestamp": time.time(),
            },
        )


class LoginFailureTracker:
    """Track and manage login failures for account locking"""

    def __init__(self, max_attempts: int = 5, lockout_duration: int = 900) -> None:
        self.max_attempts = max_attempts
        self.lockout_duration = lockout_duration  # 15 minutes
        self.failed_attempts: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"count": 0, "first_attempt": None, "locked_until": None}
        )

    def record_failure(self, identifier: str) -> None:
        """Record a failed login attempt"""
        current_time = time.time()
        attempt_data = self.failed_attempts[identifier]

        if attempt_data["first_attempt"] is None:
            attempt_data["first_attempt"] = current_time

        attempt_data["count"] += 1

        # Lock account if max attempts reached
        if attempt_data["count"] >= self.max_attempts:
            attempt_data["locked_until"] = current_time + self.lockout_duration
            logger.warning(
                "Account locked due to failed login attempts",
                extra={"identifier": identifier, "attempts": attempt_data["count"]},
            )

    def is_locked(self, identifier: str) -> bool:
        """Check if account is currently locked"""
        attempt_data = self.failed_attempts[identifier]

        if attempt_data["locked_until"] is None:
            return False

        current_time = time.time()
        if current_time > attempt_data["locked_until"]:
            # Lock expired, reset attempts
            self.reset_attempts(identifier)
            return False

        return True

    def reset_attempts(self, identifier: str) -> None:
        """Reset failed attempts for identifier"""
        if identifier in self.failed_attempts:
            del self.failed_attempts[identifier]

    def get_lockout_time_remaining(self, identifier: str) -> int:
        """Get remaining lockout time in seconds"""
        attempt_data = self.failed_attempts[identifier]

        if attempt_data["locked_until"] is None:
            return 0

        current_time = time.time()
        remaining = attempt_data["locked_until"] - current_time
        return max(0, int(remaining))


# Global login failure tracker
login_failure_tracker = LoginFailureTracker()
