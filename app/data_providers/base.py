"""
Abstract base interface for data providers
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from loguru import logger


class DataProviderError(Exception):
    """Base exception for data provider errors"""

    pass


class RateLimitError(DataProviderError):
    """Exception raised when rate limit is exceeded"""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class HealthStatus(Enum):
    """Health status enumeration"""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class OHLCV:
    """OHLCV data structure"""

    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    trades_count: int | None = None
    vwap: Decimal | None = None


@dataclass
class AssetInfo:
    """Asset information structure"""

    symbol: str
    name: str
    asset_class: str
    exchange: str
    base_currency: str | None = None
    quote_currency: str | None = None
    min_order_size: Decimal | None = None
    max_order_size: Decimal | None = None
    price_precision: int | None = None
    quantity_precision: int | None = None
    is_active: bool = True


@dataclass
class HealthCheck:
    """Health check result"""

    status: HealthStatus
    message: str
    response_time_ms: float
    last_successful_request: datetime | None = None
    error_count: int = 0


class RateLimiter:
    """Simple rate limiter with exponential backoff"""

    def __init__(self, max_requests: int, time_window: float):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests: list[float] = []
        self.backoff_until: float = 0
        self.consecutive_failures = 0

    async def acquire(self) -> None:
        """Acquire rate limit permission"""
        current_time = time.time()

        # Check if we're in backoff period
        if current_time < self.backoff_until:
            wait_time = self.backoff_until - current_time
            logger.warning(f"Rate limiter in backoff, waiting {wait_time:.2f}s")
            await asyncio.sleep(wait_time)

        # Clean old requests outside time window
        self.requests = [
            req_time
            for req_time in self.requests
            if current_time - req_time < self.time_window
        ]

        # Check if we can make a request
        if len(self.requests) >= self.max_requests:
            wait_time = self.time_window - (current_time - self.requests[0])
            logger.warning(f"Rate limit reached, waiting {wait_time:.2f}s")
            await asyncio.sleep(wait_time)
            # Clean again after waiting
            current_time = time.time()
            self.requests = [
                req_time
                for req_time in self.requests
                if current_time - req_time < self.time_window
            ]

        # Record this request
        self.requests.append(current_time)

    def on_success(self) -> None:
        """Reset backoff on successful request"""
        self.consecutive_failures = 0
        self.backoff_until = 0

    def on_failure(self) -> None:
        """Apply exponential backoff on failure"""
        self.consecutive_failures += 1
        backoff_time = min(2**self.consecutive_failures, 300)  # Max 5 minutes
        self.backoff_until = time.time() + backoff_time
        logger.warning(f"Request failed, backing off for {backoff_time}s")


class IDataProvider(ABC):
    """Abstract interface for data providers"""

    def __init__(
        self, name: str, rate_limit_requests: int = 100, rate_limit_window: float = 60.0
    ):
        self.name = name
        self.rate_limiter = RateLimiter(rate_limit_requests, rate_limit_window)
        self.last_health_check: HealthCheck | None = None
        self._is_healthy = True
        self._priority: int = 0

    @abstractmethod
    async def fetch_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        limit: int | None = None,
    ) -> list[OHLCV]:
        """
        Fetch historical OHLCV data for a symbol

        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT', 'AAPL', 'PTT.BK')
            timeframe: Time interval ('1m', '5m', '15m', '1h', '4h', '1d')
            start: Start datetime
            end: End datetime
            limit: Maximum number of records to return

        Returns:
            List of OHLCV data points

        Raises:
            DataProviderError: On API errors
            RateLimitError: When rate limit is exceeded
        """
        pass

    @abstractmethod
    async def subscribe_real_time(
        self, symbol: str, callback: Callable[[OHLCV], None]
    ) -> None:
        """
        Subscribe to real-time market data updates

        Args:
            symbol: Trading symbol
            callback: Function to call with new data

        Raises:
            DataProviderError: On subscription errors
        """
        pass

    @abstractmethod
    async def get_asset_info(self, symbol: str) -> AssetInfo:
        """
        Get asset/instrument information

        Args:
            symbol: Trading symbol

        Returns:
            Asset information

        Raises:
            DataProviderError: On API errors
        """
        pass

    @abstractmethod
    async def get_supported_symbols(self) -> list[str]:
        """
        Get list of supported trading symbols

        Returns:
            List of supported symbols

        Raises:
            DataProviderError: On API errors
        """
        pass

    @abstractmethod
    def get_supported_timeframes(self) -> list[str]:
        """
        Get list of supported timeframes

        Returns:
            List of supported timeframes
        """
        pass

    async def health_check(self) -> HealthCheck:
        """
        Perform health check on the data provider

        Returns:
            Health check result
        """
        start_time = time.time()

        try:
            # Try to get supported symbols as a basic health check
            await self.get_supported_symbols()

            response_time = (time.time() - start_time) * 1000
            self.last_health_check = HealthCheck(
                status=HealthStatus.HEALTHY,
                message="Provider is healthy",
                response_time_ms=response_time,
                last_successful_request=datetime.now(),
                error_count=0,
            )
            self._is_healthy = True

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            error_count = getattr(self.last_health_check, "error_count", 0) + 1

            status = (
                HealthStatus.DEGRADED if error_count < 5 else HealthStatus.UNHEALTHY
            )

            self.last_health_check = HealthCheck(
                status=status,
                message=f"Health check failed: {str(e)}",
                response_time_ms=response_time,
                last_successful_request=getattr(
                    self.last_health_check, "last_successful_request", None
                ),
                error_count=error_count,
            )
            self._is_healthy = False

            logger.error(f"Health check failed for {self.name}: {e}")

        return self.last_health_check

    @property
    def is_healthy(self) -> bool:
        """Check if provider is currently healthy"""
        return self._is_healthy

    async def _make_request_with_retry(
        self, request_func: Callable[[], Any], max_retries: int = 3
    ) -> Any:
        """
        Make a request with rate limiting and retry logic

        Args:
            request_func: Function that makes the actual request
            max_retries: Maximum number of retry attempts

        Returns:
            Request result

        Raises:
            DataProviderError: On persistent failures
        """
        for attempt in range(max_retries + 1):
            try:
                await self.rate_limiter.acquire()
                result = await request_func()
                self.rate_limiter.on_success()
                return result

            except RateLimitError as e:
                if e.retry_after:
                    logger.warning(f"Rate limited, waiting {e.retry_after}s")
                    await asyncio.sleep(e.retry_after)
                self.rate_limiter.on_failure()

                if attempt == max_retries:
                    raise

            except Exception as e:
                self.rate_limiter.on_failure()

                if attempt == max_retries:
                    raise DataProviderError(
                        f"Request failed after {max_retries} retries: {e}"
                    ) from e

                # Exponential backoff
                wait_time = 2**attempt
                logger.warning(
                    f"Request failed (attempt {attempt + 1}), "
                    f"retrying in {wait_time}s: {e}"
                )
                await asyncio.sleep(wait_time)
