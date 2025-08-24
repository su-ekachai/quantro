"""
Base data provider interface and common models.

This module defines the abstract base class for all data providers
in the Quantro trading platform, along with common data models.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class DataProviderError(Exception):
    """Base exception for data provider errors."""

    pass


class RateLimitError(DataProviderError):
    """Exception raised when rate limits are exceeded."""

    pass


class OHLCV(BaseModel):
    """OHLCV data model."""

    timestamp: datetime = Field(..., description="Data timestamp")
    open: float = Field(..., description="Opening price")
    high: float = Field(..., description="High price")
    low: float = Field(..., description="Low price")
    close: float = Field(..., description="Closing price")
    volume: float = Field(..., description="Trading volume")


class AssetInfo(BaseModel):
    """Asset information model."""

    symbol: str = Field(..., description="Trading symbol")
    name: str = Field(..., description="Asset name")
    asset_type: str = Field(..., description="Asset type (crypto, stock, etc.)")
    exchange: str = Field(..., description="Exchange name")
    base_currency: str | None = Field(default=None, description="Base currency")
    quote_currency: str | None = Field(default=None, description="Quote currency")


class IDataProvider(ABC):
    """Abstract interface for data providers."""

    def __init__(
        self, name: str, rate_limit_requests: int = 100, rate_limit_window: float = 60.0
    ) -> None:
        """
        Initialize data provider.

        Args:
            name: Provider name
            rate_limit_requests: Maximum requests per window
            rate_limit_window: Rate limit window in seconds
        """
        self.name = name
        self.rate_limit_requests = rate_limit_requests
        self.rate_limit_window = rate_limit_window
        self._request_times: list[float] = []
        self._request_lock = asyncio.Lock()

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
        Fetch historical OHLCV data.

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe (1m, 5m, 1h, 1d, etc.)
            start: Start date
            end: End date
            limit: Maximum number of data points to return (optional)

        Returns:
            List of OHLCV data points

        Raises:
            ValueError: If parameters are invalid
            DataProviderError: If data fetching fails
        """
        pass

    @abstractmethod
    async def fetch_current_price(self, symbol: str) -> float:
        """
        Fetch current price for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Current price

        Raises:
            ValueError: If symbol is invalid
            DataProviderError: If price fetching fails
        """
        pass

    @abstractmethod
    async def get_asset_info(self, symbol: str) -> AssetInfo:
        """
        Get asset information.

        Args:
            symbol: Trading symbol

        Returns:
            Asset information

        Raises:
            ValueError: If symbol is invalid
            DataProviderError: If asset info fetching fails
        """
        pass

    async def _make_request_with_retry(
        self,
        request_func: Any,
        *args: Any,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> Any:
        """
        Make a request with rate limiting and retry logic.

        Args:
            request_func: Function to call
            *args: Positional arguments for the function
            max_retries: Maximum number of retries
            **kwargs: Keyword arguments for the function

        Returns:
            Result of the request function

        Raises:
            DataProviderError: If all retries fail
        """
        async with self._request_lock:
            # Rate limiting logic
            now = asyncio.get_event_loop().time()

            # Remove old requests outside the window
            self._request_times = [
                t for t in self._request_times if now - t < self.rate_limit_window
            ]

            # Check if we're at the rate limit
            if len(self._request_times) >= self.rate_limit_requests:
                sleep_time = self.rate_limit_window - (now - self._request_times[0])
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

            # Record this request
            self._request_times.append(now)

        # Retry logic
        last_exception = None
        for attempt in range(max_retries + 1):
            try:
                return await request_func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < max_retries:
                    wait_time = 2**attempt  # Exponential backoff
                    await asyncio.sleep(wait_time)
                else:
                    break

        raise DataProviderError(
            f"Request failed after {max_retries + 1} attempts: {last_exception}"
        )
