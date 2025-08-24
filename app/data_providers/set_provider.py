"""
SET Smart Portal API provider for Thai stock market data
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import httpx
from loguru import logger

from .base import (
    OHLCV,
    AssetInfo,
    DataProviderError,
    IDataProvider,
    RateLimitError,
)


class SETProvider(IDataProvider):
    """SET Smart Portal API provider for Thai stock data"""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.settrade.com",
        rate_limit_requests: int = 100,
        rate_limit_window: float = 60.0,
    ):
        super().__init__("set_smart_portal", rate_limit_requests, rate_limit_window)

        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

        # HTTP client with timeout and retry configuration
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )

        # Timeframe mapping (SET API specific)
        self.timeframe_map = {
            "1m": "1",
            "5m": "5",
            "15m": "15",
            "1h": "60",
            "4h": "240",
            "1d": "D",
        }

        # Common Thai stock symbols for testing
        self.known_symbols = [
            "PTT",
            "CPALL",
            "KBANK",
            "SCB",
            "BBL",
            "AOT",
            "ADVANC",
            "TRUE",
            "SCC",
            "TCAP",
            "TISCO",
            "CPN",
            "HMPRO",
            "MINT",
            "INTUCH",
        ]

    async def __aenter__(self) -> SETProvider:
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit"""
        await self.close()

    async def close(self) -> None:
        """Close HTTP client"""
        if self.client:
            await self.client.aclose()

    def _get_headers(self) -> dict[str, str]:
        """Get request headers with API key if available"""
        headers = {
            "User-Agent": "Quantro/1.0",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        return headers

    def _convert_to_ohlcv(self, set_data: dict[str, Any]) -> OHLCV:
        """Convert SET API data to internal OHLCV format"""

        # SET API response format may vary, this is a generic implementation
        # Adjust based on actual API response structure
        timestamp_str = set_data.get("datetime") or set_data.get("date")
        if isinstance(timestamp_str, str):
            # Parse timestamp (adjust format based on actual API)
            timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        elif isinstance(timestamp_str, int | float):
            timestamp = datetime.fromtimestamp(timestamp_str / 1000, tz=timezone.utc)
        else:
            # Fallback to current time if timestamp is invalid
            timestamp = datetime.now(timezone.utc)

        return OHLCV(
            timestamp=timestamp,
            open=Decimal(str(set_data.get("open", 0))),
            high=Decimal(str(set_data.get("high", 0))),
            low=Decimal(str(set_data.get("low", 0))),
            close=Decimal(str(set_data.get("close", 0))),
            volume=Decimal(str(set_data.get("volume", 0))),
            trades_count=set_data.get("trades"),
        )

    async def fetch_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        limit: int | None = None,
    ) -> list[OHLCV]:
        """Fetch historical OHLCV data from SET Smart Portal API"""

        if timeframe not in self.timeframe_map:
            raise DataProviderError(f"Unsupported timeframe: {timeframe}")

        set_timeframe = self.timeframe_map[timeframe]

        async def _fetch() -> list[OHLCV]:
            try:
                # Prepare request parameters
                params: dict[str, Any] = {
                    "symbol": symbol.upper(),
                    "timeframe": set_timeframe,
                    "from": start.strftime("%Y-%m-%d"),
                    "to": end.strftime("%Y-%m-%d"),
                }

                if limit:
                    params["limit"] = str(limit)

                # Make API request
                url = f"{self.base_url}/api/market/historical"
                response = await self.client.get(
                    url, params=params, headers=self._get_headers()
                )

                # Handle rate limiting
                if response.status_code == 429:
                    retry_after = float(response.headers.get("Retry-After", 60))
                    raise RateLimitError("Rate limit exceeded", retry_after)

                # Handle other HTTP errors
                if response.status_code != 200:
                    raise DataProviderError(
                        f"API request failed: {response.status_code} - {response.text}"
                    )

                data = response.json()

                # Handle API error responses
                if not data.get("success", True):
                    error_msg = data.get("message", "Unknown API error")
                    raise DataProviderError(f"SET API error: {error_msg}")

                # Convert to internal format
                ohlcv_data = data.get("data", [])
                result = [self._convert_to_ohlcv(item) for item in ohlcv_data]

                logger.info(
                    f"Fetched {len(result)} OHLCV records for {symbol} from SET API"
                )
                return result

            except httpx.TimeoutException:
                raise DataProviderError("Request timeout") from None
            except httpx.NetworkError as e:
                raise DataProviderError(f"Network error: {e}") from e
            except Exception as e:
                if isinstance(e, DataProviderError | RateLimitError):
                    raise
                raise DataProviderError(f"Unexpected error: {e}") from e

        return await self._make_request_with_retry(_fetch)

    async def subscribe_real_time(
        self, symbol: str, callback: Callable[[OHLCV], None]
    ) -> None:
        """Subscribe to real-time market data updates"""

        # SET Smart Portal may not support WebSocket streaming in free tier
        # Implement polling-based real-time updates

        async def _poll() -> None:
            try:
                while True:
                    # Fetch latest data point
                    end_time = datetime.now(timezone.utc)
                    start_time = end_time.replace(
                        hour=0, minute=0, second=0, microsecond=0
                    )

                    data = await self.fetch_historical_data(
                        symbol, "1m", start_time, end_time, limit=1
                    )

                    if data:
                        callback(data[-1])  # Send latest data point

                    # Wait before next poll (adjust based on requirements)
                    await asyncio.sleep(60)  # Poll every minute

            except Exception as e:
                logger.error(f"Error in real-time polling: {e}")
                raise DataProviderError(f"Real-time subscription error: {e}") from e

        # Start polling in background task
        asyncio.create_task(_poll())

    async def get_asset_info(self, symbol: str) -> AssetInfo:
        """Get asset information from SET API"""

        async def _fetch() -> AssetInfo:
            try:
                # Make API request for symbol info
                url = f"{self.base_url}/api/market/symbol/{symbol.upper()}"
                response = await self.client.get(url, headers=self._get_headers())

                if response.status_code == 429:
                    retry_after = float(response.headers.get("Retry-After", 60))
                    raise RateLimitError("Rate limit exceeded", retry_after)

                if response.status_code == 404:
                    raise DataProviderError(f"Symbol {symbol} not found")

                if response.status_code != 200:
                    raise DataProviderError(
                        f"API request failed: {response.status_code}"
                    )

                data = response.json()

                if not data.get("success", True):
                    error_msg = data.get("message", "Unknown API error")
                    raise DataProviderError(f"SET API error: {error_msg}")

                symbol_data = data.get("data", {})

                return AssetInfo(
                    symbol=symbol.upper(),
                    name=symbol_data.get("name_th", symbol_data.get("name_en", symbol)),
                    asset_type="stock",
                    exchange="SET",
                    base_currency="THB",
                    quote_currency="THB",
                )

            except httpx.TimeoutException:
                raise DataProviderError("Request timeout") from None
            except httpx.NetworkError as e:
                raise DataProviderError(f"Network error: {e}") from e
            except Exception as e:
                if isinstance(e, DataProviderError | RateLimitError):
                    raise
                raise DataProviderError(f"Unexpected error: {e}") from e

        return await self._make_request_with_retry(_fetch)

    async def fetch_current_price(self, symbol: str) -> float:
        """Fetch current price for a symbol from SET API."""

        async def _fetch() -> float:
            try:
                # Make API request for current price
                url = f"{self.base_url}/api/market/price/{symbol.upper()}"
                response = await self.client.get(url, headers=self._get_headers())

                if response.status_code == 429:
                    retry_after = float(response.headers.get("Retry-After", 60))
                    raise RateLimitError("Rate limit exceeded", retry_after)

                if response.status_code == 404:
                    raise DataProviderError(f"Symbol {symbol} not found")

                if response.status_code != 200:
                    raise DataProviderError(
                        f"API request failed: {response.status_code}"
                    )

                data = response.json()

                if not data.get("success", True):
                    error_msg = data.get("message", "Unknown API error")
                    raise DataProviderError(f"SET API error: {error_msg}")

                price_data = data.get("data", {})
                current_price = price_data.get("last_price")

                if current_price is None:
                    raise DataProviderError("No price data available")

                return float(current_price)

            except httpx.TimeoutException:
                raise DataProviderError("Request timeout") from None
            except httpx.NetworkError as e:
                raise DataProviderError(f"Network error: {e}") from e
            except Exception as e:
                if isinstance(e, DataProviderError | RateLimitError):
                    raise
                raise DataProviderError(f"Unexpected error: {e}") from e

        return await self._make_request_with_retry(_fetch)

    async def get_supported_symbols(self) -> list[str]:
        """Get list of supported trading symbols"""

        async def _fetch() -> list[str]:
            try:
                # Try to fetch symbol list from API
                url = f"{self.base_url}/api/market/symbols"
                response = await self.client.get(url, headers=self._get_headers())

                if response.status_code == 429:
                    retry_after = float(response.headers.get("Retry-After", 60))
                    raise RateLimitError("Rate limit exceeded", retry_after)

                if response.status_code == 200:
                    data = response.json()
                    if data.get("success", True):
                        symbols = data.get("data", [])
                        if isinstance(symbols, list):
                            logger.info(f"Found {len(symbols)} symbols from SET API")
                            return symbols

                # Fallback to known symbols if API doesn't provide symbol list
                logger.warning("Using fallback symbol list for SET provider")
                return self.known_symbols

            except Exception as e:
                logger.warning(f"Failed to fetch symbols from SET API: {e}")
                # Return known symbols as fallback
                return self.known_symbols

        return await self._make_request_with_retry(_fetch)

    def get_supported_timeframes(self) -> list[str]:
        """Get list of supported timeframes"""
        return list(self.timeframe_map.keys())


# Factory function for SET provider
def create_set_provider(api_key: str | None = None) -> SETProvider:
    """Create SET Smart Portal provider instance"""
    return SETProvider(
        api_key=api_key,
        rate_limit_requests=100,  # Conservative rate limit for free tier
        rate_limit_window=60.0,
    )
