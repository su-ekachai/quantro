"""
CCXT provider for cryptocurrency market data
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import ccxt.async_support as ccxt
from loguru import logger

from .base import (
    OHLCV,
    AssetInfo,
    DataProviderError,
    IDataProvider,
    RateLimitError,
)


class CCXTProvider(IDataProvider):
    """CCXT-based cryptocurrency data provider"""

    def __init__(
        self,
        exchange_name: str = "binance",
        api_key: str | None = None,
        api_secret: str | None = None,
        sandbox: bool = False,
        rate_limit_requests: int = 100,
        rate_limit_window: float = 60.0,
    ):
        super().__init__(
            f"ccxt_{exchange_name}", rate_limit_requests, rate_limit_window
        )

        self.exchange_name = exchange_name
        self.sandbox = sandbox

        # Initialize exchange
        exchange_class = getattr(ccxt, exchange_name)
        self.exchange = exchange_class(
            {
                "apiKey": api_key,
                "secret": api_secret,
                "sandbox": sandbox,
                "enableRateLimit": True,
                "rateLimit": int(
                    rate_limit_window * 1000 / rate_limit_requests
                ),  # ms per request
            }
        )

        # Timeframe mapping
        self.timeframe_map = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "1h": "1h",
            "4h": "4h",
            "1d": "1d",
        }

    async def __aenter__(self) -> CCXTProvider:
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit"""
        await self.close()

    async def close(self) -> None:
        """Close exchange connection"""
        if self.exchange:
            await self.exchange.close()

    def _convert_to_ohlcv(self, ccxt_data: list[Any]) -> OHLCV:
        """Convert CCXT OHLCV data to internal format"""
        timestamp, open_price, high, low, close, volume = ccxt_data

        return OHLCV(
            timestamp=datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc),
            open=Decimal(str(open_price)),
            high=Decimal(str(high)),
            low=Decimal(str(low)),
            close=Decimal(str(close)),
            volume=Decimal(str(volume)),
        )

    async def fetch_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        limit: int | None = None,
    ) -> list[OHLCV]:
        """Fetch historical OHLCV data from exchange"""

        if timeframe not in self.timeframe_map:
            raise DataProviderError(f"Unsupported timeframe: {timeframe}")

        ccxt_timeframe = self.timeframe_map[timeframe]

        async def _fetch() -> list[OHLCV]:
            try:
                # Convert datetime to milliseconds
                since = int(start.timestamp() * 1000)

                # Fetch data
                ohlcv_data = await self.exchange.fetch_ohlcv(
                    symbol, ccxt_timeframe, since=since, limit=limit
                )

                # Filter by end date and convert to internal format
                result = []
                for data_point in ohlcv_data:
                    ohlcv = self._convert_to_ohlcv(data_point)
                    if ohlcv.timestamp <= end:
                        result.append(ohlcv)
                    else:
                        break

                logger.info(
                    f"Fetched {len(result)} OHLCV records for {symbol} "
                    f"from {self.exchange_name}"
                )
                return result

            except ccxt.RateLimitExceeded as e:
                raise RateLimitError(f"Rate limit exceeded: {e}") from e
            except ccxt.NetworkError as e:
                raise DataProviderError(f"Network error: {e}") from e
            except ccxt.ExchangeError as e:
                raise DataProviderError(f"Exchange error: {e}") from e
            except Exception as e:
                raise DataProviderError(f"Unexpected error: {e}") from e

        return await self._make_request_with_retry(_fetch)

    async def subscribe_real_time(
        self, symbol: str, callback: Callable[[OHLCV], None]
    ) -> None:
        """Subscribe to real-time market data updates"""

        if not self.exchange.has["watchTicker"]:
            raise DataProviderError(
                f"Real-time data not supported by {self.exchange_name}"
            )

        async def _watch() -> None:
            try:
                while True:
                    ticker = await self.exchange.watch_ticker(symbol)

                    # Convert ticker to OHLCV format
                    ohlcv = OHLCV(
                        timestamp=datetime.fromtimestamp(
                            ticker["timestamp"] / 1000, tz=timezone.utc
                        ),
                        open=Decimal(str(ticker["open"])),
                        high=Decimal(str(ticker["high"])),
                        low=Decimal(str(ticker["low"])),
                        close=Decimal(str(ticker["close"])),
                        volume=Decimal(str(ticker["baseVolume"] or 0)),
                    )

                    callback(ohlcv)

            except ccxt.NetworkError as e:
                logger.error(f"Network error in real-time subscription: {e}")
                raise DataProviderError(f"Network error: {e}") from e
            except Exception as e:
                logger.error(f"Error in real-time subscription: {e}")
                raise DataProviderError(f"Subscription error: {e}") from e

        # Start watching in background task
        asyncio.create_task(_watch())

    async def get_asset_info(self, symbol: str) -> AssetInfo:
        """Get asset information from exchange"""

        async def _fetch() -> AssetInfo:
            try:
                # Load markets if not already loaded
                if not self.exchange.markets:
                    await self.exchange.load_markets()

                if symbol not in self.exchange.markets:
                    raise DataProviderError(
                        f"Symbol {symbol} not found on {self.exchange_name}"
                    )

                market = self.exchange.markets[symbol]

                return AssetInfo(
                    symbol=symbol,
                    name=market.get("id", symbol),
                    asset_class="crypto",
                    exchange=self.exchange_name,
                    base_currency=market.get("base"),
                    quote_currency=market.get("quote"),
                    min_order_size=Decimal(
                        str(market.get("limits", {}).get("amount", {}).get("min", 0))
                    ),
                    max_order_size=Decimal(
                        str(market.get("limits", {}).get("amount", {}).get("max", 0))
                    )
                    if market.get("limits", {}).get("amount", {}).get("max")
                    else None,
                    price_precision=market.get("precision", {}).get("price"),
                    quantity_precision=market.get("precision", {}).get("amount"),
                    is_active=market.get("active", True),
                )

            except ccxt.NetworkError as e:
                raise DataProviderError(f"Network error: {e}") from e
            except ccxt.ExchangeError as e:
                raise DataProviderError(f"Exchange error: {e}") from e
            except Exception as e:
                raise DataProviderError(f"Unexpected error: {e}") from e

        return await self._make_request_with_retry(_fetch)

    async def get_supported_symbols(self) -> list[str]:
        """Get list of supported trading symbols"""

        async def _fetch() -> list[str]:
            try:
                # Load markets if not already loaded
                if not self.exchange.markets:
                    await self.exchange.load_markets()

                # Return active symbols only
                symbols = [
                    symbol
                    for symbol, market in self.exchange.markets.items()
                    if market.get("active", True)
                ]

                logger.info(
                    f"Found {len(symbols)} active symbols on {self.exchange_name}"
                )
                return symbols

            except ccxt.NetworkError as e:
                raise DataProviderError(f"Network error: {e}") from e
            except ccxt.ExchangeError as e:
                raise DataProviderError(f"Exchange error: {e}") from e
            except Exception as e:
                raise DataProviderError(f"Unexpected error: {e}") from e

        return await self._make_request_with_retry(_fetch)

    def get_supported_timeframes(self) -> list[str]:
        """Get list of supported timeframes"""
        if hasattr(self.exchange, "timeframes"):
            # Return intersection of exchange timeframes and our supported ones
            exchange_timeframes = set(self.exchange.timeframes.keys())
            supported_timeframes = set(self.timeframe_map.keys())
            return list(exchange_timeframes.intersection(supported_timeframes))
        else:
            # Return default supported timeframes
            return list(self.timeframe_map.keys())


# Factory function for common exchanges
def create_binance_provider(
    api_key: str | None = None, api_secret: str | None = None, sandbox: bool = False
) -> CCXTProvider:
    """Create Binance provider instance"""
    return CCXTProvider(
        exchange_name="binance",
        api_key=api_key,
        api_secret=api_secret,
        sandbox=sandbox,
        rate_limit_requests=1200,  # Binance allows 1200 requests per minute
        rate_limit_window=60.0,
    )


def create_coinbase_provider(
    api_key: str | None = None, api_secret: str | None = None, sandbox: bool = False
) -> CCXTProvider:
    """Create Coinbase Pro provider instance"""
    return CCXTProvider(
        exchange_name="coinbasepro",
        api_key=api_key,
        api_secret=api_secret,
        sandbox=sandbox,
        rate_limit_requests=10,  # Coinbase Pro has strict rate limits
        rate_limit_window=1.0,
    )
