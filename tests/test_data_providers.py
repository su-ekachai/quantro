"""
Tests for data provider integration framework
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.data_providers.base import (
    OHLCV,
    AssetInfo,
    DataProviderError,
    HealthCheck,
    HealthStatus,
    IDataProvider,
    RateLimiter,
    RateLimitError,
)
from app.data_providers.ccxt_provider import CCXTProvider
from app.data_providers.set_provider import SETProvider


class TestRateLimiter:
    """Test rate limiter functionality"""

    @pytest.mark.asyncio
    async def test_rate_limiter_allows_requests_within_limit(self) -> None:
        """Test that rate limiter allows requests within the limit"""
        limiter = RateLimiter(max_requests=5, time_window=1.0)

        # Should allow 5 requests without delay
        for _ in range(5):
            await limiter.acquire()

        assert len(limiter.requests) == 5

    @pytest.mark.asyncio
    async def test_rate_limiter_blocks_excess_requests(self) -> None:
        """Test that rate limiter blocks requests exceeding the limit"""
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            limiter = RateLimiter(max_requests=2, time_window=1.0)

            # First 2 requests should be immediate
            await limiter.acquire()
            await limiter.acquire()

            # Third request should trigger sleep
            await limiter.acquire()

            # Should have called sleep at least once
            assert mock_sleep.call_count >= 1

    def test_rate_limiter_backoff_on_failure(self) -> None:
        """Test exponential backoff on failures"""
        limiter = RateLimiter(max_requests=10, time_window=60.0)

        # Simulate failures
        limiter.on_failure()
        assert limiter.consecutive_failures == 1
        assert limiter.backoff_until > 0

        limiter.on_failure()
        assert limiter.consecutive_failures == 2

        # Success should reset backoff
        limiter.on_success()
        assert limiter.consecutive_failures == 0
        assert limiter.backoff_until == 0


class TestCCXTProvider:
    """Test CCXT provider functionality"""

    @pytest.fixture
    def mock_exchange(self) -> MagicMock:
        """Create mock CCXT exchange"""
        exchange = MagicMock()
        exchange.markets = {
            "BTC/USDT": {
                "id": "BTCUSDT",
                "base": "BTC",
                "quote": "USDT",
                "active": True,
                "limits": {
                    "amount": {"min": 0.001, "max": 1000},
                },
                "precision": {"price": 2, "amount": 6},
            }
        }
        exchange.timeframes = {"1m": "1m", "1h": "1h", "1d": "1d"}
        exchange.has = {"watchTicker": True}
        return exchange

    @pytest.fixture
    def ccxt_provider(self, mock_exchange: MagicMock) -> CCXTProvider:
        """Create CCXT provider with mocked exchange"""
        with patch("app.data_providers.ccxt_provider.ccxt.binance") as mock_binance:
            mock_binance.return_value = mock_exchange
            provider = CCXTProvider(exchange_name="binance")
            provider.exchange = mock_exchange
            return provider

    @pytest.mark.asyncio
    async def test_fetch_historical_data_success(
        self, ccxt_provider: CCXTProvider, mock_exchange: MagicMock
    ) -> None:
        """Test successful historical data fetch"""
        # Mock OHLCV data
        mock_ohlcv = [
            [1640995200000, 47000.0, 47500.0, 46500.0, 47200.0, 1.5],  # 2022-01-01
            [1641081600000, 47200.0, 47800.0, 46800.0, 47600.0, 2.1],  # 2022-01-02
        ]
        mock_exchange.fetch_ohlcv = AsyncMock(return_value=mock_ohlcv)

        start = datetime(2022, 1, 1, tzinfo=timezone.utc)
        end = datetime(2022, 1, 3, tzinfo=timezone.utc)

        result = await ccxt_provider.fetch_historical_data("BTC/USDT", "1d", start, end)

        assert len(result) == 2
        assert isinstance(result[0], OHLCV)
        assert result[0].open == Decimal("47000.0")
        assert result[0].close == Decimal("47200.0")
        assert result[0].volume == Decimal("1.5")

    @pytest.mark.asyncio
    async def test_fetch_historical_data_rate_limit_error(
        self, ccxt_provider: CCXTProvider, mock_exchange: MagicMock
    ) -> None:
        """Test rate limit error handling"""
        import ccxt

        mock_exchange.fetch_ohlcv = AsyncMock(
            side_effect=ccxt.RateLimitExceeded("Rate limit exceeded")
        )

        start = datetime(2022, 1, 1, tzinfo=timezone.utc)
        end = datetime(2022, 1, 2, tzinfo=timezone.utc)

        with pytest.raises(RateLimitError):
            await ccxt_provider.fetch_historical_data("BTC/USDT", "1d", start, end)

    @pytest.mark.asyncio
    async def test_get_asset_info_success(
        self, ccxt_provider: CCXTProvider, mock_exchange: MagicMock
    ) -> None:
        """Test successful asset info retrieval"""
        result = await ccxt_provider.get_asset_info("BTC/USDT")

        assert isinstance(result, AssetInfo)
        assert result.symbol == "BTC/USDT"
        assert result.asset_class == "crypto"
        assert result.exchange == "binance"
        assert result.base_currency == "BTC"
        assert result.quote_currency == "USDT"

    @pytest.mark.asyncio
    async def test_get_supported_symbols(
        self, ccxt_provider: CCXTProvider, mock_exchange: MagicMock
    ) -> None:
        """Test getting supported symbols"""
        result = await ccxt_provider.get_supported_symbols()

        assert isinstance(result, list)
        assert "BTC/USDT" in result

    def test_get_supported_timeframes(
        self, ccxt_provider: CCXTProvider, mock_exchange: MagicMock
    ) -> None:
        """Test getting supported timeframes"""
        result = ccxt_provider.get_supported_timeframes()

        assert isinstance(result, list)
        assert "1m" in result
        assert "1h" in result
        assert "1d" in result

    @pytest.mark.asyncio
    async def test_health_check_success(
        self, ccxt_provider: CCXTProvider, mock_exchange: MagicMock
    ) -> None:
        """Test successful health check"""
        mock_exchange.markets = {"BTC/USDT": {"active": True}}

        result = await ccxt_provider.health_check()

        assert isinstance(result, HealthCheck)
        assert result.status == HealthStatus.HEALTHY
        assert result.response_time_ms > 0

    @pytest.mark.asyncio
    async def test_health_check_failure(
        self, ccxt_provider: CCXTProvider, mock_exchange: MagicMock
    ) -> None:
        """Test health check failure"""
        mock_exchange.markets = {}

        with patch.object(
            ccxt_provider, "get_supported_symbols", new_callable=AsyncMock
        ) as mock_get_symbols:
            mock_get_symbols.side_effect = Exception("Connection failed")
            result = await ccxt_provider.health_check()

        assert isinstance(result, HealthCheck)
        assert result.status in [HealthStatus.DEGRADED, HealthStatus.UNHEALTHY]
        assert "Connection failed" in result.message


class TestSETProvider:
    """Test SET provider functionality"""

    @pytest.fixture
    def set_provider(self) -> SETProvider:
        """Create SET provider instance"""
        return SETProvider(api_key="test_key")

    @pytest.mark.asyncio
    async def test_fetch_historical_data_success(
        self, set_provider: SETProvider
    ) -> None:
        """Test successful historical data fetch"""
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "data": [
                {
                    "datetime": "2022-01-01T09:00:00Z",
                    "open": 100.0,
                    "high": 105.0,
                    "low": 98.0,
                    "close": 103.0,
                    "volume": 1000000,
                }
            ],
        }

        with patch.object(
            set_provider.client,
            "get",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            start = datetime(2022, 1, 1, tzinfo=timezone.utc)
            end = datetime(2022, 1, 2, tzinfo=timezone.utc)

            result = await set_provider.fetch_historical_data("PTT", "1d", start, end)

            assert len(result) == 1
            assert isinstance(result[0], OHLCV)
            assert result[0].open == Decimal("100.0")
            assert result[0].close == Decimal("103.0")

    @pytest.mark.asyncio
    async def test_fetch_historical_data_rate_limit(
        self, set_provider: SETProvider
    ) -> None:
        """Test rate limit handling"""
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {"Retry-After": "60"}

        with (
            patch.object(
                set_provider.client,
                "get",
                new_callable=AsyncMock,
                return_value=mock_response,
            ),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            start = datetime(2022, 1, 1, tzinfo=timezone.utc)
            end = datetime(2022, 1, 2, tzinfo=timezone.utc)

            with pytest.raises(RateLimitError) as exc_info:
                await set_provider.fetch_historical_data("PTT", "1d", start, end)

            assert exc_info.value.retry_after == 60.0

    @pytest.mark.asyncio
    async def test_fetch_historical_data_api_error(
        self, set_provider: SETProvider
    ) -> None:
        """Test API error handling"""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"

        with patch.object(
            set_provider.client,
            "get",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            start = datetime(2022, 1, 1, tzinfo=timezone.utc)
            end = datetime(2022, 1, 2, tzinfo=timezone.utc)

            with pytest.raises(DataProviderError) as exc_info:
                await set_provider.fetch_historical_data("PTT", "1d", start, end)

            assert "API request failed: 400" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_asset_info_success(self, set_provider: SETProvider) -> None:
        """Test successful asset info retrieval"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "data": {
                "name_th": "ปตท.",
                "name_en": "PTT Public Company Limited",
                "min_order_size": 100,
                "is_active": True,
            },
        }

        with patch.object(
            set_provider.client,
            "get",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await set_provider.get_asset_info("PTT")

            assert isinstance(result, AssetInfo)
            assert result.symbol == "PTT"
            assert result.asset_class == "stock"
            assert result.exchange == "SET"
            assert result.base_currency == "THB"

    @pytest.mark.asyncio
    async def test_get_asset_info_not_found(self, set_provider: SETProvider) -> None:
        """Test asset not found error"""
        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch.object(
            set_provider.client,
            "get",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            with pytest.raises(DataProviderError) as exc_info:
                await set_provider.get_asset_info("INVALID")

            assert "not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_supported_symbols_success(
        self, set_provider: SETProvider
    ) -> None:
        """Test getting supported symbols from API"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "data": ["PTT", "CPALL", "KBANK"],
        }

        with patch.object(
            set_provider.client,
            "get",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await set_provider.get_supported_symbols()

            assert isinstance(result, list)
            assert "PTT" in result
            assert "CPALL" in result

    @pytest.mark.asyncio
    async def test_get_supported_symbols_fallback(
        self, set_provider: SETProvider
    ) -> None:
        """Test fallback to known symbols when API fails"""
        mock_response = MagicMock()
        mock_response.status_code = 500

        with patch.object(
            set_provider.client,
            "get",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await set_provider.get_supported_symbols()

            assert isinstance(result, list)
            assert "PTT" in result  # Should be in known_symbols

    def test_get_supported_timeframes(self, set_provider: SETProvider) -> None:
        """Test getting supported timeframes"""
        result = set_provider.get_supported_timeframes()

        assert isinstance(result, list)
        assert "1m" in result
        assert "1h" in result
        assert "1d" in result

    @pytest.mark.asyncio
    async def test_network_error_handling(self, set_provider: SETProvider) -> None:
        """Test network error handling"""
        with patch.object(
            set_provider.client,
            "get",
            new_callable=AsyncMock,
            side_effect=httpx.NetworkError("Connection failed"),
        ):
            start = datetime(2022, 1, 1, tzinfo=timezone.utc)
            end = datetime(2022, 1, 2, tzinfo=timezone.utc)

            with pytest.raises(DataProviderError) as exc_info:
                await set_provider.fetch_historical_data("PTT", "1d", start, end)

            assert "Network error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_timeout_error_handling(self, set_provider: SETProvider) -> None:
        """Test timeout error handling"""
        with patch.object(
            set_provider.client,
            "get",
            new_callable=AsyncMock,
            side_effect=httpx.TimeoutException("Request timeout"),
        ):
            start = datetime(2022, 1, 1, tzinfo=timezone.utc)
            end = datetime(2022, 1, 2, tzinfo=timezone.utc)

            with pytest.raises(DataProviderError) as exc_info:
                await set_provider.fetch_historical_data("PTT", "1d", start, end)

            assert "Request timeout" in str(exc_info.value)


class TestDataProviderIntegration:
    """Integration tests for data provider framework"""

    @pytest.mark.asyncio
    async def test_provider_retry_mechanism(self) -> None:
        """Test retry mechanism with exponential backoff"""
        with patch("asyncio.sleep", new_callable=AsyncMock):
            provider = SETProvider(api_key="test_key")

            call_count = 0

            async def failing_request() -> str:
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise Exception("Temporary failure")
                return "success"

            result = await provider._make_request_with_retry(
                failing_request, max_retries=3
            )

            assert result == "success"
            assert call_count == 3

    @pytest.mark.asyncio
    async def test_provider_retry_exhaustion(self) -> None:
        """Test retry mechanism when all attempts fail"""
        with patch("asyncio.sleep", new_callable=AsyncMock):
            provider = SETProvider(api_key="test_key")

            async def always_failing_request() -> None:
                raise Exception("Persistent failure")

            with pytest.raises(DataProviderError) as exc_info:
                await provider._make_request_with_retry(
                    always_failing_request, max_retries=2
                )

            assert "Request failed after 2 retries" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_unsupported_timeframe_error(self) -> None:
        """Test error handling for unsupported timeframes"""
        provider = SETProvider(api_key="test_key")

        start = datetime(2022, 1, 1, tzinfo=timezone.utc)
        end = datetime(2022, 1, 2, tzinfo=timezone.utc)

        with pytest.raises(DataProviderError) as exc_info:
            await provider.fetch_historical_data("PTT", "30s", start, end)

        assert "Unsupported timeframe" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_provider_context_manager(self) -> None:
        """Test provider as async context manager"""
        async with SETProvider(api_key="test_key") as provider:
            assert provider.client is not None
            timeframes = provider.get_supported_timeframes()
            assert isinstance(timeframes, list)

        # Client should be closed after context exit
        # Note: In real implementation, we'd check if client is closed


class TestDataProviderManager:
    """Test data provider manager functionality"""

    @pytest.fixture
    def mock_crypto_provider(self) -> MagicMock:
        """Create mock crypto provider"""
        provider = MagicMock(spec=IDataProvider)
        provider.name = "mock_crypto"
        provider.is_healthy = True
        provider._priority = 0
        provider.fetch_historical_data = AsyncMock(
            return_value=[
                OHLCV(
                    timestamp=datetime(2022, 1, 1, tzinfo=timezone.utc),
                    open=Decimal("50000"),
                    high=Decimal("51000"),
                    low=Decimal("49000"),
                    close=Decimal("50500"),
                    volume=Decimal("100"),
                )
            ]
        )
        provider.get_asset_info = AsyncMock(
            return_value=AssetInfo(
                symbol="BTC/USDT",
                name="Bitcoin",
                asset_class="crypto",
                exchange="binance",
                base_currency="BTC",
                quote_currency="USDT",
            )
        )
        provider.health_check = AsyncMock(
            return_value=HealthCheck(
                status=HealthStatus.HEALTHY,
                message="Provider is healthy",
                response_time_ms=100.0,
            )
        )
        return provider

    @pytest.fixture
    def mock_stock_provider(self) -> MagicMock:
        """Create mock stock provider"""
        provider = MagicMock(spec=IDataProvider)
        provider.name = "mock_stock"
        provider.is_healthy = True
        provider._priority = 0
        provider.fetch_historical_data = AsyncMock(
            return_value=[
                OHLCV(
                    timestamp=datetime(2022, 1, 1, tzinfo=timezone.utc),
                    open=Decimal("100"),
                    high=Decimal("105"),
                    low=Decimal("98"),
                    close=Decimal("103"),
                    volume=Decimal("1000000"),
                )
            ]
        )
        provider.get_asset_info = AsyncMock(
            return_value=AssetInfo(
                symbol="PTT",
                name="PTT Public Company Limited",
                asset_class="stock",
                exchange="SET",
                base_currency="THB",
                quote_currency="THB",
            )
        )
        provider.health_check = AsyncMock(
            return_value=HealthCheck(
                status=HealthStatus.HEALTHY,
                message="Provider is healthy",
                response_time_ms=150.0,
            )
        )
        return provider

    @pytest.fixture
    def manager_with_providers(
        self, mock_crypto_provider: MagicMock, mock_stock_provider: MagicMock
    ) -> Any:
        """Create manager with mock providers"""
        from app.data_providers.manager import DataProviderManager

        manager = DataProviderManager(
            health_check_interval=1.0
        )  # Short interval for testing
        manager.register_provider(mock_crypto_provider, ["crypto"], priority=0)
        manager.register_provider(mock_stock_provider, ["stock"], priority=0)
        return manager

    def test_register_provider(
        self, manager_with_providers: Any, mock_crypto_provider: MagicMock
    ) -> None:
        """Test provider registration"""
        manager = manager_with_providers

        assert "mock_crypto" in manager.providers
        assert "crypto" in manager.provider_priorities
        assert manager.provider_priorities["crypto"][0] == "mock_crypto"

    def test_get_provider_for_asset_class(self, manager_with_providers: Any) -> None:
        """Test getting provider by asset class"""
        manager = manager_with_providers

        crypto_provider = manager.get_provider_for_asset_class("crypto")
        assert crypto_provider is not None
        assert crypto_provider.name == "mock_crypto"

        stock_provider = manager.get_provider_for_asset_class("stock")
        assert stock_provider is not None
        assert stock_provider.name == "mock_stock"

        # Non-existent asset class
        unknown_provider = manager.get_provider_for_asset_class("unknown")
        assert unknown_provider is None

    @pytest.mark.asyncio
    async def test_fetch_historical_data_success(
        self, manager_with_providers: Any
    ) -> None:
        """Test successful historical data fetch through manager"""
        manager = manager_with_providers

        start = datetime(2022, 1, 1, tzinfo=timezone.utc)
        end = datetime(2022, 1, 2, tzinfo=timezone.utc)

        result = await manager.fetch_historical_data(
            "BTC/USDT", "crypto", "1d", start, end
        )

        assert len(result) == 1
        assert isinstance(result[0], OHLCV)
        assert result[0].close == Decimal("50500")

    @pytest.mark.asyncio
    async def test_fetch_historical_data_fallback(
        self, manager_with_providers: Any, mock_crypto_provider: MagicMock
    ) -> None:
        """Test fallback when primary provider fails"""
        manager = manager_with_providers

        # Create a second crypto provider
        fallback_provider = MagicMock(spec=IDataProvider)
        fallback_provider.name = "fallback_crypto"
        fallback_provider.is_healthy = True
        fallback_provider._priority = 1
        fallback_provider.fetch_historical_data = AsyncMock(
            return_value=[
                OHLCV(
                    timestamp=datetime(2022, 1, 1, tzinfo=timezone.utc),
                    open=Decimal("49000"),
                    high=Decimal("50000"),
                    low=Decimal("48000"),
                    close=Decimal("49500"),
                    volume=Decimal("200"),
                )
            ]
        )

        # Register fallback provider with lower priority
        manager.register_provider(fallback_provider, ["crypto"], priority=1)

        # Make primary provider fail
        mock_crypto_provider.fetch_historical_data.side_effect = Exception(
            "Primary failed"
        )

        start = datetime(2022, 1, 1, tzinfo=timezone.utc)
        end = datetime(2022, 1, 2, tzinfo=timezone.utc)

        result = await manager.fetch_historical_data(
            "BTC/USDT", "crypto", "1d", start, end
        )

        # Should get data from fallback provider
        assert len(result) == 1
        assert result[0].close == Decimal("49500")

    @pytest.mark.asyncio
    async def test_fetch_historical_data_all_providers_fail(
        self, manager_with_providers: Any, mock_crypto_provider: MagicMock
    ) -> None:
        """Test error when all providers fail"""
        manager = manager_with_providers

        # Make provider fail
        mock_crypto_provider.fetch_historical_data.side_effect = Exception(
            "Provider failed"
        )

        start = datetime(2022, 1, 1, tzinfo=timezone.utc)
        end = datetime(2022, 1, 2, tzinfo=timezone.utc)

        with pytest.raises(DataProviderError) as exc_info:
            await manager.fetch_historical_data("BTC/USDT", "crypto", "1d", start, end)

        assert "All providers failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_asset_info_success(self, manager_with_providers: Any) -> None:
        """Test successful asset info retrieval through manager"""
        manager = manager_with_providers

        result = await manager.get_asset_info("PTT", "stock")

        assert isinstance(result, AssetInfo)
        assert result.symbol == "PTT"
        assert result.asset_class == "stock"

    @pytest.mark.asyncio
    async def test_get_health_status(self, manager_with_providers: Any) -> None:
        """Test health status retrieval"""
        manager = manager_with_providers

        health_status = await manager.get_health_status()

        assert "mock_crypto" in health_status
        assert "mock_stock" in health_status
        assert health_status["mock_crypto"].status == HealthStatus.HEALTHY
        assert health_status["mock_stock"].status == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_health_monitoring_lifecycle(
        self, manager_with_providers: Any
    ) -> None:
        """Test health monitoring start/stop"""
        manager = manager_with_providers

        # Start monitoring
        await manager.start_health_monitoring()
        assert manager._is_monitoring is True
        assert manager._health_monitor_task is not None

        # Stop monitoring
        await manager.stop_health_monitoring()
        assert manager._is_monitoring is False
        assert manager._health_monitor_task is None

    @pytest.mark.asyncio
    async def test_manager_context_manager(
        self, mock_crypto_provider: MagicMock
    ) -> None:
        """Test manager as async context manager"""
        from app.data_providers.manager import DataProviderManager

        async with DataProviderManager() as manager:
            manager.register_provider(mock_crypto_provider, ["crypto"], priority=0)
            assert "mock_crypto" in manager.providers

        # Manager should be closed after context exit
        assert manager._is_monitoring is False

    @pytest.mark.asyncio
    async def test_provider_priority_ordering(self) -> None:
        """Test that providers are ordered by priority"""
        from app.data_providers.manager import DataProviderManager

        manager = DataProviderManager()

        # Create providers with different priorities
        high_priority = MagicMock(spec=IDataProvider)
        high_priority.name = "high_priority"
        high_priority.is_healthy = True

        low_priority = MagicMock(spec=IDataProvider)
        low_priority.name = "low_priority"
        low_priority.is_healthy = True

        # Register in reverse priority order
        manager.register_provider(low_priority, ["crypto"], priority=10)
        manager.register_provider(high_priority, ["crypto"], priority=1)

        # High priority should be first
        assert manager.provider_priorities["crypto"][0] == "high_priority"
        assert manager.provider_priorities["crypto"][1] == "low_priority"

    @pytest.mark.asyncio
    async def test_unhealthy_provider_fallback(
        self, manager_with_providers: Any, mock_crypto_provider: MagicMock
    ) -> None:
        """Test fallback when provider is marked unhealthy"""
        manager = manager_with_providers

        # Create a second crypto provider
        healthy_provider = MagicMock(spec=IDataProvider)
        healthy_provider.name = "healthy_crypto"
        healthy_provider.is_healthy = True
        healthy_provider._priority = 1
        healthy_provider.fetch_historical_data = AsyncMock(
            return_value=[
                OHLCV(
                    timestamp=datetime(2022, 1, 1, tzinfo=timezone.utc),
                    open=Decimal("48000"),
                    high=Decimal("49000"),
                    low=Decimal("47000"),
                    close=Decimal("48500"),
                    volume=Decimal("150"),
                )
            ]
        )

        # Register healthy provider with lower priority
        manager.register_provider(healthy_provider, ["crypto"], priority=1)

        # Mark primary provider as unhealthy
        mock_crypto_provider.is_healthy = False

        # Should get the healthy provider
        provider = manager.get_provider_for_asset_class("crypto")
        assert provider.name == "healthy_crypto"
