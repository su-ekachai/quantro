"""
Data provider manager with health monitoring and fallback mechanisms
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from datetime import datetime
from typing import Any

from loguru import logger

from .base import (
    OHLCV,
    AssetInfo,
    DataProviderError,
    HealthCheck,
    HealthStatus,
    IDataProvider,
)


class DataProviderManager:
    """
    Manager for multiple data providers with health monitoring and fallback
    """

    def __init__(self, health_check_interval: float = 300.0):
        """
        Initialize data provider manager

        Args:
            health_check_interval: Interval in seconds between health checks
        """
        self.providers: dict[str, IDataProvider] = {}
        # asset_class -> [provider_names]
        self.provider_priorities: dict[str, list[str]] = {}
        self.health_check_interval = health_check_interval
        self._health_monitor_task: asyncio.Task | None = None
        self._is_monitoring = False

    def register_provider(
        self, provider: IDataProvider, asset_classes: list[str], priority: int = 0
    ) -> None:
        """
        Register a data provider for specific asset classes

        Args:
            provider: Data provider instance
            asset_classes: List of asset classes this provider supports
            priority: Priority level (lower number = higher priority)
        """
        self.providers[provider.name] = provider

        # Set up priorities for each asset class
        for asset_class in asset_classes:
            if asset_class not in self.provider_priorities:
                self.provider_priorities[asset_class] = []

            # Insert provider based on priority
            inserted = False
            for i, existing_provider in enumerate(
                self.provider_priorities[asset_class]
            ):
                if priority < getattr(
                    self.providers[existing_provider], "_priority", 999
                ):
                    self.provider_priorities[asset_class].insert(i, provider.name)
                    inserted = True
                    break

            if not inserted:
                self.provider_priorities[asset_class].append(provider.name)

        # Store priority on provider for sorting
        provider._priority = priority

        logger.info(
            f"Registered provider {provider.name} for asset classes: {asset_classes}"
        )

    def get_provider_for_asset_class(self, asset_class: str) -> IDataProvider | None:
        """
        Get the best available provider for an asset class

        Args:
            asset_class: Asset class (crypto, stock, commodity)

        Returns:
            Best available provider or None if no healthy providers
        """
        if asset_class not in self.provider_priorities:
            return None

        # Try providers in priority order, checking health
        for provider_name in self.provider_priorities[asset_class]:
            provider = self.providers.get(provider_name)
            if provider and provider.is_healthy:
                return provider

        # If no healthy providers, try any provider as last resort
        for provider_name in self.provider_priorities[asset_class]:
            provider = self.providers.get(provider_name)
            if provider:
                logger.warning(
                    f"Using potentially unhealthy provider {provider_name} "
                    f"for {asset_class}"
                )
                return provider

        return None

    async def fetch_historical_data(
        self,
        symbol: str,
        asset_class: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        limit: int | None = None,
    ) -> list[OHLCV]:
        """
        Fetch historical data with automatic fallback

        Args:
            symbol: Trading symbol
            asset_class: Asset class to determine provider
            timeframe: Time interval
            start: Start datetime
            end: End datetime
            limit: Maximum number of records

        Returns:
            List of OHLCV data

        Raises:
            DataProviderError: If all providers fail
        """
        if asset_class not in self.provider_priorities:
            raise DataProviderError(
                f"No providers registered for asset class: {asset_class}"
            )

        last_error = None

        # Try each provider in priority order
        for provider_name in self.provider_priorities[asset_class]:
            provider = self.providers.get(provider_name)
            if not provider:
                continue

            try:
                logger.debug(f"Trying to fetch {symbol} data from {provider_name}")
                data = await provider.fetch_historical_data(
                    symbol, timeframe, start, end, limit
                )

                if data:
                    logger.info(
                        f"Successfully fetched {len(data)} records from {provider_name}"
                    )
                    return data
                else:
                    logger.warning(
                        f"No data returned from {provider_name} for {symbol}"
                    )

            except Exception as e:
                last_error = e
                logger.warning(f"Provider {provider_name} failed for {symbol}: {e}")

                # Mark provider as potentially unhealthy
                provider._is_healthy = False
                continue

        # All providers failed
        raise DataProviderError(f"All providers failed for {symbol}: {last_error}")

    async def get_asset_info(self, symbol: str, asset_class: str) -> AssetInfo:
        """
        Get asset information with automatic fallback

        Args:
            symbol: Trading symbol
            asset_class: Asset class to determine provider

        Returns:
            Asset information

        Raises:
            DataProviderError: If all providers fail
        """
        if asset_class not in self.provider_priorities:
            raise DataProviderError(
                f"No providers registered for asset class: {asset_class}"
            )

        last_error = None

        # Try each provider in priority order
        for provider_name in self.provider_priorities[asset_class]:
            provider = self.providers.get(provider_name)
            if not provider:
                continue

            try:
                logger.debug(f"Trying to get {symbol} info from {provider_name}")
                info = await provider.get_asset_info(symbol)
                logger.info(f"Successfully got asset info from {provider_name}")
                return info

            except Exception as e:
                last_error = e
                logger.warning(
                    f"Provider {provider_name} failed for {symbol} info: {e}"
                )

                # Mark provider as potentially unhealthy
                provider._is_healthy = False
                continue

        # All providers failed
        raise DataProviderError(f"All providers failed for {symbol} info: {last_error}")

    async def subscribe_real_time(
        self, symbol: str, asset_class: str, callback: Callable[[OHLCV], None]
    ) -> None:
        """
        Subscribe to real-time data with automatic fallback

        Args:
            symbol: Trading symbol
            asset_class: Asset class to determine provider
            callback: Callback function for new data

        Raises:
            DataProviderError: If all providers fail
        """
        provider = self.get_provider_for_asset_class(asset_class)
        if not provider:
            raise DataProviderError(f"No healthy providers available for {asset_class}")

        try:
            await provider.subscribe_real_time(symbol, callback)
        except Exception as e:
            logger.error(f"Real-time subscription failed for {symbol}: {e}")
            raise DataProviderError(f"Real-time subscription failed: {e}") from e

    async def get_health_status(self) -> dict[str, HealthCheck]:
        """
        Get health status of all registered providers

        Returns:
            Dictionary mapping provider names to health checks
        """
        health_status = {}

        # Run health checks concurrently
        tasks = []
        for name, provider in self.providers.items():
            tasks.append(self._check_provider_health(name, provider))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            provider_name = list(self.providers.keys())[i]
            if isinstance(result, Exception):
                # Health check itself failed
                health_status[provider_name] = HealthCheck(
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check failed: {result}",
                    response_time_ms=0,
                    error_count=999,
                )
            elif isinstance(result, HealthCheck):
                health_status[provider_name] = result
            else:
                # Unexpected result type
                health_status[provider_name] = HealthCheck(
                    status=HealthStatus.UNHEALTHY,
                    message=f"Unexpected health check result: {type(result)}",
                    response_time_ms=0,
                    error_count=999,
                )

        return health_status

    async def _check_provider_health(
        self, name: str, provider: IDataProvider
    ) -> HealthCheck:
        """Check health of a single provider"""
        try:
            return await provider.health_check()
        except Exception as e:
            logger.error(f"Health check failed for {name}: {e}")
            return HealthCheck(
                status=HealthStatus.UNHEALTHY,
                message=f"Health check exception: {e}",
                response_time_ms=0,
                error_count=999,
            )

    async def start_health_monitoring(self) -> None:
        """Start background health monitoring"""
        if self._is_monitoring:
            return

        self._is_monitoring = True
        self._health_monitor_task = asyncio.create_task(self._health_monitor_loop())
        logger.info("Started data provider health monitoring")

    async def stop_health_monitoring(self) -> None:
        """Stop background health monitoring"""
        self._is_monitoring = False

        if self._health_monitor_task:
            self._health_monitor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._health_monitor_task
            self._health_monitor_task = None

        logger.info("Stopped data provider health monitoring")

    async def _health_monitor_loop(self) -> None:
        """Background health monitoring loop"""
        while self._is_monitoring:
            try:
                health_status = await self.get_health_status()

                # Log health status changes
                for provider_name, health in health_status.items():
                    provider = self.providers[provider_name]
                    previous_health = provider.is_healthy
                    current_health = health.status == HealthStatus.HEALTHY

                    if previous_health != current_health:
                        if current_health:
                            logger.info(f"Provider {provider_name} is now healthy")
                        else:
                            logger.warning(
                                f"Provider {provider_name} is now unhealthy: "
                                f"{health.message}"
                            )

                    # Update provider health status
                    provider._is_healthy = current_health

                # Wait for next check
                await asyncio.sleep(self.health_check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(
                    min(self.health_check_interval, 60)
                )  # Fallback interval

    async def close(self) -> None:
        """Close all providers and stop monitoring"""
        await self.stop_health_monitoring()

        # Close all providers
        for provider in self.providers.values():
            try:
                if hasattr(provider, "close"):
                    await provider.close()
            except Exception as e:
                logger.error(f"Error closing provider {provider.name}: {e}")

        logger.info("Data provider manager closed")

    async def __aenter__(self) -> DataProviderManager:
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit"""
        await self.close()


# Factory function for common setups
async def create_default_manager() -> DataProviderManager:
    """
    Create a data provider manager with default providers

    Returns:
        Configured data provider manager
    """
    from .ccxt_provider import create_binance_provider
    from .set_provider import create_set_provider

    manager = DataProviderManager(health_check_interval=300.0)  # 5 minutes

    # Register crypto provider (Binance)
    try:
        binance_provider = create_binance_provider()
        manager.register_provider(binance_provider, ["crypto"], priority=0)
    except Exception as e:
        logger.warning(f"Failed to register Binance provider: {e}")

    # Register Thai stock provider (SET)
    try:
        set_provider = create_set_provider()
        manager.register_provider(set_provider, ["stock"], priority=0)
    except Exception as e:
        logger.warning(f"Failed to register SET provider: {e}")

    # Start health monitoring
    await manager.start_health_monitoring()

    return manager
