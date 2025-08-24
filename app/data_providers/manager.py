"""
Data provider manager for handling multiple data sources.

This module provides a centralized manager for accessing different
data providers based on symbol requirements.
"""

from __future__ import annotations

from app.data_providers.base import IDataProvider


class DataProviderManager:
    """Manager for coordinating multiple data providers."""

    def __init__(self) -> None:
        """Initialize the data provider manager."""
        self.providers: dict[str, IDataProvider] = {}

    def register_provider(self, name: str, provider: IDataProvider) -> None:
        """
        Register a data provider.

        Args:
            name: Provider name
            provider: Provider instance
        """
        self.providers[name] = provider

    async def get_provider_for_symbol(self, symbol: str) -> IDataProvider:
        """
        Get appropriate data provider for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Data provider instance

        Raises:
            ValueError: If no suitable provider found
        """
        # Simple logic - can be enhanced based on symbol patterns
        if symbol.endswith("USD") or symbol.endswith("USDT"):
            # Crypto symbols
            if "ccxt" in self.providers:
                return self.providers["ccxt"]
        elif symbol.endswith(".BK") and "set" in self.providers:
            # Thai stocks
            return self.providers["set"]

        # Default to first available provider
        if self.providers:
            return next(iter(self.providers.values()))

        raise ValueError(f"No data provider available for symbol: {symbol}")


def create_default_manager() -> DataProviderManager:
    """
    Create a default data provider manager with common providers.

    Returns:
        Configured data provider manager
    """
    manager = DataProviderManager()

    # Register providers as they become available
    # This is a placeholder - actual providers would be registered here

    return manager
