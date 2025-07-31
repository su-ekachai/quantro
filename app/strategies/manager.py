"""
Strategy manager for registering and executing trading strategies.

This module provides the StrategyManager class that handles strategy registration,
lifecycle management, and execution coordination.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from loguru import logger

from app.strategies.base import IStrategy, MarketData, Signal, StrategyConfig, Timeframe


class StrategyRegistrationError(Exception):
    """Raised when strategy registration fails."""

    pass


class StrategyExecutionError(Exception):
    """Raised when strategy execution fails."""

    pass


class StrategyManager:
    """
    Manages strategy registration, lifecycle, and execution.

    The StrategyManager provides a pluggable architecture for trading strategies,
    allowing strategies to be registered and executed independently without
    modifying core application code.
    """

    def __init__(self) -> None:
        """Initialize the strategy manager."""
        self._strategy_classes: dict[str, type[IStrategy]] = {}
        self._active_strategies: dict[str, IStrategy] = {}
        self._signal_callbacks: list[Callable[[Signal], None]] = []

    def register_strategy_class(
        self, strategy_class: type[IStrategy], name: str | None = None
    ) -> None:
        """
        Register a strategy class for later instantiation.

        Args:
            strategy_class: The strategy class to register
            name: Optional name override (defaults to class name)

        Raises:
            StrategyRegistrationError: If registration fails
        """
        if not issubclass(strategy_class, IStrategy):
            raise StrategyRegistrationError(
                f"Strategy class {strategy_class.__name__} must inherit from IStrategy"
            )

        strategy_name = name or strategy_class.__name__

        if strategy_name in self._strategy_classes:
            logger.warning(f"Overriding existing strategy class: {strategy_name}")

        self._strategy_classes[strategy_name] = strategy_class
        logger.info(f"Registered strategy class: {strategy_name}")

    def get_registered_strategy_classes(self) -> dict[str, type[IStrategy]]:
        """
        Get all registered strategy classes.

        Returns:
            Dictionary mapping strategy names to strategy classes
        """
        return self._strategy_classes.copy()

    async def create_strategy(self, config: StrategyConfig) -> IStrategy:
        """
        Create and initialize a strategy instance.

        Args:
            config: Strategy configuration

        Returns:
            Initialized strategy instance

        Raises:
            StrategyRegistrationError: If strategy class not found
            StrategyExecutionError: If strategy initialization fails
        """
        strategy_class_name = config.name

        if strategy_class_name not in self._strategy_classes:
            raise StrategyRegistrationError(
                f"Strategy class '{strategy_class_name}' not registered. "
                f"Available strategies: {list(self._strategy_classes.keys())}"
            )

        strategy_class = self._strategy_classes[strategy_class_name]

        try:
            # Validate parameters before creating strategy
            temp_strategy = strategy_class(config)
            if not temp_strategy.validate_parameters(config.parameters):
                raise StrategyExecutionError(
                    f"Invalid parameters for strategy '{config.name}'"
                )

            # Create and initialize the strategy
            strategy = strategy_class(config)
            await strategy.initialize()

            logger.info(f"Created and initialized strategy: {config.name}")
            return strategy

        except Exception as e:
            raise StrategyExecutionError(
                f"Failed to create strategy '{config.name}': {e}"
            ) from e

    async def add_strategy(self, config: StrategyConfig) -> None:
        """
        Add a strategy to the active strategies.

        Args:
            config: Strategy configuration

        Raises:
            StrategyExecutionError: If strategy creation or addition fails
        """
        if not config.enabled:
            logger.info(f"Strategy '{config.name}' is disabled, skipping")
            return

        if config.name in self._active_strategies:
            logger.warning(f"Strategy '{config.name}' already active, replacing")
            await self.remove_strategy(config.name)

        try:
            strategy = await self.create_strategy(config)
            self._active_strategies[config.name] = strategy
            logger.info(f"Added active strategy: {config.name}")

        except Exception as e:
            raise StrategyExecutionError(
                f"Failed to add strategy '{config.name}': {e}"
            ) from e

    async def remove_strategy(self, strategy_name: str) -> None:
        """
        Remove a strategy from active strategies.

        Args:
            strategy_name: Name of strategy to remove
        """
        if strategy_name not in self._active_strategies:
            logger.warning(f"Strategy '{strategy_name}' not found in active strategies")
            return

        strategy = self._active_strategies[strategy_name]

        try:
            await strategy.cleanup()
        except Exception as e:
            logger.error(f"Error during strategy cleanup for '{strategy_name}': {e}")

        del self._active_strategies[strategy_name]
        logger.info(f"Removed strategy: {strategy_name}")

    def get_active_strategies(self) -> dict[str, IStrategy]:
        """
        Get all active strategies.

        Returns:
            Dictionary mapping strategy names to strategy instances
        """
        return self._active_strategies.copy()

    def get_strategies_for_timeframe(self, timeframe: Timeframe) -> list[IStrategy]:
        """
        Get all active strategies that support a specific timeframe.

        Args:
            timeframe: The timeframe to filter by

        Returns:
            List of strategies supporting the timeframe
        """
        return [
            strategy
            for strategy in self._active_strategies.values()
            if timeframe in strategy.timeframes
        ]

    async def process_market_data(self, data: MarketData) -> list[Signal]:
        """
        Process market data through all relevant strategies.

        Args:
            data: Market data to process

        Returns:
            List of signals generated by strategies
        """
        signals: list[Signal] = []
        strategies = self.get_strategies_for_timeframe(data.timeframe)

        if not strategies:
            logger.debug(
                f"No strategies found for timeframe {data.timeframe} "
                f"and symbol {data.symbol}"
            )
            return signals

        # Process strategies concurrently
        tasks = [self._process_strategy_data(strategy, data) for strategy in strategies]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    f"Error processing data with strategy "
                    f"'{strategies[i].name}': {result}"
                )
            elif isinstance(result, Signal):
                # Type guard ensures result is Signal here
                signals.append(result)
                # Notify signal callbacks
                for callback in self._signal_callbacks:
                    try:
                        callback(result)
                    except Exception as e:
                        logger.error(f"Error in signal callback: {e}")
            # If result is None, we simply skip it (no signal generated)

        return signals

    async def _process_strategy_data(
        self, strategy: IStrategy, data: MarketData
    ) -> Signal | None:
        """
        Process market data with a single strategy.

        Args:
            strategy: Strategy to process data with
            data: Market data to process

        Returns:
            Signal if generated, None otherwise
        """
        try:
            if not strategy.is_initialized:
                logger.warning(f"Strategy '{strategy.name}' not initialized, skipping")
                return None

            return await strategy.generate_signal(data)

        except Exception as e:
            logger.error(f"Error in strategy '{strategy.name}': {e}")
            return None

    def add_signal_callback(self, callback: Callable[[Signal], None]) -> None:
        """
        Add a callback function to be called when signals are generated.

        Args:
            callback: Function to call with generated signals
        """
        self._signal_callbacks.append(callback)
        logger.info("Added signal callback")

    def remove_signal_callback(self, callback: Callable[[Signal], None]) -> None:
        """
        Remove a signal callback.

        Args:
            callback: Callback function to remove
        """
        if callback in self._signal_callbacks:
            self._signal_callbacks.remove(callback)
            logger.info("Removed signal callback")

    async def shutdown(self) -> None:
        """
        Shutdown the strategy manager and clean up all strategies.
        """
        logger.info("Shutting down strategy manager")

        # Clean up all active strategies
        strategy_names = list(self._active_strategies.keys())
        for strategy_name in strategy_names:
            await self.remove_strategy(strategy_name)

        # Clear callbacks
        self._signal_callbacks.clear()

        logger.info("Strategy manager shutdown complete")

    def get_strategy_info(self, strategy_name: str) -> dict[str, Any] | None:
        """
        Get information about a specific strategy.

        Args:
            strategy_name: Name of the strategy

        Returns:
            Dictionary with strategy information or None if not found
        """
        if strategy_name not in self._active_strategies:
            return None

        strategy = self._active_strategies[strategy_name]
        return {
            "name": strategy.name,
            "class": strategy.__class__.__name__,
            "timeframes": [tf.value for tf in strategy.timeframes],
            "initialized": strategy.is_initialized,
            "config": strategy.config.model_dump(),
            "required_indicators": strategy.get_required_indicators(),
        }

    def get_all_strategies_info(self) -> dict[str, dict[str, Any]]:
        """
        Get information about all active strategies.

        Returns:
            Dictionary mapping strategy names to their information
        """
        result: dict[str, dict[str, Any]] = {}
        for name in self._active_strategies:
            info = self.get_strategy_info(name)
            if info is not None:
                result[name] = info
        return result
