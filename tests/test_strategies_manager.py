"""
Tests for the strategy manager.
"""

from datetime import datetime
from typing import Any
from unittest.mock import MagicMock

import pytest

from app.strategies.base import (
    IStrategy,
    MarketData,
    Signal,
    SignalType,
    StrategyConfig,
    Timeframe,
)
from app.strategies.manager import (
    StrategyExecutionError,
    StrategyManager,
    StrategyRegistrationError,
)


class MockStrategy(IStrategy):
    """Mock strategy for testing."""

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)
        self.initialize_called = False
        self.generate_signal_called = False
        self.cleanup_called = False
        self._should_fail_init = False
        self._should_fail_signal = False
        self._should_return_signal = True

    def set_fail_init(self, should_fail: bool) -> None:
        """Set whether initialize should fail."""
        self._should_fail_init = should_fail

    def set_fail_signal(self, should_fail: bool) -> None:
        """Set whether generate_signal should fail."""
        self._should_fail_signal = should_fail

    def set_return_signal(self, should_return: bool) -> None:
        """Set whether generate_signal should return a signal."""
        self._should_return_signal = should_return

    async def initialize(self) -> None:
        """Mock initialize method."""
        self.initialize_called = True
        if self._should_fail_init:
            raise RuntimeError("Mock initialization failure")
        self._initialized = True

    async def generate_signal(self, data: MarketData) -> Signal | None:
        """Mock generate_signal method."""
        self.generate_signal_called = True
        if self._should_fail_signal:
            raise RuntimeError("Mock signal generation failure")

        if not self._should_return_signal:
            return None

        return Signal(
            signal_type=SignalType.BUY,
            confidence=0.8,
            timestamp=data.timestamp,
            symbol=data.symbol,
            timeframe=data.timeframe,
            strategy_name=self.name,
            price=data.close,
        )

    def validate_parameters(self, params: dict[str, Any]) -> bool:
        """Mock validate_parameters method."""
        # Fail validation if 'invalid' key is present
        return "invalid" not in params

    def get_required_indicators(self) -> list[str]:
        """Mock get_required_indicators method."""
        return ["RSI", "SMA"]

    async def cleanup(self) -> None:
        """Mock cleanup method."""
        self.cleanup_called = True
        await super().cleanup()


class InvalidStrategy:
    """Invalid strategy class that doesn't inherit from IStrategy."""

    pass


class TestStrategyManager:
    """Test cases for StrategyManager."""

    @pytest.fixture
    def manager(self) -> StrategyManager:
        """Create a strategy manager for testing."""
        return StrategyManager()

    @pytest.fixture
    def mock_config(self) -> StrategyConfig:
        """Create a mock strategy config."""
        return StrategyConfig(
            name="MockStrategy",
            timeframes=[Timeframe.ONE_HOUR, Timeframe.FIVE_MINUTES],
        )

    def test_manager_initialization(self, manager: StrategyManager) -> None:
        """Test StrategyManager initialization."""
        assert len(manager.get_registered_strategy_classes()) == 0
        assert len(manager.get_active_strategies()) == 0

    def test_register_strategy_class(self, manager: StrategyManager) -> None:
        """Test strategy class registration."""
        manager.register_strategy_class(MockStrategy)

        registered = manager.get_registered_strategy_classes()
        assert "MockStrategy" in registered
        assert registered["MockStrategy"] == MockStrategy

    def test_register_strategy_class_with_custom_name(
        self, manager: StrategyManager
    ) -> None:
        """Test strategy class registration with custom name."""
        manager.register_strategy_class(MockStrategy, "CustomName")

        registered = manager.get_registered_strategy_classes()
        assert "CustomName" in registered
        assert registered["CustomName"] == MockStrategy

    def test_register_invalid_strategy_class(self, manager: StrategyManager) -> None:
        """Test registration fails for invalid strategy class."""
        with pytest.raises(StrategyRegistrationError) as exc_info:
            manager.register_strategy_class(InvalidStrategy)  # type: ignore

        assert "must inherit from IStrategy" in str(exc_info.value)

    def test_register_strategy_class_override(self, manager: StrategyManager) -> None:
        """Test strategy class registration override."""
        manager.register_strategy_class(MockStrategy)
        manager.register_strategy_class(MockStrategy)  # Should override

        registered = manager.get_registered_strategy_classes()
        assert len(registered) == 1
        assert "MockStrategy" in registered

    @pytest.mark.asyncio
    async def test_create_strategy_success(
        self, manager: StrategyManager, mock_config: StrategyConfig
    ) -> None:
        """Test successful strategy creation."""
        manager.register_strategy_class(MockStrategy)

        strategy = await manager.create_strategy(mock_config)

        assert isinstance(strategy, MockStrategy)
        assert strategy.name == "MockStrategy"
        assert strategy.is_initialized is True
        assert strategy.initialize_called is True

    @pytest.mark.asyncio
    async def test_create_strategy_unregistered_class(
        self, manager: StrategyManager, mock_config: StrategyConfig
    ) -> None:
        """Test strategy creation fails for unregistered class."""
        with pytest.raises(StrategyRegistrationError) as exc_info:
            await manager.create_strategy(mock_config)

        assert "not registered" in str(exc_info.value)
        assert "Available strategies" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_create_strategy_invalid_parameters(
        self, manager: StrategyManager
    ) -> None:
        """Test strategy creation fails for invalid parameters."""
        manager.register_strategy_class(MockStrategy)

        config = StrategyConfig(
            name="MockStrategy",
            parameters={"invalid": "param"},  # This will fail validation
        )

        with pytest.raises(StrategyExecutionError) as exc_info:
            await manager.create_strategy(config)

        assert "Invalid parameters" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_create_strategy_initialization_failure(
        self, manager: StrategyManager, mock_config: StrategyConfig
    ) -> None:
        """Test strategy creation fails when initialization fails."""

        class FailingMockStrategy(MockStrategy):
            """Mock strategy that fails during initialization."""

            async def initialize(self) -> None:
                """Mock initialize method that fails."""
                self.initialize_called = True
                raise RuntimeError("Mock initialization failure")

        manager.register_strategy_class(FailingMockStrategy, "MockStrategy")

        with pytest.raises(StrategyExecutionError) as exc_info:
            await manager.create_strategy(mock_config)

        assert "Failed to create strategy" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_add_strategy_success(
        self, manager: StrategyManager, mock_config: StrategyConfig
    ) -> None:
        """Test successful strategy addition."""
        manager.register_strategy_class(MockStrategy)

        await manager.add_strategy(mock_config)

        active = manager.get_active_strategies()
        assert "MockStrategy" in active
        assert isinstance(active["MockStrategy"], MockStrategy)

    @pytest.mark.asyncio
    async def test_add_strategy_disabled(self, manager: StrategyManager) -> None:
        """Test adding disabled strategy is skipped."""
        manager.register_strategy_class(MockStrategy)

        config = StrategyConfig(name="MockStrategy", enabled=False)
        await manager.add_strategy(config)

        active = manager.get_active_strategies()
        assert len(active) == 0

    @pytest.mark.asyncio
    async def test_add_strategy_replace_existing(
        self, manager: StrategyManager, mock_config: StrategyConfig
    ) -> None:
        """Test adding strategy replaces existing one."""
        manager.register_strategy_class(MockStrategy)

        # Add first strategy
        await manager.add_strategy(mock_config)
        first_strategy = manager.get_active_strategies()["MockStrategy"]

        # Add second strategy with same name
        await manager.add_strategy(mock_config)
        second_strategy = manager.get_active_strategies()["MockStrategy"]

        assert first_strategy is not second_strategy
        assert isinstance(first_strategy, MockStrategy)
        assert first_strategy.cleanup_called is True

    @pytest.mark.asyncio
    async def test_remove_strategy(
        self, manager: StrategyManager, mock_config: StrategyConfig
    ) -> None:
        """Test strategy removal."""
        manager.register_strategy_class(MockStrategy)
        await manager.add_strategy(mock_config)

        strategy = manager.get_active_strategies()["MockStrategy"]
        assert isinstance(strategy, MockStrategy)
        await manager.remove_strategy("MockStrategy")

        active = manager.get_active_strategies()
        assert "MockStrategy" not in active
        assert strategy.cleanup_called is True

    @pytest.mark.asyncio
    async def test_remove_nonexistent_strategy(self, manager: StrategyManager) -> None:
        """Test removing nonexistent strategy doesn't raise error."""
        await manager.remove_strategy("NonexistentStrategy")  # Should not raise

    @pytest.mark.asyncio
    async def test_get_strategies_for_timeframe(self, manager: StrategyManager) -> None:
        """Test getting strategies for specific timeframe."""
        manager.register_strategy_class(MockStrategy)

        # Create strategy with multiple timeframes
        config1 = StrategyConfig(
            name="MockStrategy", timeframes=[Timeframe.ONE_HOUR, Timeframe.FIVE_MINUTES]
        )
        config2 = StrategyConfig(name="MockStrategy2", timeframes=[Timeframe.ONE_DAY])

        # Register second strategy class
        class MockStrategy2(MockStrategy):
            pass

        manager.register_strategy_class(MockStrategy2, "MockStrategy2")

        # Add strategies
        await manager.add_strategy(config1)
        await manager.add_strategy(config2)

        # Test filtering
        hour_strategies = manager.get_strategies_for_timeframe(Timeframe.ONE_HOUR)
        assert len(hour_strategies) == 1
        assert hour_strategies[0].name == "MockStrategy"

        day_strategies = manager.get_strategies_for_timeframe(Timeframe.ONE_DAY)
        assert len(day_strategies) == 1
        assert day_strategies[0].name == "MockStrategy2"

        minute_strategies = manager.get_strategies_for_timeframe(Timeframe.ONE_MINUTE)
        assert len(minute_strategies) == 0

    @pytest.mark.asyncio
    async def test_process_market_data_success(
        self, manager: StrategyManager, mock_config: StrategyConfig
    ) -> None:
        """Test successful market data processing."""
        manager.register_strategy_class(MockStrategy)
        await manager.add_strategy(mock_config)

        data = MarketData(
            symbol="BTCUSDT",
            timestamp=datetime.now(),
            open=49000.0,
            high=51000.0,
            low=48000.0,
            close=50000.0,
            volume=1000.0,
            timeframe=Timeframe.ONE_HOUR,
        )

        signals = await manager.process_market_data(data)

        assert len(signals) == 1
        assert signals[0].signal_type == SignalType.BUY
        assert signals[0].symbol == "BTCUSDT"

    @pytest.mark.asyncio
    async def test_process_market_data_no_strategies(
        self, manager: StrategyManager
    ) -> None:
        """Test market data processing with no matching strategies."""
        data = MarketData(
            symbol="BTCUSDT",
            timestamp=datetime.now(),
            open=49000.0,
            high=51000.0,
            low=48000.0,
            close=50000.0,
            volume=1000.0,
            timeframe=Timeframe.ONE_HOUR,
        )

        signals = await manager.process_market_data(data)
        assert len(signals) == 0

    @pytest.mark.asyncio
    async def test_process_market_data_strategy_error(
        self, manager: StrategyManager, mock_config: StrategyConfig
    ) -> None:
        """Test market data processing handles strategy errors."""
        manager.register_strategy_class(MockStrategy)
        await manager.add_strategy(mock_config)

        # Make strategy fail during signal generation
        strategy = manager.get_active_strategies()["MockStrategy"]
        assert isinstance(strategy, MockStrategy)
        strategy.set_fail_signal(True)

        data = MarketData(
            symbol="BTCUSDT",
            timestamp=datetime.now(),
            open=49000.0,
            high=51000.0,
            low=48000.0,
            close=50000.0,
            volume=1000.0,
            timeframe=Timeframe.ONE_HOUR,
        )

        signals = await manager.process_market_data(data)
        assert len(signals) == 0  # Error should be handled gracefully

    @pytest.mark.asyncio
    async def test_process_market_data_no_signal(
        self, manager: StrategyManager, mock_config: StrategyConfig
    ) -> None:
        """Test market data processing when strategy returns no signal."""
        manager.register_strategy_class(MockStrategy)
        await manager.add_strategy(mock_config)

        # Make strategy return no signal
        strategy = manager.get_active_strategies()["MockStrategy"]
        assert isinstance(strategy, MockStrategy)
        strategy.set_return_signal(False)

        data = MarketData(
            symbol="BTCUSDT",
            timestamp=datetime.now(),
            open=49000.0,
            high=51000.0,
            low=48000.0,
            close=50000.0,
            volume=1000.0,
            timeframe=Timeframe.ONE_HOUR,
        )

        signals = await manager.process_market_data(data)
        assert len(signals) == 0

    def test_signal_callbacks(self, manager: StrategyManager) -> None:
        """Test signal callback management."""
        callback1 = MagicMock()
        callback2 = MagicMock()

        # Add callbacks
        manager.add_signal_callback(callback1)
        manager.add_signal_callback(callback2)

        # Remove one callback
        manager.remove_signal_callback(callback1)

        # Verify callback list
        assert callback2 in manager._signal_callbacks
        assert callback1 not in manager._signal_callbacks

    @pytest.mark.asyncio
    async def test_signal_callbacks_called(
        self, manager: StrategyManager, mock_config: StrategyConfig
    ) -> None:
        """Test signal callbacks are called when signals are generated."""
        callback = MagicMock()
        manager.add_signal_callback(callback)

        manager.register_strategy_class(MockStrategy)
        await manager.add_strategy(mock_config)

        data = MarketData(
            symbol="BTCUSDT",
            timestamp=datetime.now(),
            open=49000.0,
            high=51000.0,
            low=48000.0,
            close=50000.0,
            volume=1000.0,
            timeframe=Timeframe.ONE_HOUR,
        )

        signals = await manager.process_market_data(data)

        assert len(signals) == 1
        callback.assert_called_once_with(signals[0])

    @pytest.mark.asyncio
    async def test_shutdown(
        self, manager: StrategyManager, mock_config: StrategyConfig
    ) -> None:
        """Test manager shutdown."""
        callback = MagicMock()
        manager.add_signal_callback(callback)

        manager.register_strategy_class(MockStrategy)
        await manager.add_strategy(mock_config)

        strategy = manager.get_active_strategies()["MockStrategy"]
        assert isinstance(strategy, MockStrategy)

        await manager.shutdown()

        assert len(manager.get_active_strategies()) == 0
        assert len(manager._signal_callbacks) == 0
        assert strategy.cleanup_called is True

    @pytest.mark.asyncio
    async def test_get_strategy_info(
        self, manager: StrategyManager, mock_config: StrategyConfig
    ) -> None:
        """Test getting strategy information."""
        manager.register_strategy_class(MockStrategy)
        await manager.add_strategy(mock_config)

        info = manager.get_strategy_info("MockStrategy")

        assert info is not None
        assert info["name"] == "MockStrategy"
        assert info["class"] == "MockStrategy"
        assert info["timeframes"] == ["1h", "5m"]
        assert info["initialized"] is True
        assert "config" in info
        assert info["required_indicators"] == ["RSI", "SMA"]

    def test_get_strategy_info_nonexistent(self, manager: StrategyManager) -> None:
        """Test getting info for nonexistent strategy."""
        info = manager.get_strategy_info("NonexistentStrategy")
        assert info is None

    @pytest.mark.asyncio
    async def test_get_all_strategies_info(
        self, manager: StrategyManager, mock_config: StrategyConfig
    ) -> None:
        """Test getting all strategies information."""
        manager.register_strategy_class(MockStrategy)
        await manager.add_strategy(mock_config)

        all_info = manager.get_all_strategies_info()

        assert len(all_info) == 1
        assert "MockStrategy" in all_info
        assert all_info["MockStrategy"]["name"] == "MockStrategy"
