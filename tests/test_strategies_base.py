"""
Tests for the base strategy interface and models.
"""

from datetime import datetime
from typing import Any

import pytest
from pydantic import ValidationError

from app.strategies.base import (
    IStrategy,
    MarketData,
    Signal,
    SignalType,
    StrategyConfig,
    Timeframe,
)


class TestStrategyConfig:
    """Test cases for StrategyConfig model."""

    def test_strategy_config_defaults(self) -> None:
        """Test StrategyConfig with default values."""
        config = StrategyConfig(name="TestStrategy")

        assert config.name == "TestStrategy"
        assert config.timeframes == [Timeframe.ONE_HOUR]
        assert config.parameters == {}
        assert config.enabled is True
        assert config.max_positions == 1
        assert config.risk_per_trade == 0.02

    def test_strategy_config_custom_values(self) -> None:
        """Test StrategyConfig with custom values."""
        config = StrategyConfig(
            name="CustomStrategy",
            timeframes=[Timeframe.FIVE_MINUTES, Timeframe.ONE_DAY],
            parameters={"param1": "value1", "param2": 42},
            enabled=False,
            max_positions=3,
            risk_per_trade=0.05,
        )

        assert config.name == "CustomStrategy"
        assert config.timeframes == [Timeframe.FIVE_MINUTES, Timeframe.ONE_DAY]
        assert config.parameters == {"param1": "value1", "param2": 42}
        assert config.enabled is False
        assert config.max_positions == 3
        assert config.risk_per_trade == 0.05

    def test_strategy_config_validation_empty_name(self) -> None:
        """Test validation fails for empty strategy name."""
        with pytest.raises(ValidationError) as exc_info:
            StrategyConfig(name="")

        assert "Strategy name cannot be empty" in str(exc_info.value)

    def test_strategy_config_validation_whitespace_name(self) -> None:
        """Test validation trims whitespace from strategy name."""
        config = StrategyConfig(name="  TestStrategy  ")
        assert config.name == "TestStrategy"

    def test_strategy_config_validation_empty_timeframes(self) -> None:
        """Test validation fails for empty timeframes list."""
        with pytest.raises(ValidationError) as exc_info:
            StrategyConfig(name="TestStrategy", timeframes=[])

        assert "At least one timeframe must be specified" in str(exc_info.value)

    def test_strategy_config_validation_invalid_max_positions(self) -> None:
        """Test validation fails for invalid max_positions."""
        with pytest.raises(ValidationError):
            StrategyConfig(name="TestStrategy", max_positions=0)

    def test_strategy_config_validation_invalid_risk_per_trade(self) -> None:
        """Test validation fails for invalid risk_per_trade."""
        with pytest.raises(ValidationError):
            StrategyConfig(name="TestStrategy", risk_per_trade=0.0)

        with pytest.raises(ValidationError):
            StrategyConfig(name="TestStrategy", risk_per_trade=0.15)


class TestSignal:
    """Test cases for Signal model."""

    def test_signal_creation(self) -> None:
        """Test Signal creation with valid data."""
        timestamp = datetime.now()
        signal = Signal(
            signal_type=SignalType.BUY,
            confidence=0.85,
            timestamp=timestamp,
            symbol="BTCUSDT",
            timeframe=Timeframe.ONE_HOUR,
            strategy_name="TestStrategy",
            price=50000.0,
            metadata={"indicator": "RSI", "value": 30},
        )

        assert signal.signal_type == SignalType.BUY
        assert signal.confidence == 0.85
        assert signal.timestamp == timestamp
        assert signal.symbol == "BTCUSDT"
        assert signal.timeframe == Timeframe.ONE_HOUR
        assert signal.strategy_name == "TestStrategy"
        assert signal.price == 50000.0
        assert signal.metadata == {"indicator": "RSI", "value": 30}

    def test_signal_defaults(self) -> None:
        """Test Signal with default values."""
        timestamp = datetime.now()
        signal = Signal(
            signal_type=SignalType.HOLD,
            confidence=0.5,
            timestamp=timestamp,
            symbol="ETHUSDT",
            timeframe=Timeframe.FIVE_MINUTES,
            strategy_name="TestStrategy",
        )

        assert signal.price is None
        assert signal.metadata == {}

    def test_signal_validation_confidence_range(self) -> None:
        """Test Signal validation for confidence range."""
        timestamp = datetime.now()

        # Valid confidence values
        Signal(
            signal_type=SignalType.BUY,
            confidence=0.0,
            timestamp=timestamp,
            symbol="BTCUSDT",
            timeframe=Timeframe.ONE_HOUR,
            strategy_name="TestStrategy",
        )

        Signal(
            signal_type=SignalType.BUY,
            confidence=1.0,
            timestamp=timestamp,
            symbol="BTCUSDT",
            timeframe=Timeframe.ONE_HOUR,
            strategy_name="TestStrategy",
        )

        # Invalid confidence values
        with pytest.raises(ValidationError):
            Signal(
                signal_type=SignalType.BUY,
                confidence=-0.1,
                timestamp=timestamp,
                symbol="BTCUSDT",
                timeframe=Timeframe.ONE_HOUR,
                strategy_name="TestStrategy",
            )

        with pytest.raises(ValidationError):
            Signal(
                signal_type=SignalType.BUY,
                confidence=1.1,
                timestamp=timestamp,
                symbol="BTCUSDT",
                timeframe=Timeframe.ONE_HOUR,
                strategy_name="TestStrategy",
            )

    def test_signal_validation_empty_symbol(self) -> None:
        """Test Signal validation fails for empty symbol."""
        timestamp = datetime.now()

        with pytest.raises(ValidationError) as exc_info:
            Signal(
                signal_type=SignalType.BUY,
                confidence=0.8,
                timestamp=timestamp,
                symbol="",
                timeframe=Timeframe.ONE_HOUR,
                strategy_name="TestStrategy",
            )

        assert "Symbol cannot be empty" in str(exc_info.value)

    def test_signal_validation_symbol_uppercase(self) -> None:
        """Test Signal converts symbol to uppercase."""
        timestamp = datetime.now()
        signal = Signal(
            signal_type=SignalType.BUY,
            confidence=0.8,
            timestamp=timestamp,
            symbol="btcusdt",
            timeframe=Timeframe.ONE_HOUR,
            strategy_name="TestStrategy",
        )

        assert signal.symbol == "BTCUSDT"

    def test_signal_validation_empty_strategy_name(self) -> None:
        """Test Signal validation fails for empty strategy name."""
        timestamp = datetime.now()

        with pytest.raises(ValidationError) as exc_info:
            Signal(
                signal_type=SignalType.BUY,
                confidence=0.8,
                timestamp=timestamp,
                symbol="BTCUSDT",
                timeframe=Timeframe.ONE_HOUR,
                strategy_name="",
            )

        assert "Strategy name cannot be empty" in str(exc_info.value)


class TestMarketData:
    """Test cases for MarketData model."""

    def test_market_data_creation(self) -> None:
        """Test MarketData creation with valid data."""
        timestamp = datetime.now()
        data = MarketData(
            symbol="BTCUSDT",
            timestamp=timestamp,
            open=49000.0,
            high=51000.0,
            low=48000.0,
            close=50000.0,
            volume=1000.0,
            timeframe=Timeframe.ONE_HOUR,
        )

        assert data.symbol == "BTCUSDT"
        assert data.timestamp == timestamp
        assert data.open == 49000.0
        assert data.high == 51000.0
        assert data.low == 48000.0
        assert data.close == 50000.0
        assert data.volume == 1000.0
        assert data.timeframe == Timeframe.ONE_HOUR

    def test_market_data_validation_symbol_uppercase(self) -> None:
        """Test MarketData converts symbol to uppercase."""
        timestamp = datetime.now()
        data = MarketData(
            symbol="btcusdt",
            timestamp=timestamp,
            open=49000.0,
            high=51000.0,
            low=48000.0,
            close=50000.0,
            volume=1000.0,
            timeframe=Timeframe.ONE_HOUR,
        )

        assert data.symbol == "BTCUSDT"

    def test_market_data_validation_empty_symbol(self) -> None:
        """Test MarketData validation fails for empty symbol."""
        timestamp = datetime.now()

        with pytest.raises(ValidationError) as exc_info:
            MarketData(
                symbol="",
                timestamp=timestamp,
                open=49000.0,
                high=51000.0,
                low=48000.0,
                close=50000.0,
                volume=1000.0,
                timeframe=Timeframe.ONE_HOUR,
            )

        assert "Symbol cannot be empty" in str(exc_info.value)

    def test_market_data_validation_positive_prices(self) -> None:
        """Test MarketData validation for positive prices."""
        timestamp = datetime.now()

        # Test negative prices
        with pytest.raises(ValidationError) as exc_info:
            MarketData(
                symbol="BTCUSDT",
                timestamp=timestamp,
                open=-1.0,
                high=51000.0,
                low=48000.0,
                close=50000.0,
                volume=1000.0,
                timeframe=Timeframe.ONE_HOUR,
            )

        assert "Prices must be positive" in str(exc_info.value)

    def test_market_data_validation_non_negative_volume(self) -> None:
        """Test MarketData validation for non-negative volume."""
        timestamp = datetime.now()

        # Valid zero volume
        MarketData(
            symbol="BTCUSDT",
            timestamp=timestamp,
            open=49000.0,
            high=51000.0,
            low=48000.0,
            close=50000.0,
            volume=0.0,
            timeframe=Timeframe.ONE_HOUR,
        )

        # Invalid negative volume
        with pytest.raises(ValidationError) as exc_info:
            MarketData(
                symbol="BTCUSDT",
                timestamp=timestamp,
                open=49000.0,
                high=51000.0,
                low=48000.0,
                close=50000.0,
                volume=-1.0,
                timeframe=Timeframe.ONE_HOUR,
            )

        assert "Volume must be non-negative" in str(exc_info.value)


class MockStrategy(IStrategy):
    """Mock strategy implementation for testing."""

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)
        self.initialize_called = False
        self.generate_signal_called = False
        self.validate_parameters_called = False
        self.get_required_indicators_called = False
        self.cleanup_called = False

    async def initialize(self) -> None:
        """Mock initialize method."""
        self.initialize_called = True
        self._initialized = True

    async def generate_signal(self, data: MarketData) -> Signal | None:
        """Mock generate_signal method."""
        self.generate_signal_called = True
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
        self.validate_parameters_called = True
        return True

    def get_required_indicators(self) -> list[str]:
        """Mock get_required_indicators method."""
        self.get_required_indicators_called = True
        return ["RSI", "SMA"]

    async def cleanup(self) -> None:
        """Mock cleanup method."""
        self.cleanup_called = True
        await super().cleanup()


class TestIStrategy:
    """Test cases for IStrategy abstract base class."""

    def test_strategy_initialization(self) -> None:
        """Test strategy initialization."""
        config = StrategyConfig(name="MockStrategy")
        strategy = MockStrategy(config)

        assert strategy.name == "MockStrategy"
        assert strategy.timeframes == [Timeframe.ONE_HOUR]
        assert strategy.is_initialized is False
        assert strategy.config == config

    @pytest.mark.asyncio
    async def test_strategy_lifecycle(self) -> None:
        """Test strategy lifecycle methods."""
        config = StrategyConfig(name="MockStrategy")
        strategy = MockStrategy(config)

        # Test initialization
        await strategy.initialize()
        assert strategy.initialize_called is True
        assert strategy.is_initialized is True

        # Test signal generation
        timestamp = datetime.now()
        data = MarketData(
            symbol="BTCUSDT",
            timestamp=timestamp,
            open=49000.0,
            high=51000.0,
            low=48000.0,
            close=50000.0,
            volume=1000.0,
            timeframe=Timeframe.ONE_HOUR,
        )

        signal = await strategy.generate_signal(data)
        assert strategy.generate_signal_called is True
        assert signal is not None
        assert signal.signal_type == SignalType.BUY
        assert signal.symbol == "BTCUSDT"

        # Test parameter validation
        result = strategy.validate_parameters({"param": "value"})
        assert strategy.validate_parameters_called is True
        assert result is True

        # Test required indicators
        indicators = strategy.get_required_indicators()
        assert strategy.get_required_indicators_called is True
        assert indicators == ["RSI", "SMA"]

        # Test cleanup
        await strategy.cleanup()
        assert strategy.cleanup_called is True
        assert strategy.is_initialized is False

    def test_strategy_string_representations(self) -> None:
        """Test strategy string representations."""
        config = StrategyConfig(name="MockStrategy")
        strategy = MockStrategy(config)

        assert str(strategy) == "MockStrategy(name='MockStrategy')"
        assert "MockStrategy" in repr(strategy)
        assert "name='MockStrategy'" in repr(strategy)
        assert "initialized=False" in repr(strategy)
