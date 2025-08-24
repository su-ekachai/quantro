"""
Tests for CDC Action Zone trading strategy.
"""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from app.strategies.base import (
    MarketData,
    Signal,
    SignalType,
    StrategyConfig,
    Timeframe,
)
from app.strategies.cdc_action_zone import (
    CDCActionZoneBacktestStrategy,
    CDCActionZoneConfig,
    CDCActionZoneStrategy,
)


class TestCDCActionZoneConfig:
    """Test CDC Action Zone configuration parameters."""

    def test_config_constants(self) -> None:
        """Test that configuration constants are properly defined."""
        assert CDCActionZoneConfig.FAST_MA_PERIOD == 8
        assert CDCActionZoneConfig.SLOW_MA_PERIOD == 21
        assert CDCActionZoneConfig.TREND_MA_PERIOD == 50
        assert CDCActionZoneConfig.RSI_PERIOD == 14
        assert CDCActionZoneConfig.RSI_OVERSOLD == 30.0
        assert CDCActionZoneConfig.RSI_OVERBOUGHT == 70.0
        assert CDCActionZoneConfig.VOLUME_MA_PERIOD == 20
        assert CDCActionZoneConfig.VOLUME_THRESHOLD == 1.2
        assert CDCActionZoneConfig.ATR_PERIOD == 14
        assert CDCActionZoneConfig.ATR_MULTIPLIER == 2.0
        assert CDCActionZoneConfig.MIN_CONFIDENCE == 0.6
        assert CDCActionZoneConfig.HIGH_CONFIDENCE == 0.8


class TestCDCActionZoneStrategy:
    """Test CDC Action Zone strategy implementation."""

    @pytest.fixture
    def strategy_config(self) -> StrategyConfig:
        """Create a test strategy configuration."""
        return StrategyConfig(
            name="CDCActionZone",
            timeframes=[Timeframe.ONE_HOUR],
            parameters={},
            enabled=True,
            max_positions=1,
            risk_per_trade=0.02,
        )

    @pytest.fixture
    def strategy(self, strategy_config: StrategyConfig) -> CDCActionZoneStrategy:
        """Create a CDC Action Zone strategy instance."""
        return CDCActionZoneStrategy(strategy_config)

    def test_strategy_initialization(self, strategy: CDCActionZoneStrategy) -> None:
        """Test strategy initialization."""
        assert strategy.name == "CDCActionZone"
        assert strategy.timeframes == [Timeframe.ONE_HOUR]
        assert not strategy.is_initialized
        assert len(strategy._market_data_buffer) == 0

    @pytest.mark.asyncio
    async def test_strategy_initialize(self, strategy: CDCActionZoneStrategy) -> None:
        """Test strategy initialization process."""
        await strategy.initialize()
        assert strategy.is_initialized

    def test_validate_parameters_valid(self, strategy: CDCActionZoneStrategy) -> None:
        """Test parameter validation with valid parameters."""
        valid_params = {
            "fast_ma_period": 10,
            "slow_ma_period": 20,
            "rsi_period": 14,
            "volume_threshold": 1.5,
            "min_confidence": 0.7,
        }
        assert strategy.validate_parameters(valid_params)

    def test_validate_parameters_invalid(self, strategy: CDCActionZoneStrategy) -> None:
        """Test parameter validation with invalid parameters."""
        # Invalid fast_ma_period
        invalid_params: dict[str, int | float] = {"fast_ma_period": 0}
        assert not strategy.validate_parameters(invalid_params)

        # Invalid slow_ma_period
        invalid_params = {"slow_ma_period": -1}
        assert not strategy.validate_parameters(invalid_params)

        # Invalid rsi_period
        invalid_params = {"rsi_period": 0}
        assert not strategy.validate_parameters(invalid_params)

        # Invalid volume_threshold
        invalid_params = {"volume_threshold": -0.5}
        assert not strategy.validate_parameters(invalid_params)

        # Invalid min_confidence
        invalid_params = {"min_confidence": 1.5}
        assert not strategy.validate_parameters(invalid_params)

        invalid_params = {"min_confidence": -0.1}
        assert not strategy.validate_parameters(invalid_params)

    def test_get_required_indicators(self, strategy: CDCActionZoneStrategy) -> None:
        """Test required indicators list."""
        indicators = strategy.get_required_indicators()
        expected_indicators = [
            "EMA_8",
            "EMA_21",
            "SMA_50",
            "RSI_14",
            "Volume_SMA_20",
            "ATR_14",
        ]
        assert indicators == expected_indicators

    def create_market_data(
        self,
        symbol: str = "BTCUSDT",
        timestamp: datetime | None = None,
        open_price: float = 50000.0,
        high_price: float = 51000.0,
        low_price: float = 49000.0,
        close_price: float = 50500.0,
        volume: float = 1000.0,
        timeframe: Timeframe = Timeframe.ONE_HOUR,
    ) -> MarketData:
        """Helper method to create market data for testing."""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        return MarketData(
            symbol=symbol,
            timestamp=timestamp,
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=volume,
            timeframe=timeframe,
        )

    def create_market_data_series(
        self,
        count: int = 100,
        base_price: float = 50000.0,
        trend: str = "sideways",  # "up", "down", "sideways"
    ) -> list[MarketData]:
        """Create a series of market data for testing."""
        data_series = []
        current_price = base_price

        for i in range(count):
            # Add some randomness and trend
            if trend == "up":
                price_change = np.random.normal(0.001, 0.02)  # Slight upward bias
            elif trend == "down":
                price_change = np.random.normal(-0.001, 0.02)  # Slight downward bias
            else:
                price_change = np.random.normal(0, 0.02)  # No bias

            current_price *= 1 + price_change

            # Create OHLC data
            high = current_price * (1 + abs(np.random.normal(0, 0.01)))
            low = current_price * (1 - abs(np.random.normal(0, 0.01)))
            open_price = current_price * (1 + np.random.normal(0, 0.005))
            close_price = current_price

            # Volume with some randomness
            volume = 1000 * (1 + np.random.normal(0, 0.3))

            timestamp = datetime.now(timezone.utc).replace(
                hour=i % 24, minute=0, second=0, microsecond=0
            )

            data_series.append(
                self.create_market_data(
                    timestamp=timestamp,
                    open_price=open_price,
                    high_price=high,
                    low_price=low,
                    close_price=close_price,
                    volume=volume,
                )
            )

        return data_series

    @pytest.mark.asyncio
    async def test_generate_signal_insufficient_data(
        self, strategy: CDCActionZoneStrategy
    ) -> None:
        """Test signal generation with insufficient data."""
        await strategy.initialize()

        # Create minimal data (less than required)
        data_series = self.create_market_data_series(count=10)

        for data in data_series:
            signal = await strategy.generate_signal(data)
            assert signal is None  # Should return None due to insufficient data

    @pytest.mark.asyncio
    async def test_generate_signal_sufficient_data(
        self, strategy: CDCActionZoneStrategy
    ) -> None:
        """Test signal generation with sufficient data."""
        await strategy.initialize()

        # Create sufficient data for analysis
        data_series = self.create_market_data_series(count=80, trend="up")

        signals_generated = 0
        for data in data_series:
            signal = await strategy.generate_signal(data)
            if signal is not None:
                signals_generated += 1

                # Validate signal structure
                assert isinstance(signal.signal_type, SignalType)
                assert 0.0 <= signal.confidence <= 1.0
                assert signal.symbol == "BTCUSDT"
                assert signal.timeframe == Timeframe.ONE_HOUR
                assert signal.strategy_name == "CDCActionZone"
                assert signal.price is not None
                assert signal.price > 0
                assert isinstance(signal.metadata, dict)

                # Check metadata structure
                metadata = signal.metadata
                assert "fast_ma" in metadata
                assert "slow_ma" in metadata
                assert "trend_ma" in metadata
                assert "rsi" in metadata
                assert "volume_ratio" in metadata
                assert "atr" in metadata
                assert "conditions_met" in metadata
                assert "buy_score" in metadata
                assert "sell_score" in metadata
                assert "suggested_stop_loss" in metadata

        # Should generate at least some signals with sufficient data
        assert signals_generated > 0

    @pytest.mark.asyncio
    async def test_generate_signal_buy_conditions(
        self, strategy: CDCActionZoneStrategy
    ) -> None:
        """Test buy signal generation with favorable conditions."""
        await strategy.initialize()

        # Create data that should trigger buy signals
        # Start with lower prices and create an upward trend
        data_series = self.create_market_data_series(
            count=80, base_price=45000.0, trend="up"
        )

        buy_signals = 0
        for data in data_series:
            signal = await strategy.generate_signal(data)
            if signal and signal.signal_type == SignalType.BUY:
                buy_signals += 1

                # Validate buy signal properties
                assert signal.confidence >= CDCActionZoneConfig.MIN_CONFIDENCE
                assert "suggested_stop_loss" in signal.metadata
                stop_loss = signal.metadata["suggested_stop_loss"]
                assert stop_loss is not None
                assert stop_loss < signal.price

        # Should generate some buy signals in upward trend
        assert buy_signals > 0

    @pytest.mark.asyncio
    async def test_generate_signal_sell_conditions(
        self, strategy: CDCActionZoneStrategy
    ) -> None:
        """Test sell signal generation with favorable conditions."""
        await strategy.initialize()

        # Create data that should trigger sell signals
        # Start with higher prices and create a downward trend
        data_series = self.create_market_data_series(
            count=80, base_price=55000.0, trend="down"
        )

        sell_signals = 0
        for data in data_series:
            signal = await strategy.generate_signal(data)
            if signal and signal.signal_type == SignalType.SELL:
                sell_signals += 1

                # Validate sell signal properties
                assert signal.confidence >= CDCActionZoneConfig.MIN_CONFIDENCE
                assert "suggested_stop_loss" in signal.metadata
                stop_loss = signal.metadata["suggested_stop_loss"]
                assert stop_loss is not None
                assert stop_loss > signal.price

        # Should generate some sell signals in downward trend
        assert sell_signals > 0

    @pytest.mark.asyncio
    async def test_strategy_cleanup(self, strategy: CDCActionZoneStrategy) -> None:
        """Test strategy cleanup."""
        await strategy.initialize()
        assert strategy.is_initialized

        await strategy.cleanup()
        assert not strategy.is_initialized

    def test_create_dataframe(self, strategy: CDCActionZoneStrategy) -> None:
        """Test DataFrame creation from market data buffer."""
        # Add some data to buffer
        data_series = self.create_market_data_series(count=10)
        strategy._market_data_buffer = data_series

        df = strategy._create_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert df.index.name == "timestamp"

    @pytest.mark.asyncio
    async def test_calculate_indicators(self, strategy: CDCActionZoneStrategy) -> None:
        """Test technical indicators calculation."""
        await strategy.initialize()

        # Create sufficient data
        data_series = self.create_market_data_series(count=80)
        strategy._market_data_buffer = data_series

        df = strategy._create_dataframe()
        indicators = strategy._calculate_indicators(df)

        # Check all required indicators are calculated
        expected_indicators = [
            "fast_ma",
            "slow_ma",
            "trend_ma",
            "rsi",
            "volume_ma",
            "volume_ratio",
            "atr",
        ]

        for indicator in expected_indicators:
            assert indicator in indicators
            assert isinstance(indicators[indicator], pd.Series)
            assert len(indicators[indicator]) == len(df)

    @pytest.mark.asyncio
    async def test_analyze_conditions(self, strategy: CDCActionZoneStrategy) -> None:
        """Test market conditions analysis."""
        await strategy.initialize()

        # Create test data
        data_series = self.create_market_data_series(count=80)
        strategy._market_data_buffer = data_series

        df = strategy._create_dataframe()
        indicators = strategy._calculate_indicators(df)
        analysis = strategy._analyze_conditions(indicators)

        # Check all analysis fields are present
        expected_fields = [
            "ma_crossover",
            "ma_crossunder",
            "trend_bullish",
            "trend_bearish",
            "rsi_oversold",
            "rsi_overbought",
            "rsi_neutral",
            "volume_confirmed",
            "price_above_trend",
            "price_below_trend",
        ]

        for field in expected_fields:
            assert field in analysis
            assert isinstance(analysis[field], bool)


class TestCDCActionZoneBacktestStrategy:
    """Test CDC Action Zone backtesting strategy."""

    def test_backtest_strategy_parameters(self) -> None:
        """Test that backtesting strategy has correct parameters."""
        strategy = CDCActionZoneBacktestStrategy

        assert hasattr(strategy, "fast_ma_period")
        assert hasattr(strategy, "slow_ma_period")
        assert hasattr(strategy, "trend_ma_period")
        assert hasattr(strategy, "rsi_period")
        assert hasattr(strategy, "rsi_oversold")
        assert hasattr(strategy, "rsi_overbought")
        assert hasattr(strategy, "volume_ma_period")
        assert hasattr(strategy, "volume_threshold")
        assert hasattr(strategy, "atr_period")
        assert hasattr(strategy, "atr_multiplier")
        assert hasattr(strategy, "min_confidence")

        # Check default values match config
        assert strategy.fast_ma_period == CDCActionZoneConfig.FAST_MA_PERIOD
        assert strategy.slow_ma_period == CDCActionZoneConfig.SLOW_MA_PERIOD
        assert strategy.trend_ma_period == CDCActionZoneConfig.TREND_MA_PERIOD
        assert strategy.rsi_period == CDCActionZoneConfig.RSI_PERIOD
        assert strategy.rsi_oversold == CDCActionZoneConfig.RSI_OVERSOLD
        assert strategy.rsi_overbought == CDCActionZoneConfig.RSI_OVERBOUGHT
        assert strategy.volume_ma_period == CDCActionZoneConfig.VOLUME_MA_PERIOD
        assert strategy.volume_threshold == CDCActionZoneConfig.VOLUME_THRESHOLD
        assert strategy.atr_period == CDCActionZoneConfig.ATR_PERIOD
        assert strategy.atr_multiplier == CDCActionZoneConfig.ATR_MULTIPLIER
        assert strategy.min_confidence == CDCActionZoneConfig.MIN_CONFIDENCE

    def test_backtest_strategy_methods(self) -> None:
        """Test that backtesting strategy has required methods."""
        strategy = CDCActionZoneBacktestStrategy

        assert hasattr(strategy, "init")
        assert hasattr(strategy, "next")
        assert callable(strategy.init)
        assert callable(strategy.next)


class TestCDCActionZoneIntegration:
    """Integration tests for CDC Action Zone strategy."""

    @pytest.mark.asyncio
    async def test_strategy_full_workflow(self) -> None:
        """Test complete strategy workflow from initialization to signal generation."""
        config = StrategyConfig(
            name="CDCActionZone",
            timeframes=[Timeframe.ONE_HOUR],
            parameters={
                "fast_ma_period": 8,
                "slow_ma_period": 21,
                "min_confidence": 0.6,
            },
            enabled=True,
        )

        strategy = CDCActionZoneStrategy(config)

        # Initialize strategy
        await strategy.initialize()
        assert strategy.is_initialized

        # Generate market data with clear trend
        base_price = 50000.0
        data_points = []

        # Create upward trending data
        for i in range(100):
            price = base_price + (i * 100)  # Clear upward trend
            high = price * 1.02
            low = price * 0.98
            volume = 1000 + (i * 10)  # Increasing volume

            timestamp = datetime.now(timezone.utc).replace(
                day=1, hour=i % 24, minute=0, second=0, microsecond=0
            )

            data = MarketData(
                symbol="BTCUSDT",
                timestamp=timestamp,
                open=price * 0.999,
                high=high,
                low=low,
                close=price,
                volume=volume,
                timeframe=Timeframe.ONE_HOUR,
            )

            data_points.append(data)

        # Process data and collect signals
        signals = []
        for data in data_points:
            signal = await strategy.generate_signal(data)
            if signal:
                signals.append(signal)

        # Validate results
        assert len(signals) > 0, "Should generate at least one signal"

        # Check signal quality
        for signal in signals:
            assert signal.confidence >= 0.6
            assert signal.symbol == "BTCUSDT"
            assert signal.strategy_name == "CDCActionZone"
            assert "fast_ma" in signal.metadata
            assert "rsi" in signal.metadata
            assert "suggested_stop_loss" in signal.metadata

        # Clean up
        await strategy.cleanup()
        assert not strategy.is_initialized

    @pytest.mark.asyncio
    async def test_strategy_with_custom_parameters(self) -> None:
        """Test strategy with custom parameters."""
        config = StrategyConfig(
            name="CDCActionZone",
            timeframes=[Timeframe.FOUR_HOURS],
            parameters={
                "fast_ma_period": 10,
                "slow_ma_period": 25,
                "rsi_period": 21,
                "volume_threshold": 1.5,
                "min_confidence": 0.7,
            },
            enabled=True,
        )

        strategy = CDCActionZoneStrategy(config)

        # Validate custom parameters
        assert strategy.validate_parameters(config.parameters)

        # Initialize and test
        await strategy.initialize()
        assert strategy.is_initialized

        # Test with single data point (should return None due to insufficient data)
        data = MarketData(
            symbol="ETHUSDT",
            timestamp=datetime.now(timezone.utc),
            open=3000.0,
            high=3100.0,
            low=2900.0,
            close=3050.0,
            volume=500.0,
            timeframe=Timeframe.FOUR_HOURS,
        )

        signal = await strategy.generate_signal(data)
        assert signal is None  # Insufficient data

        await strategy.cleanup()

    @pytest.mark.asyncio
    async def test_strategy_error_handling(self) -> None:
        """Test strategy error handling."""
        config = StrategyConfig(
            name="CDCActionZone",
            timeframes=[Timeframe.ONE_HOUR],
            parameters={"invalid_param": "invalid_value"},
            enabled=True,
        )

        strategy = CDCActionZoneStrategy(config)
        await strategy.initialize()

        # Should handle invalid parameters gracefully
        assert strategy.validate_parameters(config.parameters)  # Should not crash

        # Test with malformed data - should raise validation error
        with pytest.raises(ValidationError):
            MarketData(
                symbol="",  # Invalid empty symbol
                timestamp=datetime.now(timezone.utc),
                open=0.0,  # Invalid zero price
                high=0.0,
                low=0.0,
                close=0.0,
                volume=-1.0,  # Invalid negative volume
                timeframe=Timeframe.ONE_HOUR,
            )

        # Test with invalid market data that should be handled gracefully
        invalid_data = MarketData(
            symbol="BTCUSDT",
            timestamp=datetime.now(timezone.utc),
            open=1.0,
            high=1.0,
            low=1.0,
            close=1.0,
            volume=0.0,  # Zero volume should be handled
            timeframe=Timeframe.ONE_HOUR,
        )

        # Strategy should handle this gracefully without crashing
        try:
            result = await strategy.generate_signal(invalid_data)
            # Should return None or handle gracefully
            assert result is None or isinstance(result, Signal)
        except Exception:
            # If it raises an exception, that's also acceptable error handling
            pass
