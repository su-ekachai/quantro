"""
CDC Action Zone Trading Strategy Implementation.

This module implements the CDC Action Zone strategy, which combines multiple
technical indicators to identify high-probability trading opportunities.
The strategy uses moving averages, RSI, and volume analysis to generate
buy/sell signals with confidence scoring.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from backtesting import Strategy as BacktestStrategy
from loguru import logger

from app.strategies.base import (
    IStrategy,
    MarketData,
    Signal,
    SignalType,
    StrategyConfig,
)
from app.utils.indicators import atr, ema, rsi, sma, volume_ratio, volume_sma


class CDCActionZoneConfig:
    """Configuration parameters for CDC Action Zone strategy."""

    # Moving Average parameters
    FAST_MA_PERIOD: int = 8
    SLOW_MA_PERIOD: int = 21
    TREND_MA_PERIOD: int = 50

    # RSI parameters
    RSI_PERIOD: int = 14
    RSI_OVERSOLD: float = 30.0
    RSI_OVERBOUGHT: float = 70.0
    RSI_NEUTRAL_LOW: float = 40.0
    RSI_NEUTRAL_HIGH: float = 60.0

    # Volume parameters
    VOLUME_MA_PERIOD: int = 20
    VOLUME_THRESHOLD: float = 1.2  # Volume must be 20% above average

    # ATR parameters for stop loss
    ATR_PERIOD: int = 14
    ATR_MULTIPLIER: float = 2.0

    # Signal confidence thresholds
    MIN_CONFIDENCE: float = 0.6
    HIGH_CONFIDENCE: float = 0.8


class CDCActionZoneStrategy(IStrategy):
    """
    CDC Action Zone Trading Strategy.

    This strategy identifies action zones where multiple technical indicators
    align to suggest high-probability trading opportunities. It combines:

    1. Moving Average Crossovers (Fast MA vs Slow MA)
    2. Trend Confirmation (Price vs Trend MA)
    3. RSI Momentum Analysis
    4. Volume Confirmation
    5. Confidence Scoring based on indicator alignment
    """

    def __init__(self, config: StrategyConfig) -> None:
        """Initialize the CDC Action Zone strategy."""
        super().__init__(config)
        self._market_data_buffer: list[MarketData] = []
        self._min_data_points = (
            max(
                CDCActionZoneConfig.TREND_MA_PERIOD,
                CDCActionZoneConfig.VOLUME_MA_PERIOD,
                CDCActionZoneConfig.ATR_PERIOD,
            )
            + 2
        )  # Minimal buffer for indicator calculations

    async def initialize(self) -> None:
        """Initialize the strategy and validate parameters."""
        logger.info(f"Initializing CDC Action Zone strategy: {self.name}")

        # Validate strategy parameters
        if not self.validate_parameters(self.config.parameters):
            raise ValueError("Invalid strategy parameters")

        self._initialized = True
        logger.info(f"CDC Action Zone strategy initialized: {self.name}")

    async def generate_signal(self, data: MarketData) -> Signal | None:
        """
        Generate trading signal based on CDC Action Zone analysis.

        Args:
            data: Current market data

        Returns:
            Signal if conditions are met, None otherwise
        """
        if not self.is_initialized:
            logger.warning(f"Strategy {self.name} not initialized")
            return None

        # Add data to buffer
        self._market_data_buffer.append(data)

        # Keep only necessary data points
        if len(self._market_data_buffer) > self._min_data_points * 2:
            self._market_data_buffer = self._market_data_buffer[
                -self._min_data_points :
            ]

        # Need minimum data points for analysis
        if len(self._market_data_buffer) < self._min_data_points:
            logger.debug(
                f"Insufficient data points: "
                f"{len(self._market_data_buffer)}/{self._min_data_points}"
            )
            return None

        try:
            # Convert buffer to DataFrame for analysis
            df = self._create_dataframe()

            # Calculate technical indicators
            indicators = self._calculate_indicators(df)

            # Analyze current market conditions
            analysis = self._analyze_conditions(indicators)

            # Generate signal based on analysis
            signal = self._generate_signal_from_analysis(data, analysis, indicators)

            if signal:
                logger.info(
                    f"Generated {signal.signal_type} signal for {signal.symbol} "
                    f"with confidence {signal.confidence:.2f}"
                )
            else:
                logger.debug(
                    f"No signal generated for {data.symbol} - analysis: {analysis}"
                )

            return signal

        except Exception as e:
            logger.error(f"Error generating signal for {data.symbol}: {e}")
            return None

    def _create_dataframe(self) -> pd.DataFrame:
        """Create DataFrame from market data buffer."""
        data_dict = {
            "timestamp": [d.timestamp for d in self._market_data_buffer],
            "open": [float(d.open) for d in self._market_data_buffer],
            "high": [float(d.high) for d in self._market_data_buffer],
            "low": [float(d.low) for d in self._market_data_buffer],
            "close": [float(d.close) for d in self._market_data_buffer],
            "volume": [float(d.volume) for d in self._market_data_buffer],
        }

        df = pd.DataFrame(data_dict)
        df.set_index("timestamp", inplace=True)
        return df

    def _calculate_indicators(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        """Calculate all required technical indicators."""
        indicators = {}

        # Moving Averages
        indicators["fast_ma"] = ema(df["close"], CDCActionZoneConfig.FAST_MA_PERIOD)
        indicators["slow_ma"] = ema(df["close"], CDCActionZoneConfig.SLOW_MA_PERIOD)
        indicators["trend_ma"] = sma(df["close"], CDCActionZoneConfig.TREND_MA_PERIOD)

        # RSI
        indicators["rsi"] = rsi(df["close"], CDCActionZoneConfig.RSI_PERIOD)

        # Volume indicators
        indicators["volume_ma"] = volume_sma(
            df["volume"], CDCActionZoneConfig.VOLUME_MA_PERIOD
        )
        indicators["volume_ratio"] = volume_ratio(df["volume"], indicators["volume_ma"])

        # ATR for stop loss calculation
        indicators["atr"] = atr(
            df["high"], df["low"], df["close"], CDCActionZoneConfig.ATR_PERIOD
        )

        return indicators

    def _analyze_conditions(self, indicators: dict[str, pd.Series]) -> dict[str, Any]:
        """Analyze current market conditions based on indicators."""
        current_idx = -1  # Latest data point
        prev_idx = -2  # Previous data point

        analysis = {
            "ma_crossover": False,
            "ma_crossunder": False,
            "trend_bullish": False,
            "trend_bearish": False,
            "rsi_oversold": False,
            "rsi_overbought": False,
            "rsi_neutral": False,
            "volume_confirmed": False,
            "price_above_trend": False,
            "price_below_trend": False,
        }

        # Moving Average Analysis
        fast_ma_current = indicators["fast_ma"].iloc[current_idx]
        fast_ma_prev = indicators["fast_ma"].iloc[prev_idx]
        slow_ma_current = indicators["slow_ma"].iloc[current_idx]
        slow_ma_prev = indicators["slow_ma"].iloc[prev_idx]

        # Check for crossovers
        if fast_ma_prev <= slow_ma_prev and fast_ma_current > slow_ma_current:
            analysis["ma_crossover"] = True
        elif fast_ma_prev >= slow_ma_prev and fast_ma_current < slow_ma_current:
            analysis["ma_crossunder"] = True

        # Trend analysis
        trend_ma_current = indicators["trend_ma"].iloc[current_idx]
        current_price = indicators["fast_ma"].iloc[
            current_idx
        ]  # Use fast MA as price proxy

        if current_price > trend_ma_current:
            analysis["trend_bullish"] = True
            analysis["price_above_trend"] = True
        else:
            analysis["trend_bearish"] = True
            analysis["price_below_trend"] = True

        # RSI Analysis
        rsi_current = indicators["rsi"].iloc[current_idx]

        if rsi_current <= CDCActionZoneConfig.RSI_OVERSOLD:
            analysis["rsi_oversold"] = True
        elif rsi_current >= CDCActionZoneConfig.RSI_OVERBOUGHT:
            analysis["rsi_overbought"] = True
        elif (
            CDCActionZoneConfig.RSI_NEUTRAL_LOW
            <= rsi_current
            <= CDCActionZoneConfig.RSI_NEUTRAL_HIGH
        ):
            analysis["rsi_neutral"] = True

        # Volume Analysis
        volume_ratio_current = indicators["volume_ratio"].iloc[current_idx]
        if volume_ratio_current >= CDCActionZoneConfig.VOLUME_THRESHOLD:
            analysis["volume_confirmed"] = True

        return analysis

    def _generate_signal_from_analysis(
        self,
        data: MarketData,
        analysis: dict[str, Any],
        indicators: dict[str, pd.Series],
    ) -> Signal | None:
        """Generate trading signal based on analysis results."""

        # Buy Signal Conditions (more flexible scoring)
        buy_score = 0.0

        # MA crossover is strong signal
        if analysis["ma_crossover"]:
            buy_score += 0.4

        # Trend alignment is important
        if analysis["trend_bullish"]:
            buy_score += 0.3

        # RSI conditions - be more flexible for trending markets
        if analysis["rsi_oversold"]:
            buy_score += 0.3  # Strong signal for oversold
        elif analysis["rsi_neutral"]:
            buy_score += 0.2  # Good signal for neutral
        elif analysis["rsi_overbought"]:
            # In strong uptrends, overbought can continue - don't penalize too much
            buy_score += 0.1

        # Volume confirmation - make it less strict
        if analysis["volume_confirmed"]:
            buy_score += 0.2  # Bonus for high volume
        else:
            buy_score += 0.05  # Small bonus even for normal volume

        # Additional scoring for strong trends (fast MA above slow MA)
        fast_ma_current = indicators["fast_ma"].iloc[-1]
        slow_ma_current = indicators["slow_ma"].iloc[-1]
        if fast_ma_current > slow_ma_current and analysis["trend_bullish"]:
            buy_score += 0.15  # Increased bonus for aligned trend

        # Normalize buy score
        buy_score = min(buy_score, 1.0)

        # Sell Signal Conditions (more flexible scoring)
        sell_score = 0.0

        # MA crossunder is strong signal
        if analysis["ma_crossunder"]:
            sell_score += 0.4

        # Trend alignment is important
        if analysis["trend_bearish"]:
            sell_score += 0.3

        # RSI conditions - be more flexible for trending markets
        if analysis["rsi_overbought"]:
            sell_score += 0.3  # Strong signal for overbought
        elif analysis["rsi_neutral"]:
            sell_score += 0.2  # Good signal for neutral
        elif analysis["rsi_oversold"]:
            # In strong downtrends, oversold can continue
            sell_score += 0.1

        # Volume confirmation - make it less strict
        if analysis["volume_confirmed"]:
            sell_score += 0.2  # Bonus for high volume
        else:
            sell_score += 0.05  # Small bonus even for normal volume

        # Additional scoring for strong trends (fast MA below slow MA)
        if fast_ma_current < slow_ma_current and analysis["trend_bearish"]:
            sell_score += 0.15  # Increased bonus for aligned trend

        # Normalize sell score
        sell_score = min(sell_score, 1.0)

        signal_type = None
        confidence = 0.0

        # Get current parameter values (allow override from config)
        min_confidence = self.config.parameters.get(
            "min_confidence", CDCActionZoneConfig.MIN_CONFIDENCE
        )

        logger.debug(
            f"Signal scores - Buy: {buy_score:.3f}, Sell: {sell_score:.3f}, "
            f"Min confidence: {min_confidence}"
        )

        if buy_score >= min_confidence and buy_score > sell_score:
            signal_type = SignalType.BUY
            confidence = buy_score
        elif sell_score >= min_confidence and sell_score > buy_score:
            signal_type = SignalType.SELL
            confidence = sell_score

        if signal_type is None:
            logger.debug(
                f"No signal: buy_score={buy_score:.3f}, sell_score={sell_score:.3f}"
            )
            logger.debug(
                f"No signal generated for {data.symbol} - analysis: {analysis}"
            )
            return None

        # Create signal metadata
        metadata = {
            "fast_ma": float(indicators["fast_ma"].iloc[-1]),
            "slow_ma": float(indicators["slow_ma"].iloc[-1]),
            "trend_ma": float(indicators["trend_ma"].iloc[-1]),
            "rsi": float(indicators["rsi"].iloc[-1]),
            "volume_ratio": float(indicators["volume_ratio"].iloc[-1]),
            "atr": float(indicators["atr"].iloc[-1]),
            "conditions_met": {
                "ma_crossover": analysis["ma_crossover"],
                "ma_crossunder": analysis["ma_crossunder"],
                "trend_bullish": analysis["trend_bullish"],
                "trend_bearish": analysis["trend_bearish"],
                "rsi_oversold": analysis["rsi_oversold"],
                "rsi_overbought": analysis["rsi_overbought"],
                "rsi_neutral": analysis["rsi_neutral"],
                "volume_confirmed": analysis["volume_confirmed"],
            },
            "buy_score": buy_score,
            "sell_score": sell_score,
        }

        # Add stop loss suggestion based on ATR
        atr_value = indicators["atr"].iloc[-1]
        if signal_type == SignalType.BUY:
            metadata["suggested_stop_loss"] = float(data.close) - (
                atr_value * CDCActionZoneConfig.ATR_MULTIPLIER
            )
        else:
            metadata["suggested_stop_loss"] = float(data.close) + (
                atr_value * CDCActionZoneConfig.ATR_MULTIPLIER
            )

        return Signal(
            signal_type=signal_type,
            confidence=confidence,
            timestamp=data.timestamp,
            symbol=data.symbol,
            timeframe=data.timeframe,
            strategy_name=self.name,
            price=float(data.close),
            metadata=metadata,
        )

    def validate_parameters(self, params: dict[str, Any]) -> bool:
        """
        Validate strategy-specific parameters.

        Args:
            params: Dictionary of parameters to validate

        Returns:
            True if parameters are valid, False otherwise
        """
        try:
            # Check for required parameters and their types/ranges
            if "fast_ma_period" in params and (
                not isinstance(params["fast_ma_period"], int)
                or params["fast_ma_period"] < 1
            ):
                return False

            if "slow_ma_period" in params and (
                not isinstance(params["slow_ma_period"], int)
                or params["slow_ma_period"] < 1
            ):
                return False

            if "rsi_period" in params and (
                not isinstance(params["rsi_period"], int) or params["rsi_period"] < 1
            ):
                return False

            if "volume_threshold" in params and (
                not isinstance(params["volume_threshold"], int | float)
                or params["volume_threshold"] <= 0
            ):
                return False

            return not (
                "min_confidence" in params
                and (
                    not isinstance(params["min_confidence"], int | float)
                    or not 0 <= params["min_confidence"] <= 1
                )
            )

        except Exception as e:
            logger.error(f"Parameter validation error: {e}")
            return False

    def get_required_indicators(self) -> list[str]:
        """
        Get list of required technical indicators.

        Returns:
            List of indicator names required by this strategy
        """
        return [
            "EMA_8",  # Fast moving average
            "EMA_21",  # Slow moving average
            "SMA_50",  # Trend moving average
            "RSI_14",  # Relative Strength Index
            "Volume_SMA_20",  # Volume moving average
            "ATR_14",  # Average True Range
        ]


class CDCActionZoneBacktestStrategy(BacktestStrategy):
    """
    CDC Action Zone strategy adapted for backtesting.py framework.

    This class adapts the CDC Action Zone strategy for use with the
    backtesting.py library, allowing for comprehensive backtesting
    and performance analysis.
    """

    # Strategy parameters (can be optimized)
    fast_ma_period = CDCActionZoneConfig.FAST_MA_PERIOD
    slow_ma_period = CDCActionZoneConfig.SLOW_MA_PERIOD
    trend_ma_period = CDCActionZoneConfig.TREND_MA_PERIOD
    rsi_period = CDCActionZoneConfig.RSI_PERIOD
    rsi_oversold = CDCActionZoneConfig.RSI_OVERSOLD
    rsi_overbought = CDCActionZoneConfig.RSI_OVERBOUGHT
    volume_ma_period = CDCActionZoneConfig.VOLUME_MA_PERIOD
    volume_threshold = CDCActionZoneConfig.VOLUME_THRESHOLD
    atr_period = CDCActionZoneConfig.ATR_PERIOD
    atr_multiplier = CDCActionZoneConfig.ATR_MULTIPLIER
    min_confidence = CDCActionZoneConfig.MIN_CONFIDENCE

    def init(self) -> None:
        """Initialize indicators for backtesting."""
        # Moving Averages
        self.fast_ma = self.I(ema, self.data.Close, self.fast_ma_period)
        self.slow_ma = self.I(ema, self.data.Close, self.slow_ma_period)
        self.trend_ma = self.I(sma, self.data.Close, self.trend_ma_period)

        # RSI
        self.rsi_values = self.I(rsi, self.data.Close, self.rsi_period)

        # Volume indicators
        self.volume_ma = self.I(volume_sma, self.data.Volume, self.volume_ma_period)
        self.volume_ratio_values = self.I(
            volume_ratio, self.data.Volume, self.volume_ma
        )

        # ATR
        self.atr_values = self.I(
            atr, self.data.High, self.data.Low, self.data.Close, self.atr_period
        )

    def next(self) -> None:
        """Execute strategy logic for each bar."""
        # Skip if not enough data
        if len(self.data) < max(
            self.trend_ma_period, self.volume_ma_period, self.atr_period
        ):
            return

        # Current values
        fast_ma_current = self.fast_ma[-1]
        fast_ma_prev = self.fast_ma[-2] if len(self.fast_ma) > 1 else fast_ma_current
        slow_ma_current = self.slow_ma[-1]
        slow_ma_prev = self.slow_ma[-2] if len(self.slow_ma) > 1 else slow_ma_current

        trend_ma_current = self.trend_ma[-1]
        rsi_current = self.rsi_values[-1]
        volume_ratio_current = self.volume_ratio_values[-1]
        atr_current = self.atr_values[-1]
        current_price = self.data.Close[-1]

        # Analyze conditions
        ma_crossover = (
            fast_ma_prev <= slow_ma_prev and fast_ma_current > slow_ma_current
        )
        ma_crossunder = (
            fast_ma_prev >= slow_ma_prev and fast_ma_current < slow_ma_current
        )
        trend_bullish = current_price > trend_ma_current
        trend_bearish = current_price <= trend_ma_current
        rsi_oversold = rsi_current <= self.rsi_oversold
        rsi_overbought = rsi_current >= self.rsi_overbought
        rsi_neutral = self.rsi_oversold < rsi_current < self.rsi_overbought
        volume_confirmed = volume_ratio_current >= self.volume_threshold

        # Buy conditions
        buy_conditions = [
            ma_crossover,
            trend_bullish,
            rsi_oversold or rsi_neutral,
            volume_confirmed,
        ]

        # Sell conditions
        sell_conditions = [
            ma_crossunder,
            trend_bearish,
            rsi_overbought or rsi_neutral,
            volume_confirmed,
        ]

        buy_score = sum(buy_conditions) / len(buy_conditions)
        sell_score = sum(sell_conditions) / len(sell_conditions)

        # Execute trades based on signals
        if not self.position:
            # Enter long position
            if buy_score >= self.min_confidence and buy_score > sell_score:
                stop_loss = current_price - (atr_current * self.atr_multiplier)
                self.buy(sl=stop_loss)

            # Enter short position
            elif sell_score >= self.min_confidence and sell_score > buy_score:
                stop_loss = current_price + (atr_current * self.atr_multiplier)
                self.sell(sl=stop_loss)

        else:
            # Exit long position
            if (
                self.position.is_long
                and (ma_crossunder or rsi_overbought)
                or self.position.is_short
                and (ma_crossover or rsi_oversold)
            ):
                self.position.close()
