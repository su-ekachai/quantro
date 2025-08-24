"""
Technical indicators utilities for trading strategies.

This module provides common technical indicators used in trading strategies,
including moving averages, RSI, and volume analysis.
"""

from __future__ import annotations

import pandas as pd


def sma(data: pd.Series, period: int) -> pd.Series:
    """
    Calculate Simple Moving Average.

    Args:
        data: Price data series
        period: Number of periods for the moving average

    Returns:
        Simple moving average series
    """
    return data.rolling(window=period, min_periods=period).mean()


def ema(data: pd.Series, period: int) -> pd.Series:
    """
    Calculate Exponential Moving Average.

    Args:
        data: Price data series
        period: Number of periods for the moving average

    Returns:
        Exponential moving average series
    """
    return data.ewm(span=period, adjust=False).mean()


def rsi(data: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index.

    Args:
        data: Price data series (typically close prices)
        period: Number of periods for RSI calculation (default: 14)

    Returns:
        RSI series with values between 0 and 100
    """
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    # Handle subsequent calculations with exponential smoothing
    for i in range(period, len(data)):
        avg_gain.iloc[i] = (avg_gain.iloc[i - 1] * (period - 1) + gain.iloc[i]) / period
        avg_loss.iloc[i] = (avg_loss.iloc[i - 1] * (period - 1) + loss.iloc[i]) / period

    rs = avg_gain / avg_loss
    rsi_values = 100 - (100 / (1 + rs))

    return rsi_values


def volume_sma(volume: pd.Series, period: int) -> pd.Series:
    """
    Calculate Simple Moving Average of volume.

    Args:
        volume: Volume data series
        period: Number of periods for the moving average

    Returns:
        Volume moving average series
    """
    return volume.rolling(window=period, min_periods=period).mean()


def volume_ratio(volume: pd.Series, volume_ma: pd.Series) -> pd.Series:
    """
    Calculate volume ratio (current volume / average volume).

    Args:
        volume: Current volume series
        volume_ma: Volume moving average series

    Returns:
        Volume ratio series
    """
    return volume / volume_ma


def atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """
    Calculate Average True Range.

    Args:
        high: High price series
        low: Low price series
        close: Close price series
        period: Number of periods for ATR calculation

    Returns:
        ATR series
    """
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    return true_range.rolling(window=period, min_periods=period).mean()


def bollinger_bands(
    data: pd.Series, period: int = 20, std_dev: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands.

    Args:
        data: Price data series
        period: Number of periods for moving average
        std_dev: Number of standard deviations for bands

    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    """
    middle_band = sma(data, period)
    std = data.rolling(window=period, min_periods=period).std()

    upper_band = middle_band + (std * std_dev)
    lower_band = middle_band - (std * std_dev)

    return upper_band, middle_band, lower_band


def macd(
    data: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate MACD (Moving Average Convergence Divergence).

    Args:
        data: Price data series
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        signal_period: Signal line EMA period

    Returns:
        Tuple of (macd_line, signal_line, histogram)
    """
    fast_ema = ema(data, fast_period)
    slow_ema = ema(data, slow_period)

    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal_period)
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram
