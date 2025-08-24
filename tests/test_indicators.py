"""
Tests for technical indicators utilities.
"""

import numpy as np
import pandas as pd
import pytest

from app.utils.indicators import (
    atr,
    bollinger_bands,
    ema,
    macd,
    rsi,
    sma,
    volume_ratio,
    volume_sma,
)


class TestMovingAverages:
    """Test moving average calculations."""

    @pytest.fixture
    def sample_data(self) -> pd.Series:
        """Create sample price data for testing."""
        return pd.Series([10, 12, 11, 13, 15, 14, 16, 18, 17, 19, 20, 18, 21, 23, 22])

    def test_sma_calculation(self, sample_data: pd.Series) -> None:
        """Test Simple Moving Average calculation."""
        period = 5
        sma_result = sma(sample_data, period)

        assert isinstance(sma_result, pd.Series)
        assert len(sma_result) == len(sample_data)

        # First few values should be NaN
        assert pd.isna(sma_result.iloc[: period - 1]).all()

        # Check specific calculation
        expected_first_sma = sample_data.iloc[:period].mean()
        assert abs(sma_result.iloc[period - 1] - expected_first_sma) < 1e-10

        # Check last value
        expected_last_sma = sample_data.iloc[-period:].mean()
        assert abs(sma_result.iloc[-1] - expected_last_sma) < 1e-10

    def test_ema_calculation(self, sample_data: pd.Series) -> None:
        """Test Exponential Moving Average calculation."""
        period = 5
        ema_result = ema(sample_data, period)

        assert isinstance(ema_result, pd.Series)
        assert len(ema_result) == len(sample_data)

        # EMA should not have NaN values after first value
        assert not pd.isna(ema_result.iloc[0])

        # EMA should be different from SMA
        sma_result = sma(sample_data, period)
        # Compare where both have values
        valid_indices = ~pd.isna(sma_result)
        assert not np.allclose(ema_result[valid_indices], sma_result[valid_indices])

    def test_sma_edge_cases(self) -> None:
        """Test SMA with edge cases."""
        # Single value
        single_value = pd.Series([10])
        result = sma(single_value, 1)
        assert result.iloc[0] == 10

        # Period larger than data
        small_data = pd.Series([1, 2, 3])
        result = sma(small_data, 5)
        assert pd.isna(result).all()

        # Empty series
        empty_data = pd.Series([], dtype=float)
        result = sma(empty_data, 5)
        assert len(result) == 0

    def test_ema_edge_cases(self) -> None:
        """Test EMA with edge cases."""
        # Single value
        single_value = pd.Series([10])
        result = ema(single_value, 1)
        assert result.iloc[0] == 10

        # Empty series
        empty_data = pd.Series([], dtype=float)
        result = ema(empty_data, 5)
        assert len(result) == 0


class TestRSI:
    """Test RSI calculation."""

    @pytest.fixture
    def trending_up_data(self) -> pd.Series:
        """Create upward trending data."""
        return pd.Series([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])

    @pytest.fixture
    def trending_down_data(self) -> pd.Series:
        """Create downward trending data."""
        return pd.Series([24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10])

    @pytest.fixture
    def oscillating_data(self) -> pd.Series:
        """Create oscillating data."""
        return pd.Series([10, 15, 12, 18, 11, 16, 13, 17, 14, 19, 12, 15, 11, 16, 13])

    def test_rsi_calculation(self, oscillating_data: pd.Series) -> None:
        """Test basic RSI calculation."""
        period = 14
        rsi_result = rsi(oscillating_data, period)

        assert isinstance(rsi_result, pd.Series)
        assert len(rsi_result) == len(oscillating_data)

        # First values should be NaN until we have enough data
        assert pd.isna(rsi_result.iloc[: period - 1]).all()

        # RSI values should be between 0 and 100
        valid_rsi = rsi_result.dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()

    def test_rsi_trending_up(self, trending_up_data: pd.Series) -> None:
        """Test RSI with upward trending data."""
        rsi_result = rsi(trending_up_data, 14)

        # RSI should be high for upward trending data
        final_rsi = rsi_result.iloc[-1]
        assert final_rsi > 50  # Should be above 50 for upward trend

    def test_rsi_trending_down(self, trending_down_data: pd.Series) -> None:
        """Test RSI with downward trending data."""
        rsi_result = rsi(trending_down_data, 14)

        # RSI should be low for downward trending data
        final_rsi = rsi_result.iloc[-1]
        assert final_rsi < 50  # Should be below 50 for downward trend

    def test_rsi_different_periods(self, oscillating_data: pd.Series) -> None:
        """Test RSI with different periods."""
        rsi_14 = rsi(oscillating_data, 14)
        rsi_7 = rsi(oscillating_data, 7)

        # Shorter period should be more sensitive (more variation)
        rsi_14.dropna().std()
        rsi_7.dropna().std()

        # Compare only the overlapping valid values
        rsi_14_valid = rsi_14.dropna()
        rsi_7_valid = rsi_7.dropna()

        # Find the common index range (both have valid values)
        common_start = max(rsi_14_valid.index[0], rsi_7_valid.index[0])
        common_end = min(rsi_14_valid.index[-1], rsi_7_valid.index[-1])

        rsi_14_common = rsi_14.loc[common_start:common_end]
        rsi_7_common = rsi_7.loc[common_start:common_end]

        # Generally, shorter periods are more volatile
        # But this might not always be true, so we just check they're different
        assert not np.allclose(rsi_14_common, rsi_7_common)

    def test_rsi_edge_cases(self) -> None:
        """Test RSI with edge cases."""
        # Constant values (no change)
        constant_data = pd.Series([10] * 20)
        rsi_result = rsi(constant_data, 14)

        # RSI should be around 50 for no change (or NaN due to division by zero)
        valid_rsi = rsi_result.dropna()
        if len(valid_rsi) > 0:
            # If not NaN, should be around 50
            assert all(abs(val - 50) < 1e-6 or pd.isna(val) for val in valid_rsi)

        # Very small dataset
        small_data = pd.Series([1, 2, 3])
        rsi_result = rsi(small_data, 14)
        assert pd.isna(rsi_result).all()


class TestVolumeIndicators:
    """Test volume-related indicators."""

    @pytest.fixture
    def volume_data(self) -> pd.Series:
        """Create sample volume data."""
        return pd.Series([1000, 1200, 800, 1500, 900, 1100, 1300, 700, 1400, 1000])

    def test_volume_sma(self, volume_data: pd.Series) -> None:
        """Test volume simple moving average."""
        period = 5
        vol_sma = volume_sma(volume_data, period)

        assert isinstance(vol_sma, pd.Series)
        assert len(vol_sma) == len(volume_data)

        # First few values should be NaN
        assert pd.isna(vol_sma.iloc[: period - 1]).all()

        # Check calculation
        expected_first = volume_data.iloc[:period].mean()
        assert abs(vol_sma.iloc[period - 1] - expected_first) < 1e-10

    def test_volume_ratio(self, volume_data: pd.Series) -> None:
        """Test volume ratio calculation."""
        period = 5
        vol_sma = volume_sma(volume_data, period)
        vol_ratio = volume_ratio(volume_data, vol_sma)

        assert isinstance(vol_ratio, pd.Series)
        assert len(vol_ratio) == len(volume_data)

        # Where volume_sma is not NaN, ratio should be calculated
        valid_indices = ~pd.isna(vol_sma)
        valid_ratios = vol_ratio[valid_indices]

        # Ratios should be positive
        assert (valid_ratios > 0).all()

        # Check specific calculation
        for i in valid_indices[valid_indices].index:
            expected_ratio = volume_data.iloc[i] / vol_sma.iloc[i]
            assert abs(vol_ratio.iloc[i] - expected_ratio) < 1e-10


class TestATR:
    """Test Average True Range calculation."""

    @pytest.fixture
    def ohlc_data(self) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Create sample OHLC data."""
        high = pd.Series([12, 14, 13, 16, 15, 17, 19, 18, 20, 19])
        low = pd.Series([10, 11, 10, 13, 12, 14, 16, 15, 17, 16])
        close = pd.Series([11, 13, 12, 15, 14, 16, 18, 17, 19, 18])
        return high, low, close

    def test_atr_calculation(
        self, ohlc_data: tuple[pd.Series, pd.Series, pd.Series]
    ) -> None:
        """Test ATR calculation."""
        high, low, close = ohlc_data
        period = 5

        atr_result = atr(high, low, close, period)

        assert isinstance(atr_result, pd.Series)
        assert len(atr_result) == len(high)

        # First few values should be NaN
        assert pd.isna(atr_result.iloc[: period - 1]).all()

        # ATR values should be positive
        valid_atr = atr_result.dropna()
        assert (valid_atr > 0).all()

    def test_atr_edge_cases(self) -> None:
        """Test ATR with edge cases."""
        # Single bar
        high = pd.Series([12])
        low = pd.Series([10])
        close = pd.Series([11])

        atr_result = atr(high, low, close, 1)
        assert atr_result.iloc[0] == 2  # high - low = 12 - 10 = 2

        # Equal high and low (no volatility)
        high = pd.Series([10] * 10)
        low = pd.Series([10] * 10)
        close = pd.Series([10] * 10)

        atr_result = atr(high, low, close, 5)
        valid_atr = atr_result.dropna()
        assert (valid_atr == 0).all()


class TestBollingerBands:
    """Test Bollinger Bands calculation."""

    @pytest.fixture
    def price_data(self) -> pd.Series:
        """Create sample price data."""
        return pd.Series([10, 12, 11, 13, 15, 14, 16, 18, 17, 19, 20, 18, 21, 23, 22])

    def test_bollinger_bands_calculation(self, price_data: pd.Series) -> None:
        """Test Bollinger Bands calculation."""
        period = 10
        std_dev = 2.0

        upper, middle, lower = bollinger_bands(price_data, period, std_dev)

        # Check types and lengths
        assert isinstance(upper, pd.Series)
        assert isinstance(middle, pd.Series)
        assert isinstance(lower, pd.Series)
        assert len(upper) == len(price_data)
        assert len(middle) == len(price_data)
        assert len(lower) == len(price_data)

        # Middle band should be SMA
        expected_middle = sma(price_data, period)
        pd.testing.assert_series_equal(middle, expected_middle)

        # Upper band should be above middle, lower should be below
        valid_indices = ~pd.isna(middle)
        assert (upper[valid_indices] > middle[valid_indices]).all()
        assert (lower[valid_indices] < middle[valid_indices]).all()

    def test_bollinger_bands_different_std(self, price_data: pd.Series) -> None:
        """Test Bollinger Bands with different standard deviations."""
        period = 10

        upper_1, middle_1, lower_1 = bollinger_bands(price_data, period, 1.0)
        upper_2, middle_2, lower_2 = bollinger_bands(price_data, period, 2.0)

        # Middle bands should be the same
        pd.testing.assert_series_equal(middle_1, middle_2)

        # Wider std_dev should create wider bands
        valid_indices = ~pd.isna(middle_1)
        assert (upper_2[valid_indices] > upper_1[valid_indices]).all()
        assert (lower_2[valid_indices] < lower_1[valid_indices]).all()


class TestMACD:
    """Test MACD calculation."""

    @pytest.fixture
    def price_data(self) -> pd.Series:
        """Create sample price data for MACD."""
        # Create more data points for MACD calculation
        np.random.seed(42)
        base_price = 100
        returns = np.random.normal(0.001, 0.02, 50)
        prices = [base_price]

        for ret in returns:
            prices.append(prices[-1] * (1 + ret))

        return pd.Series(prices)

    def test_macd_calculation(self, price_data: pd.Series) -> None:
        """Test MACD calculation."""
        fast_period = 12
        slow_period = 26
        signal_period = 9

        macd_line, signal_line, histogram = macd(
            price_data, fast_period, slow_period, signal_period
        )

        # Check types and lengths
        assert isinstance(macd_line, pd.Series)
        assert isinstance(signal_line, pd.Series)
        assert isinstance(histogram, pd.Series)
        assert len(macd_line) == len(price_data)
        assert len(signal_line) == len(price_data)
        assert len(histogram) == len(price_data)

        # MACD line should be difference of EMAs
        fast_ema = ema(price_data, fast_period)
        slow_ema = ema(price_data, slow_period)
        expected_macd = fast_ema - slow_ema
        pd.testing.assert_series_equal(macd_line, expected_macd)

        # Signal line should be EMA of MACD line
        expected_signal = ema(macd_line, signal_period)
        pd.testing.assert_series_equal(signal_line, expected_signal)

        # Histogram should be difference
        expected_histogram = macd_line - signal_line
        pd.testing.assert_series_equal(histogram, expected_histogram)

    def test_macd_default_parameters(self, price_data: pd.Series) -> None:
        """Test MACD with default parameters."""
        macd_line, signal_line, histogram = macd(price_data)

        # Should work with default parameters
        assert len(macd_line) == len(price_data)
        assert len(signal_line) == len(price_data)
        assert len(histogram) == len(price_data)


class TestIndicatorsIntegration:
    """Integration tests for indicators working together."""

    def test_indicators_with_real_data_pattern(self) -> None:
        """Test indicators with realistic market data pattern."""
        # Create realistic price data
        np.random.seed(42)

        # Start with base price and create trending data
        base_price = 50000
        trend_data = []
        current_price = base_price

        # Create upward trend with noise
        for i in range(100):
            # Add trend component
            trend = 0.001 if i < 50 else -0.001  # Up then down
            # Add noise
            noise = np.random.normal(0, 0.02)
            # Combine
            change = trend + noise
            current_price *= 1 + change
            trend_data.append(current_price)

        price_series = pd.Series(trend_data)

        # Calculate multiple indicators
        sma_20 = sma(price_series, 20)
        ema_12 = ema(price_series, 12)
        rsi_14 = rsi(price_series, 14)

        # Create volume data
        volume_series = pd.Series(np.random.lognormal(7, 0.3, 100))
        vol_sma_20 = volume_sma(volume_series, 20)
        vol_ratio = volume_ratio(volume_series, vol_sma_20)

        # Create OHLC data from close prices
        high_series = price_series * (1 + np.abs(np.random.normal(0, 0.01, 100)))
        low_series = price_series * (1 - np.abs(np.random.normal(0, 0.01, 100)))
        atr_14 = atr(high_series, low_series, price_series, 14)

        # Verify all indicators are calculated
        assert len(sma_20) == 100
        assert len(ema_12) == 100
        assert len(rsi_14) == 100
        assert len(vol_ratio) == 100
        assert len(atr_14) == 100

        # Check that indicators have reasonable values
        valid_rsi = rsi_14.dropna()
        assert (valid_rsi >= 0).all() and (valid_rsi <= 100).all()

        valid_atr = atr_14.dropna()
        assert (valid_atr > 0).all()

        valid_vol_ratio = vol_ratio.dropna()
        assert (valid_vol_ratio > 0).all()

    def test_indicators_consistency(self) -> None:
        """Test that indicators produce consistent results."""
        # Create deterministic data
        price_data = pd.Series([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

        # Calculate same indicator multiple times
        sma1 = sma(price_data, 5)
        sma2 = sma(price_data, 5)

        # Should be identical
        pd.testing.assert_series_equal(sma1, sma2)

        # EMA should also be consistent
        ema1 = ema(price_data, 5)
        ema2 = ema(price_data, 5)

        pd.testing.assert_series_equal(ema1, ema2)

        # RSI should be consistent
        rsi1 = rsi(price_data, 5)
        rsi2 = rsi(price_data, 5)

        pd.testing.assert_series_equal(rsi1, rsi2)
