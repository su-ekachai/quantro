"""
Tests for backtesting metrics calculation and performance analysis.

This module tests the comprehensive performance metrics calculations
including risk-adjusted returns, drawdown analysis, and trade statistics.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from app.backtesting.metrics import (
    MetricsCalculator,
    calculate_buy_hold_return,
    calculate_rolling_metrics,
)
from app.backtesting.models import TradeResult


class TestMetricsCalculator:
    """Test comprehensive metrics calculation functionality."""

    @pytest.fixture
    def sample_equity_curve(self) -> pd.Series:
        """Create sample equity curve with known characteristics."""
        # Create 252 days (1 year) of data
        dates = pd.date_range(start=datetime(2023, 1, 1), periods=252, freq="D")

        # Create equity curve with:
        # - Overall upward trend (15% annual return)
        # - Some volatility (12% annual vol)
        # - A moderate drawdown period

        np.random.seed(42)  # For reproducible results
        daily_returns = np.random.normal(0.0006, 0.012, 252)  # ~15% return, 12% vol

        # Add a moderate drawdown period (days 100-110)
        daily_returns[100:111] = np.random.normal(
            -0.005, 0.015, 11
        )  # Moderate drawdown period

        # Calculate cumulative equity
        equity_values = [10000.0]  # Starting value
        for ret in daily_returns:
            equity_values.append(equity_values[-1] * (1 + ret))

        return pd.Series(equity_values[1:], index=dates)

    @pytest.fixture
    def sample_trades_profitable(self) -> list[TradeResult]:
        """Create sample trades with known win rate and profit factor."""
        base_time = datetime(2023, 1, 1)
        trades = []

        # Create 10 trades: 7 winners, 3 losers
        # Winners: avg $100 profit
        # Losers: avg $50 loss
        # Expected profit factor: (7 * 100) / (3 * 50) = 4.67

        for i in range(7):  # Winning trades
            trades.append(
                TradeResult(
                    entry_bar=i * 10,
                    exit_bar=i * 10 + 5,
                    entry_time=base_time + timedelta(days=i * 10),
                    exit_time=base_time + timedelta(days=i * 10 + 5),
                    duration=5,
                    entry_price=100.0 + i,
                    exit_price=110.0 + i,
                    size=1.0,
                    pnl=100.0 + i * 5,  # Varying profits
                    pnl_pct=0.1,
                    return_pct=0.1,
                    is_long=True,
                )
            )

        for i in range(3):  # Losing trades
            trades.append(
                TradeResult(
                    entry_bar=(i + 7) * 10,
                    exit_bar=(i + 7) * 10 + 3,
                    entry_time=base_time + timedelta(days=(i + 7) * 10),
                    exit_time=base_time + timedelta(days=(i + 7) * 10 + 3),
                    duration=3,
                    entry_price=100.0 + i + 7,
                    exit_price=95.0 + i + 7,
                    size=1.0,
                    pnl=-(50.0 + i * 10),  # Varying losses
                    pnl_pct=-0.05,
                    return_pct=-0.05,
                    is_long=True,
                )
            )

        return trades

    def test_total_return_calculation(self, sample_equity_curve: pd.Series) -> None:
        """Test total return calculation accuracy."""
        calculator = MetricsCalculator(
            trades=[],
            equity_curve=sample_equity_curve,
            initial_cash=10000.0,
        )

        total_return = calculator._calculate_total_return()

        # Calculate expected return
        expected_return = (sample_equity_curve.iloc[-1] - 10000.0) / 10000.0

        assert abs(total_return - expected_return) < 0.0001
        assert isinstance(total_return, float)

    def test_annualized_return_calculation(
        self, sample_equity_curve: pd.Series
    ) -> None:
        """Test annualized return calculation."""
        calculator = MetricsCalculator(
            trades=[],
            equity_curve=sample_equity_curve,
            initial_cash=10000.0,
        )

        annualized_return = calculator._calculate_annualized_return()

        # Should be reasonable for our test data (around 10-20%)
        assert 0.05 <= annualized_return <= 0.25
        assert isinstance(annualized_return, float)

    def test_volatility_calculation(self, sample_equity_curve: pd.Series) -> None:
        """Test volatility calculation."""
        calculator = MetricsCalculator(
            trades=[],
            equity_curve=sample_equity_curve,
            initial_cash=10000.0,
        )

        volatility = calculator._calculate_volatility()

        # Should be around 12% (our input volatility)
        assert 0.08 <= volatility <= 0.30
        assert isinstance(volatility, float)

    def test_sharpe_ratio_calculation(self, sample_equity_curve: pd.Series) -> None:
        """Test Sharpe ratio calculation."""
        calculator = MetricsCalculator(
            trades=[],
            equity_curve=sample_equity_curve,
            initial_cash=10000.0,
            risk_free_rate=0.02,  # 2% risk-free rate
        )

        sharpe_ratio = calculator._calculate_sharpe_ratio()

        # Should be reasonable for our strategy (can be negative with drawdowns)
        assert -2.0 <= sharpe_ratio <= 3.0
        assert isinstance(sharpe_ratio, float)
        # Reasonable range for Sharpe ratio
        assert -2.0 <= sharpe_ratio <= 3.0

    def test_sortino_ratio_calculation(self, sample_equity_curve: pd.Series) -> None:
        """Test Sortino ratio calculation."""
        calculator = MetricsCalculator(
            trades=[],
            equity_curve=sample_equity_curve,
            initial_cash=10000.0,
        )

        sortino_ratio = calculator._calculate_sortino_ratio()

        # Should be higher than Sharpe ratio (only penalizes downside)
        assert isinstance(sortino_ratio, float)
        # Should be reasonable
        assert -3.0 <= sortino_ratio <= 5.0

    def test_max_drawdown_calculation(self, sample_equity_curve: pd.Series) -> None:
        """Test maximum drawdown calculation."""
        calculator = MetricsCalculator(
            trades=[],
            equity_curve=sample_equity_curve,
            initial_cash=10000.0,
        )

        max_drawdown = calculator._calculate_max_drawdown()

        # Should be negative (drawdown)
        assert max_drawdown <= 0
        assert isinstance(max_drawdown, float)
        # Should detect the drawdown we built into the data
        assert max_drawdown < -0.05  # At least 5% drawdown

    def test_drawdown_metrics_comprehensive(
        self, sample_equity_curve: pd.Series
    ) -> None:
        """Test comprehensive drawdown metrics."""
        calculator = MetricsCalculator(
            trades=[],
            equity_curve=sample_equity_curve,
            initial_cash=10000.0,
        )

        drawdown_metrics = calculator._calculate_drawdown_metrics()

        assert "max_drawdown" in drawdown_metrics
        assert "max_drawdown_duration" in drawdown_metrics
        assert "avg_drawdown" in drawdown_metrics
        assert "avg_drawdown_duration" in drawdown_metrics

        # Max drawdown should be negative
        assert drawdown_metrics["max_drawdown"] <= 0

        # Duration should be positive if there are drawdowns
        assert drawdown_metrics["max_drawdown_duration"] >= 0

        # Average drawdown should be negative or zero
        assert drawdown_metrics["avg_drawdown"] <= 0

    def test_trade_statistics_calculation(
        self, sample_trades_profitable: list[TradeResult]
    ) -> None:
        """Test trade statistics calculation with known data."""
        calculator = MetricsCalculator(
            trades=sample_trades_profitable,
            equity_curve=pd.Series(
                [10000, 11000], index=pd.date_range("2023-01-01", periods=2)
            ),
            initial_cash=10000.0,
        )

        stats = calculator._calculate_trade_statistics()

        # Verify known statistics
        assert stats["total_trades"] == 10
        assert stats["winning_trades"] == 7
        assert stats["losing_trades"] == 3
        assert abs(stats["win_rate"] - 0.7) < 0.001  # 70% win rate

        # Check profit factor calculation
        # Gross profit: 100+105+110+115+120+125+130 = 805
        # Gross loss: 50+60+70 = 180
        # Profit factor: 805/180 ≈ 4.47
        expected_profit_factor = 805 / 180
        assert abs(stats["profit_factor"] - expected_profit_factor) < 0.1

        # Check best and worst trades
        assert stats["best_trade"] == 130.0  # Last winning trade
        assert stats["worst_trade"] == -70.0  # Last losing trade

    def test_exposure_time_calculation(
        self, sample_trades_profitable: list[TradeResult]
    ) -> None:
        """Test market exposure time calculation."""
        # Create equity curve spanning the trade period
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 4, 1)  # 90 days
        dates = pd.date_range(start=start_date, end=end_date, freq="D")
        equity_curve = pd.Series(range(len(dates)), index=dates)

        calculator = MetricsCalculator(
            trades=sample_trades_profitable,
            equity_curve=equity_curve,
            initial_cash=10000.0,
        )

        exposure_time = calculator._calculate_exposure_time()

        # Should be between 0 and 1
        assert 0 <= exposure_time <= 1
        assert isinstance(exposure_time, float)

        # With our sample trades (10 trades, 5 days each on average)
        # Total trade time: 10 * 4 days average = 40 days
        # Total period: ~90 days
        # Expected exposure: ~40/90 ≈ 0.44
        assert 0.3 <= exposure_time <= 0.6

    def test_calmar_ratio_calculation(self, sample_equity_curve: pd.Series) -> None:
        """Test Calmar ratio calculation."""
        calculator = MetricsCalculator(
            trades=[],
            equity_curve=sample_equity_curve,
            initial_cash=10000.0,
        )

        calmar_ratio = calculator._calculate_calmar_ratio()

        # Should be positive for profitable strategy
        assert isinstance(calmar_ratio, float)
        # Reasonable range for Calmar ratio
        assert -5.0 <= calmar_ratio <= 10.0

    def test_comprehensive_metrics_integration(
        self,
        sample_equity_curve: pd.Series,
        sample_trades_profitable: list[TradeResult],
    ) -> None:
        """Test that all metrics integrate correctly."""
        calculator = MetricsCalculator(
            trades=sample_trades_profitable,
            equity_curve=sample_equity_curve,
            initial_cash=10000.0,
        )

        metrics = calculator.calculate_all_metrics(buy_hold_return=0.15)

        # Verify all required fields exist
        required_fields = [
            "total_return",
            "annualized_return",
            "volatility",
            "sharpe_ratio",
            "sortino_ratio",
            "calmar_ratio",
            "max_drawdown",
            "max_drawdown_duration",
            "total_trades",
            "winning_trades",
            "win_rate",
            "profit_factor",
            "best_trade",
            "worst_trade",
            "exposure_time",
            "buy_hold_return",
        ]

        for field in required_fields:
            assert hasattr(metrics, field), f"Missing field: {field}"
            value = getattr(metrics, field)
            assert value is not None, f"Field {field} is None"
            assert isinstance(value, int | float), (
                f"Field {field} has wrong type: {type(value)}"
            )

        # Verify percentage conversions
        assert 0 <= metrics.win_rate <= 100  # Should be percentage
        assert (
            metrics.total_return == metrics.total_return_pct * 100
        )  # Consistency check

        # Verify trade counts
        assert metrics.total_trades == metrics.winning_trades + metrics.losing_trades
        assert metrics.total_trades == len(
            sample_trades_profitable
        )  # From our sample data
        assert metrics.winning_trades == 7
        assert metrics.losing_trades == 3

    def test_empty_data_handling(self) -> None:
        """Test graceful handling of empty data."""
        calculator = MetricsCalculator(
            trades=[],
            equity_curve=pd.Series(dtype=float),
            initial_cash=10000.0,
        )

        metrics = calculator.calculate_all_metrics()

        # Should return default values without errors
        assert metrics.total_trades == 0
        assert metrics.winning_trades == 0
        assert metrics.losing_trades == 0
        assert metrics.win_rate == 0.0
        assert metrics.total_return == 0.0
        assert metrics.volatility == 0.0
        assert metrics.sharpe_ratio == 0.0

    def test_single_trade_handling(self) -> None:
        """Test handling of single trade scenario."""
        single_trade = [
            TradeResult(
                entry_bar=10,
                exit_bar=20,
                entry_time=datetime(2023, 1, 10),
                exit_time=datetime(2023, 1, 20),
                duration=10,
                entry_price=100.0,
                exit_price=110.0,
                size=1.0,
                pnl=10.0,
                pnl_pct=0.1,
                return_pct=0.1,
                is_long=True,
            )
        ]

        equity_curve = pd.Series(
            [10000, 10010], index=pd.date_range("2023-01-01", periods=2)
        )

        calculator = MetricsCalculator(
            trades=single_trade,
            equity_curve=equity_curve,
            initial_cash=10000.0,
        )

        metrics = calculator.calculate_all_metrics()

        assert metrics.total_trades == 1
        assert metrics.winning_trades == 1
        assert metrics.losing_trades == 0
        assert metrics.win_rate == 100.0
        assert metrics.best_trade == 10.0
        assert metrics.worst_trade == 10.0
        assert metrics.profit_factor == 0.0  # No losses to divide by


class TestBuyHoldCalculation:
    """Test buy and hold return calculation utilities."""

    def test_normal_buy_hold_calculation(self) -> None:
        """Test normal buy and hold calculation."""
        dates = pd.date_range(start=datetime(2023, 1, 1), periods=100, freq="D")

        # Create data with 20% total return
        prices = [100 + i * 0.2 for i in range(100)]  # Linear growth
        data = pd.DataFrame({"Close": prices}, index=dates)

        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 4, 10)

        buy_hold_return = calculate_buy_hold_return(data, start_date, end_date)

        # Should be approximately (119.8 - 100) / 100 = 0.198
        expected_return = (prices[-1] - prices[0]) / prices[0]
        assert abs(buy_hold_return - expected_return) < 0.001

    def test_buy_hold_with_volatility(self) -> None:
        """Test buy and hold calculation with volatile data."""
        dates = pd.date_range(start=datetime(2023, 1, 1), periods=50, freq="D")

        # Create volatile data but with overall upward trend
        np.random.seed(42)
        prices = [100]
        for _i in range(49):
            # Add trend + noise
            change = 0.5 + np.random.normal(0, 2)  # Trend + volatility
            prices.append(max(prices[-1] + change, 50))  # Prevent going too low

        data = pd.DataFrame({"Close": prices}, index=dates)

        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 2, 19)

        buy_hold_return = calculate_buy_hold_return(data, start_date, end_date)

        expected_return = (prices[-1] - prices[0]) / prices[0]
        assert abs(buy_hold_return - expected_return) < 0.001

    def test_buy_hold_negative_return(self) -> None:
        """Test buy and hold calculation with negative returns."""
        dates = pd.date_range(start=datetime(2023, 1, 1), periods=30, freq="D")

        # Create declining prices
        prices = [100 - i * 1.5 for i in range(30)]  # Declining trend
        data = pd.DataFrame({"Close": prices}, index=dates)

        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 30)

        buy_hold_return = calculate_buy_hold_return(data, start_date, end_date)

        # Should be negative
        assert buy_hold_return < 0
        expected_return = (prices[-1] - prices[0]) / prices[0]
        assert abs(buy_hold_return - expected_return) < 0.001

    def test_buy_hold_empty_data(self) -> None:
        """Test buy and hold calculation with empty data."""
        data = pd.DataFrame({"Close": []})
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 10)

        buy_hold_return = calculate_buy_hold_return(data, start_date, end_date)

        assert buy_hold_return == 0.0

    def test_buy_hold_insufficient_data(self) -> None:
        """Test buy and hold calculation with insufficient data."""
        dates = pd.date_range(start=datetime(2023, 1, 1), periods=1, freq="D")
        data = pd.DataFrame({"Close": [100]}, index=dates)

        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 10)

        buy_hold_return = calculate_buy_hold_return(data, start_date, end_date)

        assert buy_hold_return == 0.0


class TestRollingMetrics:
    """Test rolling metrics calculation."""

    def test_rolling_sharpe_calculation(self) -> None:
        """Test rolling Sharpe ratio calculation."""
        # Create 500 days of data for meaningful rolling calculations
        dates = pd.date_range(start=datetime(2023, 1, 1), periods=500, freq="D")

        np.random.seed(42)
        daily_returns = np.random.normal(0.001, 0.02, 500)  # Positive expected return

        equity_values = [10000]
        for ret in daily_returns:
            equity_values.append(equity_values[-1] * (1 + ret))

        equity_curve = pd.Series(equity_values[1:], index=dates)

        rolling_metrics = calculate_rolling_metrics(equity_curve, window_days=252)

        assert "sharpe" in rolling_metrics
        assert "volatility" in rolling_metrics
        assert "max_drawdown" in rolling_metrics

        # Check that we get reasonable values
        sharpe_values = rolling_metrics["sharpe"].dropna()
        assert len(sharpe_values) > 0

        # Sharpe ratios should be in reasonable range
        assert sharpe_values.min() > -3.0
        assert sharpe_values.max() < 5.0

    def test_rolling_metrics_empty_data(self) -> None:
        """Test rolling metrics with empty data."""
        equity_curve = pd.Series(dtype=float)

        rolling_metrics = calculate_rolling_metrics(equity_curve)

        # Should return empty dict without errors
        assert isinstance(rolling_metrics, dict)

    def test_rolling_metrics_insufficient_data(self) -> None:
        """Test rolling metrics with insufficient data."""
        dates = pd.date_range(start=datetime(2023, 1, 1), periods=10, freq="D")
        equity_curve = pd.Series(range(10), index=dates)

        rolling_metrics = calculate_rolling_metrics(equity_curve, window_days=252)

        # Should handle gracefully
        assert isinstance(rolling_metrics, dict)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_zero_initial_cash(self) -> None:
        """Test handling of zero initial cash."""
        equity_curve = pd.Series(
            [0, 100, 200], index=pd.date_range("2023-01-01", periods=3)
        )

        calculator = MetricsCalculator(
            trades=[],
            equity_curve=equity_curve,
            initial_cash=0.0,  # Edge case
        )

        # Should handle gracefully without division by zero
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            metrics = calculator.calculate_all_metrics()
            assert isinstance(metrics.total_return, float)
            assert metrics.total_return == 0.0  # Should return 0 for zero initial cash

    def test_constant_equity_curve(self) -> None:
        """Test handling of constant equity curve (no volatility)."""
        dates = pd.date_range(start=datetime(2023, 1, 1), periods=100, freq="D")
        equity_curve = pd.Series([10000] * 100, index=dates)  # Constant value

        calculator = MetricsCalculator(
            trades=[],
            equity_curve=equity_curve,
            initial_cash=10000.0,
        )

        metrics = calculator.calculate_all_metrics()

        # Should handle zero volatility gracefully
        assert metrics.total_return == 0.0
        assert metrics.volatility == 0.0
        assert metrics.sharpe_ratio == 0.0  # Should be 0 when no volatility

    def test_all_losing_trades(self) -> None:
        """Test handling of all losing trades."""
        losing_trades = [
            TradeResult(
                entry_bar=i * 10,
                exit_bar=i * 10 + 5,
                entry_time=datetime(2023, 1, 1) + timedelta(days=i * 10),
                exit_time=datetime(2023, 1, 1) + timedelta(days=i * 10 + 5),
                duration=5,
                entry_price=100.0,
                exit_price=95.0,
                size=1.0,
                pnl=-5.0,
                pnl_pct=-0.05,
                return_pct=-0.05,
                is_long=True,
            )
            for i in range(5)
        ]

        equity_curve = pd.Series(
            [10000, 9000], index=pd.date_range("2023-01-01", periods=2)
        )

        calculator = MetricsCalculator(
            trades=losing_trades,
            equity_curve=equity_curve,
            initial_cash=10000.0,
        )

        metrics = calculator.calculate_all_metrics()

        assert metrics.total_trades == 5
        assert metrics.winning_trades == 0
        assert metrics.losing_trades == 5
        assert metrics.win_rate == 0.0
        assert metrics.profit_factor == 0.0  # No profits to divide
        assert metrics.avg_win == 0.0
        assert metrics.avg_loss == -5.0


if __name__ == "__main__":
    pytest.main([__file__])
