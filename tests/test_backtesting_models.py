"""
Tests for backtesting models and data validation.

This module tests the Pydantic models used for backtesting configuration
and results, ensuring proper validation and data integrity.
"""

from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from app.backtesting.models import (
    BacktestConfig,
    BacktestResult,
    PerformanceMetrics,
    TradeResult,
)


class TestBacktestConfig:
    """Test BacktestConfig model validation and functionality."""

    def test_valid_config_creation(self) -> None:
        """Test creation of valid backtest configuration."""
        config = BacktestConfig(
            strategy_name="Test Strategy",
            strategy_class="app.strategies.test.TestStrategy",
            symbol="BTCUSD",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            initial_cash=10000.0,
            commission=0.001,
        )

        assert config.strategy_name == "Test Strategy"
        assert config.symbol == "BTCUSD"
        assert config.initial_cash == 10000.0
        assert config.commission == 0.001
        assert config.timeframe == "1d"  # Default value
        assert config.margin == 1.0  # Default value
        assert config.trade_on_open is False  # Default value

    def test_symbol_validation_and_cleaning(self) -> None:
        """Test symbol validation and automatic cleaning."""
        # Test whitespace removal and uppercasing
        config = BacktestConfig(
            strategy_name="Test",
            strategy_class="test.Strategy",
            symbol="  btcusd  ",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
        )
        assert config.symbol == "BTCUSD"

        # Test empty symbol validation
        with pytest.raises(ValidationError, match="Symbol cannot be empty"):
            BacktestConfig(
                strategy_name="Test",
                strategy_class="test.Strategy",
                symbol="",
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2023, 12, 31),
            )

        # Test whitespace-only symbol
        with pytest.raises(ValidationError, match="Symbol cannot be empty"):
            BacktestConfig(
                strategy_name="Test",
                strategy_class="test.Strategy",
                symbol="   ",
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2023, 12, 31),
            )

    def test_strategy_name_validation(self) -> None:
        """Test strategy name validation."""
        # Valid strategy name
        config = BacktestConfig(
            strategy_name="  Valid Strategy  ",
            strategy_class="test.Strategy",
            symbol="BTCUSD",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
        )
        assert config.strategy_name == "Valid Strategy"  # Trimmed

        # Empty strategy name
        with pytest.raises(ValidationError, match="Strategy name cannot be empty"):
            BacktestConfig(
                strategy_name="",
                strategy_class="test.Strategy",
                symbol="BTCUSD",
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2023, 12, 31),
            )

    def test_date_range_validation(self) -> None:
        """Test date range validation."""
        # Valid date range
        config = BacktestConfig(
            strategy_name="Test",
            strategy_class="test.Strategy",
            symbol="BTCUSD",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
        )
        assert config.start_date < config.end_date

        # Invalid date range (end before start)
        with pytest.raises(ValidationError, match="End date must be after start date"):
            BacktestConfig(
                strategy_name="Test",
                strategy_class="test.Strategy",
                symbol="BTCUSD",
                start_date=datetime(2023, 12, 31),
                end_date=datetime(2023, 1, 1),
            )

        # Same start and end date
        with pytest.raises(ValidationError, match="End date must be after start date"):
            BacktestConfig(
                strategy_name="Test",
                strategy_class="test.Strategy",
                symbol="BTCUSD",
                start_date=datetime(2023, 6, 15),
                end_date=datetime(2023, 6, 15),
            )

    def test_financial_parameter_validation(self) -> None:
        """Test validation of financial parameters."""
        # Valid parameters
        config = BacktestConfig(
            strategy_name="Test",
            strategy_class="test.Strategy",
            symbol="BTCUSD",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            initial_cash=10000.0,
            commission=0.001,
            margin=2.0,
        )
        assert config.initial_cash == 10000.0
        assert config.commission == 0.001
        assert config.margin == 2.0

        # Invalid initial cash (negative)
        with pytest.raises(ValidationError):
            BacktestConfig(
                strategy_name="Test",
                strategy_class="test.Strategy",
                symbol="BTCUSD",
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2023, 12, 31),
                initial_cash=-1000.0,
            )

        # Invalid commission (negative)
        with pytest.raises(ValidationError):
            BacktestConfig(
                strategy_name="Test",
                strategy_class="test.Strategy",
                symbol="BTCUSD",
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2023, 12, 31),
                commission=-0.001,
            )

        # Invalid commission (too high)
        with pytest.raises(ValidationError):
            BacktestConfig(
                strategy_name="Test",
                strategy_class="test.Strategy",
                symbol="BTCUSD",
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2023, 12, 31),
                commission=1.5,  # 150% commission
            )

    def test_risk_management_parameters(self) -> None:
        """Test validation of risk management parameters."""
        config = BacktestConfig(
            strategy_name="Test",
            strategy_class="test.Strategy",
            symbol="BTCUSD",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            max_drawdown_pct=0.2,  # 20% max drawdown
            stop_loss_pct=0.05,  # 5% stop loss
            take_profit_pct=0.15,  # 15% take profit
        )

        assert config.max_drawdown_pct == 0.2
        assert config.stop_loss_pct == 0.05
        assert config.take_profit_pct == 0.15

        # Invalid max drawdown (negative)
        with pytest.raises(ValidationError):
            BacktestConfig(
                strategy_name="Test",
                strategy_class="test.Strategy",
                symbol="BTCUSD",
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2023, 12, 31),
                max_drawdown_pct=-0.1,
            )

        # Invalid max drawdown (over 100%)
        with pytest.raises(ValidationError):
            BacktestConfig(
                strategy_name="Test",
                strategy_class="test.Strategy",
                symbol="BTCUSD",
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2023, 12, 31),
                max_drawdown_pct=1.5,
            )


class TestTradeResult:
    """Test TradeResult model validation and functionality."""

    def test_valid_trade_creation(self) -> None:
        """Test creation of valid trade result."""
        trade = TradeResult(
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

        assert trade.entry_bar == 10
        assert trade.exit_bar == 20
        assert trade.duration == 10
        assert trade.pnl == 10.0
        assert trade.is_long is True
        assert trade.exit_reason is None  # Default value

    def test_long_vs_short_trade(self) -> None:
        """Test long and short trade creation."""
        # Long trade
        long_trade = TradeResult(
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

        # Short trade
        short_trade = TradeResult(
            entry_bar=30,
            exit_bar=40,
            entry_time=datetime(2023, 1, 30),
            exit_time=datetime(2023, 2, 9),
            duration=10,
            entry_price=100.0,
            exit_price=90.0,
            size=-1.0,
            pnl=10.0,  # Profit on short trade
            pnl_pct=0.1,
            return_pct=0.1,
            is_long=False,
        )

        assert long_trade.is_long is True
        assert short_trade.is_long is False
        assert long_trade.pnl == short_trade.pnl  # Both profitable

    def test_trade_with_exit_reason(self) -> None:
        """Test trade with exit reason."""
        trade = TradeResult(
            entry_bar=10,
            exit_bar=15,
            entry_time=datetime(2023, 1, 10),
            exit_time=datetime(2023, 1, 15),
            duration=5,
            entry_price=100.0,
            exit_price=95.0,
            size=1.0,
            pnl=-5.0,
            pnl_pct=-0.05,
            return_pct=-0.05,
            is_long=True,
            exit_reason="stop_loss",
        )

        assert trade.exit_reason == "stop_loss"
        assert trade.pnl < 0  # Loss


class TestPerformanceMetrics:
    """Test PerformanceMetrics model validation and functionality."""

    def test_valid_metrics_creation(self) -> None:
        """Test creation of valid performance metrics."""
        metrics = PerformanceMetrics(
            total_return=15.5,
            total_return_pct=0.155,
            annualized_return=12.3,
            volatility=18.2,
            sharpe_ratio=1.25,
            sortino_ratio=1.45,
            calmar_ratio=0.85,
            max_drawdown=-8.5,
            max_drawdown_duration=45,
            avg_drawdown=-3.2,
            avg_drawdown_duration=15,
            total_trades=25,
            winning_trades=18,
            losing_trades=7,
            win_rate=72.0,
            avg_win=150.0,
            avg_loss=-75.0,
            avg_trade=85.0,
            profit_factor=2.8,
            best_trade=450.0,
            worst_trade=-200.0,
            avg_trade_duration=5.5,
            exposure_time=65.0,
            buy_hold_return=8.5,
        )

        assert metrics.total_return == 15.5
        assert metrics.sharpe_ratio == 1.25
        assert metrics.total_trades == 25
        assert metrics.win_rate == 72.0

    def test_percentage_validation(self) -> None:
        """Test validation of percentage fields."""
        # Valid percentages
        metrics = PerformanceMetrics(
            total_return=15.5,
            total_return_pct=0.155,
            annualized_return=12.3,
            volatility=18.2,
            sharpe_ratio=1.25,
            sortino_ratio=1.45,
            calmar_ratio=0.85,
            max_drawdown=-8.5,
            max_drawdown_duration=45,
            avg_drawdown=-3.2,
            avg_drawdown_duration=15,
            total_trades=25,
            winning_trades=18,
            losing_trades=7,
            win_rate=75.0,  # Valid percentage
            avg_win=150.0,
            avg_loss=-75.0,
            avg_trade=85.0,
            profit_factor=2.8,
            best_trade=450.0,
            worst_trade=-200.0,
            avg_trade_duration=5.5,
            exposure_time=80.0,  # Valid percentage
            buy_hold_return=8.5,
        )

        assert 0 <= metrics.win_rate <= 100
        assert 0 <= metrics.exposure_time <= 100

        # Invalid win rate (over 100%)
        with pytest.raises(ValidationError):
            PerformanceMetrics(
                total_return=15.5,
                total_return_pct=0.155,
                annualized_return=12.3,
                volatility=18.2,
                sharpe_ratio=1.25,
                sortino_ratio=1.45,
                calmar_ratio=0.85,
                max_drawdown=-8.5,
                max_drawdown_duration=45,
                avg_drawdown=-3.2,
                avg_drawdown_duration=15,
                total_trades=25,
                winning_trades=18,
                losing_trades=7,
                win_rate=150.0,  # Invalid
                avg_win=150.0,
                avg_loss=-75.0,
                avg_trade=85.0,
                profit_factor=2.8,
                best_trade=450.0,
                worst_trade=-200.0,
                avg_trade_duration=5.5,
                exposure_time=80.0,
                buy_hold_return=8.5,
            )

        # Invalid exposure time (negative)
        with pytest.raises(ValidationError):
            PerformanceMetrics(
                total_return=15.5,
                total_return_pct=0.155,
                annualized_return=12.3,
                volatility=18.2,
                sharpe_ratio=1.25,
                sortino_ratio=1.45,
                calmar_ratio=0.85,
                max_drawdown=-8.5,
                max_drawdown_duration=45,
                avg_drawdown=-3.2,
                avg_drawdown_duration=15,
                total_trades=25,
                winning_trades=18,
                losing_trades=7,
                win_rate=75.0,
                avg_win=150.0,
                avg_loss=-75.0,
                avg_trade=85.0,
                profit_factor=2.8,
                best_trade=450.0,
                worst_trade=-200.0,
                avg_trade_duration=5.5,
                exposure_time=-10.0,  # Invalid
                buy_hold_return=8.5,
            )


class TestBacktestResult:
    """Test BacktestResult model validation and functionality."""

    def test_valid_result_creation(self) -> None:
        """Test creation of valid backtest result."""
        config = BacktestConfig(
            strategy_name="Test Strategy",
            strategy_class="test.Strategy",
            symbol="BTCUSD",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
        )

        metrics = PerformanceMetrics(
            total_return=15.5,
            total_return_pct=0.155,
            annualized_return=12.3,
            volatility=18.2,
            sharpe_ratio=1.25,
            sortino_ratio=1.45,
            calmar_ratio=0.85,
            max_drawdown=-8.5,
            max_drawdown_duration=45,
            avg_drawdown=-3.2,
            avg_drawdown_duration=15,
            total_trades=25,
            winning_trades=18,
            losing_trades=7,
            win_rate=72.0,
            avg_win=150.0,
            avg_loss=-75.0,
            avg_trade=85.0,
            profit_factor=2.8,
            best_trade=450.0,
            worst_trade=-200.0,
            avg_trade_duration=5.5,
            exposure_time=65.0,
            buy_hold_return=8.5,
        )

        started_at = datetime(2023, 1, 1, 10, 0, 0)
        completed_at = datetime(2023, 1, 1, 10, 5, 30)

        result = BacktestResult(
            config=config,
            metrics=metrics,
            execution_time=330.5,
            started_at=started_at,
            completed_at=completed_at,
            data_points=365,
        )

        assert result.config.strategy_name == "Test Strategy"
        assert result.metrics.total_return == 15.5
        assert result.execution_time == 330.5
        assert result.status == "completed"  # Default value
        assert result.error_message is None  # Default value

    def test_status_validation(self) -> None:
        """Test status field validation."""
        config = BacktestConfig(
            strategy_name="Test",
            strategy_class="test.Strategy",
            symbol="BTCUSD",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
        )

        metrics = PerformanceMetrics(
            total_return=0,
            total_return_pct=0,
            annualized_return=0,
            volatility=0,
            sharpe_ratio=0,
            sortino_ratio=0,
            calmar_ratio=0,
            max_drawdown=0,
            max_drawdown_duration=0,
            avg_drawdown=0,
            avg_drawdown_duration=0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0,
            avg_win=0,
            avg_loss=0,
            avg_trade=0,
            profit_factor=0,
            best_trade=0,
            worst_trade=0,
            avg_trade_duration=0,
            exposure_time=0,
            buy_hold_return=0,
        )

        # Valid statuses
        for status in ["running", "completed", "failed", "cancelled"]:
            result = BacktestResult(
                config=config,
                metrics=metrics,
                execution_time=100.0,
                started_at=datetime.now(),
                completed_at=datetime.now(),
                data_points=100,
                status=status,
            )
            assert result.status == status

        # Invalid status
        with pytest.raises(ValidationError):
            BacktestResult(
                config=config,
                metrics=metrics,
                execution_time=100.0,
                started_at=datetime.now(),
                completed_at=datetime.now(),
                data_points=100,
                status="invalid_status",
            )

    def test_duration_property(self) -> None:
        """Test duration property calculation."""
        started_at = datetime(2023, 1, 1, 10, 0, 0)
        completed_at = datetime(2023, 1, 1, 10, 5, 30)  # 5 minutes 30 seconds later

        config = BacktestConfig(
            strategy_name="Test",
            strategy_class="test.Strategy",
            symbol="BTCUSD",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
        )

        metrics = PerformanceMetrics(
            total_return=0,
            total_return_pct=0,
            annualized_return=0,
            volatility=0,
            sharpe_ratio=0,
            sortino_ratio=0,
            calmar_ratio=0,
            max_drawdown=0,
            max_drawdown_duration=0,
            avg_drawdown=0,
            avg_drawdown_duration=0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0,
            avg_win=0,
            avg_loss=0,
            avg_trade=0,
            profit_factor=0,
            best_trade=0,
            worst_trade=0,
            avg_trade_duration=0,
            exposure_time=0,
            buy_hold_return=0,
        )

        result = BacktestResult(
            config=config,
            metrics=metrics,
            execution_time=100.0,
            started_at=started_at,
            completed_at=completed_at,
            data_points=100,
        )

        expected_duration = (completed_at - started_at).total_seconds()
        assert result.duration == expected_duration
        assert result.duration == 330.0  # 5.5 minutes in seconds

    def test_success_rate_property(self) -> None:
        """Test success rate property calculation."""
        config = BacktestConfig(
            strategy_name="Test",
            strategy_class="test.Strategy",
            symbol="BTCUSD",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
        )

        # Test with trades
        metrics = PerformanceMetrics(
            total_return=15.5,
            total_return_pct=0.155,
            annualized_return=12.3,
            volatility=18.2,
            sharpe_ratio=1.25,
            sortino_ratio=1.45,
            calmar_ratio=0.85,
            max_drawdown=-8.5,
            max_drawdown_duration=45,
            avg_drawdown=-3.2,
            avg_drawdown_duration=15,
            total_trades=10,
            winning_trades=7,
            losing_trades=3,
            win_rate=70.0,
            avg_win=150.0,
            avg_loss=-75.0,
            avg_trade=85.0,
            profit_factor=2.8,
            best_trade=450.0,
            worst_trade=-200.0,
            avg_trade_duration=5.5,
            exposure_time=65.0,
            buy_hold_return=8.5,
        )

        result = BacktestResult(
            config=config,
            metrics=metrics,
            execution_time=100.0,
            started_at=datetime.now(),
            completed_at=datetime.now(),
            data_points=100,
        )

        assert result.success_rate == 70.0  # 7/10 * 100

        # Test with no trades
        metrics_no_trades = PerformanceMetrics(
            total_return=0,
            total_return_pct=0,
            annualized_return=0,
            volatility=0,
            sharpe_ratio=0,
            sortino_ratio=0,
            calmar_ratio=0,
            max_drawdown=0,
            max_drawdown_duration=0,
            avg_drawdown=0,
            avg_drawdown_duration=0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0,
            avg_win=0,
            avg_loss=0,
            avg_trade=0,
            profit_factor=0,
            best_trade=0,
            worst_trade=0,
            avg_trade_duration=0,
            exposure_time=0,
            buy_hold_return=0,
        )

        result_no_trades = BacktestResult(
            config=config,
            metrics=metrics_no_trades,
            execution_time=100.0,
            started_at=datetime.now(),
            completed_at=datetime.now(),
            data_points=100,
        )

        assert result_no_trades.success_rate == 0.0

    def test_get_monthly_returns(self) -> None:
        """Test monthly returns calculation."""
        config = BacktestConfig(
            strategy_name="Test",
            strategy_class="test.Strategy",
            symbol="BTCUSD",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
        )

        metrics = PerformanceMetrics(
            total_return=0,
            total_return_pct=0,
            annualized_return=0,
            volatility=0,
            sharpe_ratio=0,
            sortino_ratio=0,
            calmar_ratio=0,
            max_drawdown=0,
            max_drawdown_duration=0,
            avg_drawdown=0,
            avg_drawdown_duration=0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0,
            avg_win=0,
            avg_loss=0,
            avg_trade=0,
            profit_factor=0,
            best_trade=0,
            worst_trade=0,
            avg_trade_duration=0,
            exposure_time=0,
            buy_hold_return=0,
        )

        # Create equity curve data spanning multiple months
        equity_curve = [
            {"date": datetime(2023, 1, 31), "equity": 10000},  # End of Jan
            {"date": datetime(2023, 2, 28), "equity": 10500},  # End of Feb
            {"date": datetime(2023, 3, 31), "equity": 10200},  # End of Mar
        ]

        result = BacktestResult(
            config=config,
            metrics=metrics,
            equity_curve=equity_curve,
            execution_time=100.0,
            started_at=datetime.now(),
            completed_at=datetime.now(),
            data_points=100,
        )

        monthly_returns = result.get_monthly_returns()

        # Should have returns for all three months
        assert "2023-01" in monthly_returns
        assert "2023-02" in monthly_returns
        assert "2023-03" in monthly_returns

        # Jan return: (10000 - 10000) / 10000 * 100 = 0%
        # (same start and end for first month)
        assert abs(monthly_returns["2023-01"] - 0.0) < 0.001

        # Feb return: (10500 - 10000) / 10000 * 100 = 5%
        assert abs(monthly_returns["2023-02"] - 5.0) < 0.001

        # Mar return: (10200 - 10500) / 10500 * 100 ≈ -2.86%
        assert abs(monthly_returns["2023-03"] - (-2.857142857142857)) < 0.001

    def test_get_trade_summary(self) -> None:
        """Test trade summary calculation."""
        config = BacktestConfig(
            strategy_name="Test",
            strategy_class="test.Strategy",
            symbol="BTCUSD",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
        )

        metrics = PerformanceMetrics(
            total_return=0,
            total_return_pct=0,
            annualized_return=0,
            volatility=0,
            sharpe_ratio=0,
            sortino_ratio=0,
            calmar_ratio=0,
            max_drawdown=0,
            max_drawdown_duration=0,
            avg_drawdown=0,
            avg_drawdown_duration=0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0,
            avg_win=0,
            avg_loss=0,
            avg_trade=0,
            profit_factor=0,
            best_trade=0,
            worst_trade=0,
            avg_trade_duration=0,
            exposure_time=0,
            buy_hold_return=0,
        )

        # Create sample trades
        trades = [
            TradeResult(
                entry_bar=10,
                exit_bar=15,
                duration=5,
                entry_time=datetime(2023, 1, 10),
                exit_time=datetime(2023, 1, 15),
                entry_price=100.0,
                exit_price=110.0,
                size=1.0,
                pnl=10.0,
                pnl_pct=0.1,
                return_pct=0.1,
                is_long=True,
            ),
            TradeResult(
                entry_bar=20,
                exit_bar=30,
                duration=10,
                entry_time=datetime(2023, 1, 20),
                exit_time=datetime(2023, 1, 30),
                entry_price=110.0,
                exit_price=105.0,
                size=1.0,
                pnl=-5.0,
                pnl_pct=-0.045,
                return_pct=-0.045,
                is_long=True,
            ),
            TradeResult(
                entry_bar=40,
                exit_bar=43,
                duration=3,
                entry_time=datetime(2023, 2, 9),
                exit_time=datetime(2023, 2, 12),
                entry_price=105.0,
                exit_price=115.0,
                size=-1.0,  # Short trade
                pnl=10.0,
                pnl_pct=0.095,
                return_pct=0.095,
                is_long=False,
            ),
        ]

        result = BacktestResult(
            config=config,
            metrics=metrics,
            trades=trades,
            execution_time=100.0,
            started_at=datetime.now(),
            completed_at=datetime.now(),
            data_points=100,
        )

        summary = result.get_trade_summary()

        assert summary["total_trades"] == 3
        assert summary["long_trades"] == 2
        assert summary["short_trades"] == 1
        assert summary["avg_duration_days"] == (5 + 10 + 3) / 3  # 6 days
        assert summary["longest_trade_days"] == 10
        assert summary["shortest_trade_days"] == 3
        assert summary["long_win_rate"] == 50.0  # 1 out of 2 long trades won
        assert summary["short_win_rate"] == 100.0  # 1 out of 1 short trade won

        # Test with empty trades
        result_empty = BacktestResult(
            config=config,
            metrics=metrics,
            trades=[],
            execution_time=100.0,
            started_at=datetime.now(),
            completed_at=datetime.now(),
            data_points=100,
        )

        empty_summary = result_empty.get_trade_summary()
        assert empty_summary == {}


if __name__ == "__main__":
    pytest.main([__file__])
