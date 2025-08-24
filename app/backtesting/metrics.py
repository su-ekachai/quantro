"""
Performance metrics calculation utilities for backtesting.

This module provides functions to calculate comprehensive trading performance
metrics including risk-adjusted returns, drawdown analysis, and trade statistics.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

import pandas as pd
from loguru import logger

from app.backtesting.models import PerformanceMetrics, TradeResult


class MetricsCalculator:
    """Calculator for comprehensive backtesting performance metrics."""

    def __init__(
        self,
        trades: list[TradeResult],
        equity_curve: pd.Series,
        initial_cash: float,
        risk_free_rate: float = 0.02,
    ) -> None:
        """
        Initialize metrics calculator.

        Args:
            trades: List of trade results
            equity_curve: Equity curve as pandas Series with datetime index
            initial_cash: Initial portfolio value
            risk_free_rate: Risk-free rate for Sharpe ratio calculation (default: 2%)
        """
        self.trades = trades
        self.equity_curve = equity_curve
        self.initial_cash = initial_cash
        self.risk_free_rate = risk_free_rate

        # Calculate returns
        self.returns = equity_curve.pct_change().dropna()
        self.daily_returns = self._resample_to_daily()

    def _resample_to_daily(self) -> pd.Series:
        """Resample returns to daily frequency for consistent calculations."""
        if self.equity_curve.empty:
            return pd.Series(dtype=float)

        # If already daily or higher frequency, resample to daily
        daily_equity = self.equity_curve.resample("D").last().ffill()
        return daily_equity.pct_change().dropna()

    def calculate_all_metrics(self, buy_hold_return: float = 0.0) -> PerformanceMetrics:
        """
        Calculate all performance metrics.

        Args:
            buy_hold_return: Buy and hold return for comparison

        Returns:
            Complete performance metrics
        """
        try:
            # Basic return metrics
            total_return_pct = self._calculate_total_return()
            annualized_return = self._calculate_annualized_return()

            # Risk metrics
            volatility = self._calculate_volatility()
            sharpe_ratio = self._calculate_sharpe_ratio()
            sortino_ratio = self._calculate_sortino_ratio()
            calmar_ratio = self._calculate_calmar_ratio()

            # Drawdown metrics
            drawdown_metrics = self._calculate_drawdown_metrics()

            # Trade statistics
            trade_stats = self._calculate_trade_statistics()

            # Additional metrics
            exposure_time = self._calculate_exposure_time()

            return PerformanceMetrics(
                total_return=total_return_pct * 100,
                total_return_pct=total_return_pct,
                annualized_return=annualized_return * 100,
                volatility=volatility * 100,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                max_drawdown=drawdown_metrics["max_drawdown"] * 100,
                max_drawdown_duration=drawdown_metrics["max_drawdown_duration"],
                avg_drawdown=drawdown_metrics["avg_drawdown"] * 100,
                avg_drawdown_duration=drawdown_metrics["avg_drawdown_duration"],
                total_trades=trade_stats["total_trades"],
                winning_trades=trade_stats["winning_trades"],
                losing_trades=trade_stats["losing_trades"],
                win_rate=trade_stats["win_rate"] * 100,
                avg_win=trade_stats["avg_win"],
                avg_loss=trade_stats["avg_loss"],
                avg_trade=trade_stats["avg_trade"],
                profit_factor=trade_stats["profit_factor"],
                best_trade=trade_stats["best_trade"],
                worst_trade=trade_stats["worst_trade"],
                avg_trade_duration=trade_stats["avg_trade_duration"],
                exposure_time=exposure_time * 100,
                buy_hold_return=buy_hold_return * 100,
            )

        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            # For debugging, re-raise the exception in tests
            import os

            if os.getenv("PYTEST_CURRENT_TEST"):
                raise
            # Return default metrics on error
            return self._get_default_metrics()

    def _calculate_total_return(self) -> float:
        """Calculate total return percentage."""
        if self.equity_curve.empty:
            return 0.0

        if self.initial_cash == 0:
            return 0.0  # Avoid division by zero

        final_value = self.equity_curve.iloc[-1]
        return (final_value - self.initial_cash) / self.initial_cash

    def _calculate_annualized_return(self) -> float:
        """Calculate annualized return."""
        if self.equity_curve.empty or len(self.equity_curve) < 2:
            return 0.0

        total_return = self._calculate_total_return()

        # Calculate time period in years
        start_date = self.equity_curve.index[0]
        end_date = self.equity_curve.index[-1]
        years = (end_date - start_date).days / 365.25

        if years <= 0:
            return 0.0

        # Annualized return formula: (1 + total_return)^(1/years) - 1
        return (1 + total_return) ** (1 / years) - 1

    def _calculate_volatility(self) -> float:
        """Calculate annualized volatility."""
        if self.daily_returns.empty:
            return 0.0

        daily_vol = self.daily_returns.std()

        # Handle NaN or infinite values
        if pd.isna(daily_vol) or not math.isfinite(daily_vol):
            return 0.0

        return daily_vol * math.sqrt(252)  # Annualize assuming 252 trading days

    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio."""
        if self.daily_returns.empty:
            return 0.0

        excess_returns = self.daily_returns - (self.risk_free_rate / 252)

        # Handle edge cases
        excess_std = excess_returns.std()
        if excess_std == 0 or pd.isna(excess_std) or not math.isfinite(excess_std):
            return 0.0

        excess_mean = excess_returns.mean()
        if pd.isna(excess_mean) or not math.isfinite(excess_mean):
            return 0.0

        return excess_mean / excess_std * math.sqrt(252)

    def _calculate_sortino_ratio(self) -> float:
        """Calculate Sortino ratio (downside deviation)."""
        if self.daily_returns.empty:
            return 0.0

        excess_returns = self.daily_returns - (self.risk_free_rate / 252)
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0:
            return 0.0

        downside_std = downside_returns.std()
        if (
            downside_std == 0
            or pd.isna(downside_std)
            or not math.isfinite(downside_std)
        ):
            return 0.0

        excess_mean = excess_returns.mean()
        if pd.isna(excess_mean) or not math.isfinite(excess_mean):
            return 0.0

        return excess_mean / downside_std * math.sqrt(252)

    def _calculate_calmar_ratio(self) -> float:
        """Calculate Calmar ratio (annualized return / max drawdown)."""
        annualized_return = self._calculate_annualized_return()
        max_drawdown = self._calculate_max_drawdown()

        if max_drawdown == 0:
            return 0.0

        return annualized_return / abs(max_drawdown)

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        if self.equity_curve.empty:
            return 0.0

        # Calculate running maximum
        running_max = self.equity_curve.expanding().max()

        # Calculate drawdown
        drawdown = (self.equity_curve - running_max) / running_max

        return drawdown.min()

    def _calculate_drawdown_metrics(self) -> dict[str, Any]:
        """Calculate comprehensive drawdown metrics."""
        if self.equity_curve.empty:
            return {
                "max_drawdown": 0.0,
                "max_drawdown_duration": 0,
                "avg_drawdown": 0.0,
                "avg_drawdown_duration": 0,
            }

        # Calculate running maximum
        running_max = self.equity_curve.expanding().max()

        # Calculate drawdown
        drawdown = (self.equity_curve - running_max) / running_max

        # Find drawdown periods
        in_drawdown = drawdown < 0
        drawdown_periods = []

        start_idx = None
        for i, is_dd in enumerate(in_drawdown):
            if is_dd and start_idx is None:
                start_idx = i
            elif not is_dd and start_idx is not None:
                drawdown_periods.append((start_idx, i - 1))
                start_idx = None

        # Handle case where drawdown continues to end
        if start_idx is not None:
            drawdown_periods.append((start_idx, len(drawdown) - 1))

        # Calculate metrics
        max_drawdown = drawdown.min()

        if drawdown_periods:
            durations = []
            drawdown_values = []

            for start, end in drawdown_periods:
                duration = (
                    self.equity_curve.index[end] - self.equity_curve.index[start]
                ).days
                durations.append(duration)
                drawdown_values.append(drawdown.iloc[start : end + 1].min())

            max_drawdown_duration = max(durations) if durations else 0
            avg_drawdown = (
                sum(drawdown_values) / len(drawdown_values) if drawdown_values else 0.0
            )
            avg_drawdown_duration = (
                float(sum(durations) / len(durations)) if durations else 0.0
            )
        else:
            max_drawdown_duration = 0
            avg_drawdown = 0.0
            avg_drawdown_duration = 0

        return {
            "max_drawdown": max_drawdown,
            "max_drawdown_duration": max_drawdown_duration,
            "avg_drawdown": avg_drawdown,
            "avg_drawdown_duration": avg_drawdown_duration,
        }

    def _calculate_trade_statistics(self) -> dict[str, Any]:
        """Calculate comprehensive trade statistics."""
        if not self.trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "avg_trade": 0.0,
                "profit_factor": 0.0,
                "best_trade": 0.0,
                "worst_trade": 0.0,
                "avg_trade_duration": 0.0,
            }

        # Basic counts
        total_trades = len(self.trades)
        winning_trades = sum(1 for trade in self.trades if trade.pnl > 0)
        losing_trades = sum(1 for trade in self.trades if trade.pnl < 0)

        # Win rate
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        # P&L statistics
        pnls = [trade.pnl for trade in self.trades]
        winning_pnls = [trade.pnl for trade in self.trades if trade.pnl > 0]
        losing_pnls = [trade.pnl for trade in self.trades if trade.pnl < 0]

        avg_win = sum(winning_pnls) / len(winning_pnls) if winning_pnls else 0.0
        avg_loss = sum(losing_pnls) / len(losing_pnls) if losing_pnls else 0.0
        avg_trade = sum(pnls) / len(pnls) if pnls else 0.0

        # Profit factor
        gross_profit = sum(winning_pnls) if winning_pnls else 0.0
        gross_loss = abs(sum(losing_pnls)) if losing_pnls else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

        # Best and worst trades
        best_trade = max(pnls) if pnls else 0.0
        worst_trade = min(pnls) if pnls else 0.0

        # Average trade duration
        durations = [trade.duration for trade in self.trades]
        avg_trade_duration = sum(durations) / len(durations) if durations else 0.0

        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "avg_trade": avg_trade,
            "profit_factor": profit_factor,
            "best_trade": best_trade,
            "worst_trade": worst_trade,
            "avg_trade_duration": avg_trade_duration,
        }

    def _calculate_exposure_time(self) -> float:
        """Calculate market exposure time percentage."""
        if not self.trades or self.equity_curve.empty:
            return 0.0

        total_time = (
            self.equity_curve.index[-1] - self.equity_curve.index[0]
        ).total_seconds()

        if total_time <= 0:
            return 0.0

        # Calculate time in market
        exposure_time = 0.0
        for trade in self.trades:
            trade_duration = (trade.exit_time - trade.entry_time).total_seconds()
            exposure_time += trade_duration

        return min(exposure_time / total_time, 1.0)

    def _get_default_metrics(self) -> PerformanceMetrics:
        """Get default metrics when calculation fails."""
        return PerformanceMetrics(
            total_return=0.0,
            total_return_pct=0.0,
            annualized_return=0.0,
            volatility=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            max_drawdown=0.0,
            max_drawdown_duration=0,
            avg_drawdown=0.0,
            avg_drawdown_duration=0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            avg_trade=0.0,
            profit_factor=0.0,
            best_trade=0.0,
            worst_trade=0.0,
            avg_trade_duration=0.0,
            exposure_time=0.0,
            buy_hold_return=0.0,
        )


def calculate_buy_hold_return(
    data: pd.DataFrame, start_date: datetime, end_date: datetime
) -> float:
    """
    Calculate buy and hold return for comparison.

    Args:
        data: OHLCV data with datetime index
        start_date: Start date for calculation
        end_date: End date for calculation

    Returns:
        Buy and hold return percentage
    """
    try:
        # Filter data to date range
        mask = (data.index >= start_date) & (data.index <= end_date)
        filtered_data = data[mask]

        if filtered_data.empty or len(filtered_data) < 2:
            return 0.0

        start_price = filtered_data.iloc[0]["Close"]
        end_price = filtered_data.iloc[-1]["Close"]

        return (end_price - start_price) / start_price

    except Exception as e:
        logger.error(f"Error calculating buy and hold return: {e}")
        return 0.0


def calculate_rolling_metrics(
    equity_curve: pd.Series, window_days: int = 252
) -> dict[str, pd.Series]:
    """
    Calculate rolling performance metrics.

    Args:
        equity_curve: Equity curve with datetime index
        window_days: Rolling window in days

    Returns:
        Dictionary of rolling metrics
    """
    try:
        returns = equity_curve.pct_change().dropna()

        # Resample to daily if needed
        if not returns.empty:
            daily_returns = returns.resample("D").sum()
        else:
            daily_returns = pd.Series(dtype=float)

        rolling_metrics = {}

        if not daily_returns.empty:
            # Rolling Sharpe ratio
            rolling_mean = daily_returns.rolling(window=window_days).mean()
            rolling_std = daily_returns.rolling(window=window_days).std()
            rolling_metrics["sharpe"] = (rolling_mean / rolling_std) * math.sqrt(252)

            # Rolling volatility
            rolling_metrics["volatility"] = rolling_std * math.sqrt(252)

            # Rolling max drawdown
            rolling_max = equity_curve.rolling(window=window_days).max()
            rolling_drawdown = (equity_curve - rolling_max) / rolling_max
            rolling_metrics["max_drawdown"] = rolling_drawdown.rolling(
                window=window_days
            ).min()

        return rolling_metrics

    except Exception as e:
        logger.error(f"Error calculating rolling metrics: {e}")
        return {}
