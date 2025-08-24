"""
Pydantic models for backtesting configuration and results.

This module defines the data models used for configuring backtests
and storing comprehensive performance results.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


class BacktestConfig(BaseModel):
    """Configuration for running a backtest."""

    # Strategy configuration
    strategy_name: str = Field(..., description="Name of the strategy to backtest")
    strategy_class: str = Field(..., description="Fully qualified strategy class name")
    strategy_parameters: dict[str, Any] = Field(
        default_factory=dict, description="Strategy-specific parameters"
    )

    # Data configuration
    symbol: str = Field(..., description="Trading symbol to backtest")
    start_date: datetime = Field(..., description="Backtest start date")
    end_date: datetime = Field(..., description="Backtest end date")
    timeframe: str = Field(default="1d", description="Data timeframe")

    # Trading configuration
    initial_cash: float = Field(
        default=10000.0, gt=0, description="Initial cash for backtesting"
    )
    commission: float = Field(
        default=0.001, ge=0, le=0.1, description="Commission rate (0.001 = 0.1%)"
    )
    margin: float = Field(
        default=1.0, gt=0, description="Margin requirement (1.0 = no margin)"
    )
    trade_on_open: bool = Field(
        default=False, description="Execute trades on bar open vs close"
    )
    hedging: bool = Field(
        default=False, description="Allow hedging (multiple positions)"
    )
    exclusive_orders: bool = Field(
        default=False, description="Cancel pending orders on new signal"
    )

    # Risk management
    max_drawdown_pct: float | None = Field(
        default=None, ge=0, le=1, description="Maximum allowed drawdown (0.2 = 20%)"
    )
    stop_loss_pct: float | None = Field(
        default=None, ge=0, le=1, description="Global stop loss percentage"
    )
    take_profit_pct: float | None = Field(
        default=None, ge=0, description="Global take profit percentage"
    )

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Validate symbol is not empty."""
        if not v.strip():
            raise ValueError("Symbol cannot be empty")
        return v.strip().upper()

    @field_validator("strategy_name")
    @classmethod
    def validate_strategy_name(cls, v: str) -> str:
        """Validate strategy name is not empty."""
        if not v.strip():
            raise ValueError("Strategy name cannot be empty")
        return v.strip()

    @field_validator("end_date")
    @classmethod
    def validate_date_range(cls, v: datetime, info: Any) -> datetime:
        """Validate end date is after start date."""
        if "start_date" in info.data and v <= info.data["start_date"]:
            raise ValueError("End date must be after start date")
        return v


class TradeResult(BaseModel):
    """Individual trade result from backtest."""

    entry_bar: int = Field(..., description="Bar index when trade was entered")
    exit_bar: int = Field(..., description="Bar index when trade was exited")
    entry_time: datetime = Field(..., description="Entry timestamp")
    exit_time: datetime = Field(..., description="Exit timestamp")
    duration: int = Field(..., description="Trade duration in bars")

    entry_price: float = Field(..., description="Entry price")
    exit_price: float = Field(..., description="Exit price")
    size: float = Field(..., description="Position size")

    pnl: float = Field(..., description="Profit/Loss in currency")
    pnl_pct: float = Field(..., description="Profit/Loss as percentage")

    return_pct: float = Field(..., description="Return percentage")

    # Trade metadata
    is_long: bool = Field(..., description="True if long position")
    exit_reason: str | None = Field(default=None, description="Reason for exit")


class PerformanceMetrics(BaseModel):
    """Comprehensive performance metrics from backtest."""

    # Basic metrics
    total_return: float = Field(..., description="Total return percentage")
    total_return_pct: float = Field(..., description="Total return as decimal")
    annualized_return: float = Field(..., description="Annualized return percentage")

    # Risk metrics
    volatility: float = Field(..., description="Annualized volatility")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    sortino_ratio: float = Field(..., description="Sortino ratio")
    calmar_ratio: float = Field(..., description="Calmar ratio")

    # Drawdown metrics
    max_drawdown: float = Field(..., description="Maximum drawdown percentage")
    max_drawdown_duration: int = Field(..., description="Max drawdown duration in days")
    avg_drawdown: float = Field(..., description="Average drawdown percentage")
    avg_drawdown_duration: float = Field(..., description="Average drawdown duration")

    # Trade statistics
    total_trades: int = Field(..., description="Total number of trades")
    winning_trades: int = Field(..., description="Number of winning trades")
    losing_trades: int = Field(..., description="Number of losing trades")
    win_rate: float = Field(..., description="Win rate percentage")

    # Profit/Loss metrics
    avg_win: float = Field(..., description="Average winning trade")
    avg_loss: float = Field(..., description="Average losing trade")
    avg_trade: float = Field(..., description="Average trade P&L")
    profit_factor: float = Field(
        ..., description="Profit factor (gross profit / gross loss)"
    )

    # Additional metrics
    best_trade: float = Field(..., description="Best single trade")
    worst_trade: float = Field(..., description="Worst single trade")
    avg_trade_duration: float = Field(..., description="Average trade duration in days")

    # Exposure metrics
    exposure_time: float = Field(..., description="Market exposure time percentage")

    # Buy & Hold comparison
    buy_hold_return: float = Field(..., description="Buy & hold return percentage")

    @field_validator("win_rate", "exposure_time")
    @classmethod
    def validate_percentage(cls, v: float) -> float:
        """Validate percentage values are between 0 and 100."""
        if not 0 <= v <= 100:
            raise ValueError("Percentage must be between 0 and 100")
        return v


class BacktestResult(BaseModel):
    """Complete backtest result with metrics and trade details."""

    # Configuration
    config: BacktestConfig = Field(..., description="Backtest configuration used")

    # Performance metrics
    metrics: PerformanceMetrics = Field(..., description="Performance metrics")

    # Trade details
    trades: list[TradeResult] = Field(
        default_factory=list, description="Individual trade results"
    )

    # Equity curve data
    equity_curve: list[dict[str, Any]] = Field(
        default_factory=list, description="Equity curve data points"
    )

    # Drawdown data
    drawdown_curve: list[dict[str, Any]] = Field(
        default_factory=list, description="Drawdown curve data points"
    )

    # Execution metadata
    execution_time: float = Field(..., description="Backtest execution time in seconds")
    data_points: int = Field(..., description="Number of data points processed")

    # Status and timestamps
    status: str = Field(default="completed", description="Backtest status")
    started_at: datetime = Field(..., description="Backtest start timestamp")
    completed_at: datetime = Field(..., description="Backtest completion timestamp")

    # Optional error information
    error_message: str | None = Field(
        default=None, description="Error message if failed"
    )

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        """Validate status is one of allowed values."""
        allowed_statuses = ["running", "completed", "failed", "cancelled"]
        if v not in allowed_statuses:
            raise ValueError(f"Status must be one of: {allowed_statuses}")
        return v

    @property
    def duration(self) -> float:
        """Get backtest duration in seconds."""
        return (self.completed_at - self.started_at).total_seconds()

    @property
    def success_rate(self) -> float:
        """Get success rate as percentage."""
        if self.metrics.total_trades == 0:
            return 0.0
        return (self.metrics.winning_trades / self.metrics.total_trades) * 100

    def get_monthly_returns(self) -> dict[str, float]:
        """Extract monthly returns from equity curve."""
        monthly_returns: dict[str, float] = {}

        if not self.equity_curve:
            return monthly_returns

        # Group equity points by month
        monthly_data: dict[str, list[dict[str, Any]]] = {}
        for point in self.equity_curve:
            date = point["date"]
            month_key = f"{date.year}-{date.month:02d}"

            if month_key not in monthly_data:
                monthly_data[month_key] = []
            monthly_data[month_key].append(point)

        # Calculate monthly returns
        prev_month_end_value = self.config.initial_cash

        for month_key in sorted(monthly_data.keys()):
            month_points = monthly_data[month_key]

            # Get start and end values for the month
            month_start_value = month_points[0]["equity"]
            month_end_value = month_points[-1]["equity"]

            # For the first month, use the actual start value
            if month_key == min(monthly_data.keys()):
                start_value = month_start_value
            else:
                start_value = prev_month_end_value

            # Calculate monthly return
            if start_value > 0:
                monthly_return = ((month_end_value - start_value) / start_value) * 100
                monthly_returns[month_key] = monthly_return

            prev_month_end_value = month_end_value

        return monthly_returns

    def get_trade_summary(self) -> dict[str, Any]:
        """Get summary statistics for trades."""
        if not self.trades:
            return {}

        long_trades = [t for t in self.trades if t.is_long]
        short_trades = [t for t in self.trades if not t.is_long]

        return {
            "total_trades": len(self.trades),
            "long_trades": len(long_trades),
            "short_trades": len(short_trades),
            "avg_duration_days": sum(t.duration for t in self.trades) / len(self.trades)
            if self.trades
            else 0,
            "longest_trade_days": max(t.duration for t in self.trades)
            if self.trades
            else 0,
            "shortest_trade_days": min(t.duration for t in self.trades)
            if self.trades
            else 0,
            "long_win_rate": (
                sum(1 for t in long_trades if t.pnl > 0) / len(long_trades) * 100
            )
            if long_trades
            else 0,
            "short_win_rate": (
                sum(1 for t in short_trades if t.pnl > 0) / len(short_trades) * 100
            )
            if short_trades
            else 0,
        }
