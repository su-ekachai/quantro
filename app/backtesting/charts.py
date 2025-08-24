"""
Chart generation utilities for backtesting results.

This module provides functions to generate interactive charts for
equity curves, drawdown analysis, and trade visualization using
Plotly for web-based display.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
from loguru import logger

from app.backtesting.models import BacktestResult


class ChartGenerator:
    """Generator for interactive backtesting charts."""

    def __init__(self) -> None:
        """Initialize chart generator."""
        pass

    def generate_equity_curve_chart(self, result: BacktestResult) -> dict[str, Any]:
        """
        Generate equity curve chart data.

        Args:
            result: Backtest result containing equity curve data

        Returns:
            Chart configuration for frontend rendering
        """
        try:
            if not result.equity_curve:
                return self._get_empty_chart("No equity data available")

            # Extract data
            dates = [point["date"] for point in result.equity_curve]
            equity_values = [point["equity"] for point in result.equity_curve]

            # Calculate buy & hold line for comparison
            initial_value = (
                equity_values[0] if equity_values else result.config.initial_cash
            )
            buy_hold_final = initial_value * (1 + result.metrics.buy_hold_return / 100)
            buy_hold_values = (
                [
                    initial_value
                    + (buy_hold_final - initial_value) * i / (len(equity_values) - 1)
                    for i in range(len(equity_values))
                ]
                if len(equity_values) > 1
                else [initial_value]
            )

            chart_config = {
                "type": "line",
                "title": f"Equity Curve - {result.config.strategy_name}",
                "subtitle": (
                    f"{result.config.symbol} | "
                    f"{result.config.start_date.strftime('%Y-%m-%d')} to "
                    f"{result.config.end_date.strftime('%Y-%m-%d')}"
                ),
                "data": {
                    "labels": [
                        d.isoformat() if isinstance(d, datetime) else str(d)
                        for d in dates
                    ],
                    "datasets": [
                        {
                            "label": "Strategy Equity",
                            "data": equity_values,
                            "borderColor": "#3b82f6",
                            "backgroundColor": "rgba(59, 130, 246, 0.1)",
                            "fill": True,
                            "tension": 0.1,
                        },
                        {
                            "label": "Buy & Hold",
                            "data": buy_hold_values,
                            "borderColor": "#ef4444",
                            "backgroundColor": "transparent",
                            "borderDash": [5, 5],
                            "fill": False,
                        },
                    ],
                },
                "options": {
                    "responsive": True,
                    "maintainAspectRatio": False,
                    "scales": {
                        "x": {
                            "type": "time",
                            "title": {"display": True, "text": "Date"},
                        },
                        "y": {
                            "title": {"display": True, "text": "Portfolio Value ($)"},
                            "beginAtZero": False,
                        },
                    },
                    "plugins": {
                        "legend": {"display": True, "position": "top"},
                        "tooltip": {
                            "mode": "index",
                            "intersect": False,
                            "callbacks": {
                                "label": (
                                    "function(context) { "
                                    "return context.dataset.label + ': $' + "
                                    "context.parsed.y.toLocaleString(); }"
                                )
                            },
                        },
                    },
                    "interaction": {"mode": "nearest", "axis": "x", "intersect": False},
                },
            }

            return chart_config

        except Exception as e:
            logger.error(f"Failed to generate equity curve chart: {e}")
            return self._get_empty_chart("Error generating equity curve")

    def generate_drawdown_chart(self, result: BacktestResult) -> dict[str, Any]:
        """
        Generate drawdown chart data.

        Args:
            result: Backtest result containing drawdown data

        Returns:
            Chart configuration for drawdown visualization
        """
        try:
            if not result.drawdown_curve:
                return self._get_empty_chart("No drawdown data available")

            # Extract data
            dates = [point["date"] for point in result.drawdown_curve]
            drawdown_values = [point["drawdown"] for point in result.drawdown_curve]

            chart_config = {
                "type": "line",
                "title": "Drawdown Analysis",
                "subtitle": f"Maximum Drawdown: {result.metrics.max_drawdown:.2f}%",
                "data": {
                    "labels": [
                        d.isoformat() if isinstance(d, datetime) else str(d)
                        for d in dates
                    ],
                    "datasets": [
                        {
                            "label": "Drawdown %",
                            "data": drawdown_values,
                            "borderColor": "#ef4444",
                            "backgroundColor": "rgba(239, 68, 68, 0.2)",
                            "fill": True,
                            "tension": 0.1,
                        }
                    ],
                },
                "options": {
                    "responsive": True,
                    "maintainAspectRatio": False,
                    "scales": {
                        "x": {
                            "type": "time",
                            "title": {"display": True, "text": "Date"},
                        },
                        "y": {
                            "title": {"display": True, "text": "Drawdown (%)"},
                            "max": 0,
                            "reverse": False,
                        },
                    },
                    "plugins": {
                        "legend": {"display": True, "position": "top"},
                        "tooltip": {
                            "mode": "index",
                            "intersect": False,
                            "callbacks": {
                                "label": (
                                    "function(context) { "
                                    "return 'Drawdown: ' + "
                                    "context.parsed.y.toFixed(2) + '%'; }"
                                )
                            },
                        },
                    },
                },
            }

            return chart_config

        except Exception as e:
            logger.error(f"Failed to generate drawdown chart: {e}")
            return self._get_empty_chart("Error generating drawdown chart")

    def generate_monthly_returns_chart(self, result: BacktestResult) -> dict[str, Any]:
        """
        Generate monthly returns heatmap data.

        Args:
            result: Backtest result

        Returns:
            Chart configuration for monthly returns heatmap
        """
        try:
            monthly_returns = result.get_monthly_returns()

            if not monthly_returns:
                return self._get_empty_chart("No monthly returns data available")

            # Organize data by year and month
            years: dict[int, dict[int, float]] = {}
            for month_key, return_pct in monthly_returns.items():
                year_str, month_str = month_key.split("-")
                year = int(year_str)
                month = int(month_str)

                if year not in years:
                    years[year] = {}
                years[year][month] = return_pct

            # Create heatmap data
            heatmap_data = []
            month_names = [
                "Jan",
                "Feb",
                "Mar",
                "Apr",
                "May",
                "Jun",
                "Jul",
                "Aug",
                "Sep",
                "Oct",
                "Nov",
                "Dec",
            ]

            for year in sorted(years.keys()):
                year_data: list[float | None] = []
                for month_num in range(1, 13):
                    return_val = years[year].get(month_num, None)
                    year_data.append(return_val)
                heatmap_data.append(year_data)

            chart_config = {
                "type": "heatmap",
                "title": "Monthly Returns Heatmap",
                "subtitle": "Returns by month and year (%)",
                "data": {
                    "labels": month_names,
                    "datasets": [
                        {
                            "label": "Monthly Returns",
                            "data": heatmap_data,
                            "years": sorted(years.keys()),
                        }
                    ],
                },
                "options": {
                    "responsive": True,
                    "maintainAspectRatio": False,
                    "plugins": {
                        "legend": {"display": False},
                        "tooltip": {
                            "callbacks": {
                                "title": (
                                    "function(context) { "
                                    "return context[0].dataset.years"
                                    "[context[0].dataIndex] + ' ' + "
                                    "context[0].label; }"
                                ),
                                "label": (
                                    "function(context) { "
                                    "return 'Return: ' + "
                                    "(context.parsed.v || 0).toFixed(2) + '%'; }"
                                ),
                            }
                        },
                    },
                },
            }

            return chart_config

        except Exception as e:
            logger.error(f"Failed to generate monthly returns chart: {e}")
            return self._get_empty_chart("Error generating monthly returns chart")

    def generate_trade_analysis_chart(self, result: BacktestResult) -> dict[str, Any]:
        """
        Generate trade analysis scatter plot.

        Args:
            result: Backtest result containing trade data

        Returns:
            Chart configuration for trade analysis
        """
        try:
            if not result.trades:
                return self._get_empty_chart("No trades available for analysis")

            # Separate winning and losing trades
            winning_trades = [t for t in result.trades if t.pnl > 0]
            losing_trades = [t for t in result.trades if t.pnl <= 0]

            chart_config = {
                "type": "scatter",
                "title": "Trade Analysis",
                "subtitle": (
                    f"Win Rate: {result.metrics.win_rate:.1f}% | "
                    f"Profit Factor: {result.metrics.profit_factor:.2f}"
                ),
                "data": {
                    "datasets": [
                        {
                            "label": f"Winning Trades ({len(winning_trades)})",
                            "data": [
                                {"x": i + 1, "y": trade.pnl, "duration": trade.duration}
                                for i, trade in enumerate(winning_trades)
                            ],
                            "backgroundColor": "#10b981",
                            "borderColor": "#059669",
                        },
                        {
                            "label": f"Losing Trades ({len(losing_trades)})",
                            "data": [
                                {
                                    "x": len(winning_trades) + i + 1,
                                    "y": trade.pnl,
                                    "duration": trade.duration,
                                }
                                for i, trade in enumerate(losing_trades)
                            ],
                            "backgroundColor": "#ef4444",
                            "borderColor": "#dc2626",
                        },
                    ]
                },
                "options": {
                    "responsive": True,
                    "maintainAspectRatio": False,
                    "scales": {
                        "x": {
                            "title": {"display": True, "text": "Trade Number"},
                            "beginAtZero": True,
                        },
                        "y": {
                            "title": {"display": True, "text": "P&L ($)"},
                        },
                    },
                    "plugins": {
                        "legend": {"display": True, "position": "top"},
                        "tooltip": {
                            "callbacks": {
                                "label": (
                                    "function(context) { "
                                    "return context.dataset.label + ': $' + "
                                    "context.parsed.y.toFixed(2) + ' (Duration: ' + "
                                    "context.raw.duration + ' bars)'; }"
                                )
                            }
                        },
                    },
                },
            }

            return chart_config

        except Exception as e:
            logger.error(f"Failed to generate trade analysis chart: {e}")
            return self._get_empty_chart("Error generating trade analysis chart")

    def generate_performance_summary_chart(
        self, result: BacktestResult
    ) -> dict[str, Any]:
        """
        Generate performance summary radar chart.

        Args:
            result: Backtest result

        Returns:
            Chart configuration for performance summary
        """
        try:
            # Normalize metrics to 0-100 scale for radar chart
            metrics = result.metrics

            # Calculate normalized scores (higher is better)
            total_return_score = (
                min(max(metrics.total_return, -50), 100) + 50
            )  # -50% to 100% -> 0 to 150, capped at 100
            sharpe_score = min(
                max(metrics.sharpe_ratio * 25, 0), 100
            )  # 0 to 4 -> 0 to 100
            win_rate_score = metrics.win_rate  # Already 0-100
            profit_factor_score = min(
                max((metrics.profit_factor - 1) * 50, 0), 100
            )  # 1+ -> 0+, capped at 100
            drawdown_score = max(
                100 + metrics.max_drawdown, 0
            )  # Less drawdown is better
            exposure_score = metrics.exposure_time  # Already 0-100

            chart_config = {
                "type": "radar",
                "title": "Performance Summary",
                "subtitle": (
                    f"Overall Score: "
                    f"{
                        (
                            total_return_score
                            + sharpe_score
                            + win_rate_score
                            + profit_factor_score
                            + drawdown_score
                            + exposure_score
                        )
                        / 6:.1f}/10"
                ),
                "data": {
                    "labels": [
                        "Total Return",
                        "Sharpe Ratio",
                        "Win Rate",
                        "Profit Factor",
                        "Drawdown Control",
                        "Market Exposure",
                    ],
                    "datasets": [
                        {
                            "label": "Strategy Performance",
                            "data": [
                                total_return_score,
                                sharpe_score,
                                win_rate_score,
                                profit_factor_score,
                                drawdown_score,
                                exposure_score,
                            ],
                            "borderColor": "#3b82f6",
                            "backgroundColor": "rgba(59, 130, 246, 0.2)",
                            "pointBackgroundColor": "#3b82f6",
                            "pointBorderColor": "#1d4ed8",
                            "pointHoverBackgroundColor": "#1d4ed8",
                            "pointHoverBorderColor": "#3b82f6",
                        }
                    ],
                },
                "options": {
                    "responsive": True,
                    "maintainAspectRatio": False,
                    "scales": {
                        "r": {
                            "beginAtZero": True,
                            "max": 100,
                            "ticks": {"stepSize": 20},
                        }
                    },
                    "plugins": {
                        "legend": {"display": False},
                        "tooltip": {
                            "callbacks": {
                                "label": (
                                    "function(context) { "
                                    "return context.label + ': ' + "
                                    "context.parsed.r.toFixed(1) + '/100'; }"
                                )
                            }
                        },
                    },
                },
            }

            return chart_config

        except Exception as e:
            logger.error(f"Failed to generate performance summary chart: {e}")
            return self._get_empty_chart("Error generating performance summary")

    def _get_empty_chart(self, message: str) -> dict[str, Any]:
        """
        Get empty chart configuration with error message.

        Args:
            message: Error message to display

        Returns:
            Empty chart configuration
        """
        return {
            "type": "line",
            "title": "Chart Unavailable",
            "subtitle": message,
            "data": {"labels": [], "datasets": []},
            "options": {
                "responsive": True,
                "maintainAspectRatio": False,
                "plugins": {"legend": {"display": False}},
            },
        }


def generate_all_charts(result: BacktestResult) -> dict[str, dict[str, Any]]:
    """
    Generate all charts for a backtest result.

    Args:
        result: Backtest result

    Returns:
        Dictionary of chart configurations
    """
    generator = ChartGenerator()

    charts = {
        "equity_curve": generator.generate_equity_curve_chart(result),
        "drawdown": generator.generate_drawdown_chart(result),
        "monthly_returns": generator.generate_monthly_returns_chart(result),
        "trade_analysis": generator.generate_trade_analysis_chart(result),
        "performance_summary": generator.generate_performance_summary_chart(result),
    }

    logger.info(f"Generated {len(charts)} charts for backtest result")
    return charts


def export_chart_data(result: BacktestResult, format: str = "json") -> dict[str, Any]:
    """
    Export chart data in various formats.

    Args:
        result: Backtest result
        format: Export format ("json", "csv")

    Returns:
        Exported data
    """
    try:
        if format.lower() == "json":
            return {
                "equity_curve": result.equity_curve,
                "drawdown_curve": result.drawdown_curve,
                "trades": [trade.model_dump() for trade in result.trades],
                "metrics": result.metrics.model_dump(),
                "config": result.config.model_dump(),
            }
        elif format.lower() == "csv":
            # Convert to CSV-friendly format
            equity_df = pd.DataFrame(result.equity_curve)
            drawdown_df = pd.DataFrame(result.drawdown_curve)
            trades_df = pd.DataFrame([trade.model_dump() for trade in result.trades])

            return {
                "equity_curve_csv": equity_df.to_csv(index=False),
                "drawdown_curve_csv": drawdown_df.to_csv(index=False),
                "trades_csv": trades_df.to_csv(index=False),
            }
        else:
            raise ValueError(f"Unsupported export format: {format}")

    except Exception as e:
        logger.error(f"Failed to export chart data: {e}")
        return {"error": str(e)}
