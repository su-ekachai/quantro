"""
Backtesting module for the Quantro trading platform.

This module provides comprehensive backtesting capabilities using the backtesting.py
library, including performance metrics calculation, result storage, and chart
generation.
"""

from app.backtesting.engine import BacktestEngine
from app.backtesting.metrics import PerformanceMetrics
from app.backtesting.models import BacktestConfig, BacktestResult

__all__ = [
    "BacktestEngine",
    "BacktestConfig",
    "BacktestResult",
    "PerformanceMetrics",
]
