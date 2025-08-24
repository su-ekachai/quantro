"""
Backtesting engine for running strategy backtests using backtesting.py library.

This module provides the main BacktestEngine class that integrates with
the backtesting.py library to run comprehensive backtests with performance
analysis and result storage.
"""

from __future__ import annotations

import importlib
import time
from datetime import datetime
from typing import Any

import pandas as pd
from backtesting import Backtest
from backtesting.backtesting import Strategy
from loguru import logger

from app.backtesting.metrics import MetricsCalculator, calculate_buy_hold_return
from app.backtesting.models import (
    BacktestConfig,
    BacktestResult,
    PerformanceMetrics,
    TradeResult,
)
from app.data_providers.manager import DataProviderManager


class BacktestEngine:
    """
    Main backtesting engine for running strategy backtests.

    This class provides a high-level interface for running backtests using
    the backtesting.py library, with comprehensive performance analysis
    and result storage capabilities.
    """

    def __init__(self, data_provider_manager: DataProviderManager) -> None:
        """
        Initialize the backtest engine.

        Args:
            data_provider_manager: Manager for data provider access
        """
        self.data_provider_manager = data_provider_manager

    async def run_backtest(self, config: BacktestConfig) -> BacktestResult:
        """
        Run a complete backtest with the given configuration.

        Args:
            config: Backtest configuration

        Returns:
            Complete backtest results with metrics and trade details

        Raises:
            ValueError: If configuration is invalid
            RuntimeError: If backtest execution fails
        """
        start_time = time.time()
        started_at = datetime.now()

        try:
            logger.info(
                f"Starting backtest for {config.strategy_name} on {config.symbol}"
            )

            # Validate configuration
            self._validate_config(config)

            # Load historical data
            data = await self._load_data(config)

            # Load strategy class
            strategy_class = self._load_strategy_class(config.strategy_class)

            # Configure strategy parameters
            configured_strategy = self._configure_strategy(
                strategy_class, config.strategy_parameters
            )

            # Run backtest
            bt = Backtest(
                data=data,
                strategy=configured_strategy,
                cash=config.initial_cash,
                commission=config.commission,
                margin=config.margin,
                trade_on_open=config.trade_on_open,
                hedging=config.hedging,
                exclusive_orders=config.exclusive_orders,
            )

            logger.info(f"Running backtest with {len(data)} data points")
            result = bt.run()

            # Calculate additional metrics
            buy_hold_return = calculate_buy_hold_return(
                data, config.start_date, config.end_date
            )

            # Process results
            backtest_result = await self._process_results(
                config=config,
                bt_result=result,
                data=data,
                buy_hold_return=buy_hold_return,
                execution_time=time.time() - start_time,
                started_at=started_at,
            )

            logger.info(
                f"Backtest completed successfully in "
                f"{backtest_result.execution_time:.2f}s"
            )
            logger.info(
                f"Total return: {backtest_result.metrics.total_return:.2f}%, "
                f"Sharpe ratio: {backtest_result.metrics.sharpe_ratio:.2f}"
            )

            return backtest_result

        except Exception as e:
            logger.error(f"Backtest failed: {e}")

            # Return failed result
            return BacktestResult(
                config=config,
                metrics=self._get_empty_metrics(),
                execution_time=time.time() - start_time,
                started_at=started_at,
                completed_at=datetime.now(),
                status="failed",
                error_message=str(e),
                data_points=0,
            )

    def _validate_config(self, config: BacktestConfig) -> None:
        """
        Validate backtest configuration.

        Args:
            config: Configuration to validate

        Raises:
            ValueError: If configuration is invalid
        """
        if config.end_date <= config.start_date:
            raise ValueError("End date must be after start date")

        if config.initial_cash <= 0:
            raise ValueError("Initial cash must be positive")

        if not (0 <= config.commission <= 1):
            raise ValueError("Commission must be between 0 and 1")

        if config.margin <= 0:
            raise ValueError("Margin must be positive")

        # Validate date range is not too long (prevent memory issues)
        days_diff = (config.end_date - config.start_date).days
        if days_diff > 3650:  # 10 years
            logger.warning(f"Long backtest period: {days_diff} days")

    async def _load_data(self, config: BacktestConfig) -> pd.DataFrame:
        """
        Load historical data for backtesting.

        Args:
            config: Backtest configuration

        Returns:
            OHLCV data formatted for backtesting.py

        Raises:
            RuntimeError: If data loading fails
        """
        try:
            # Get appropriate data provider
            provider = await self.data_provider_manager.get_provider_for_symbol(
                config.symbol
            )

            # Fetch historical data
            ohlcv_data = await provider.fetch_historical_data(
                symbol=config.symbol,
                timeframe=config.timeframe,
                start=config.start_date,
                end=config.end_date,
            )

            if not ohlcv_data:
                raise RuntimeError(f"No data available for {config.symbol}")

            # Convert to DataFrame format expected by backtesting.py
            df_data = []
            timestamps = []

            for ohlcv in ohlcv_data:
                df_data.append(
                    {
                        "Open": float(ohlcv.open),
                        "High": float(ohlcv.high),
                        "Low": float(ohlcv.low),
                        "Close": float(ohlcv.close),
                        "Volume": float(ohlcv.volume),
                    }
                )
                timestamps.append(ohlcv.timestamp)

            df = pd.DataFrame(df_data)
            df.index = pd.to_datetime(timestamps)

            # Validate data quality
            if df.empty:
                raise RuntimeError("Empty dataset")

            if df.isnull().any().any():
                logger.warning("Data contains null values, forward filling")
                df = df.fillna(method="ffill")

            # Ensure required columns exist
            required_columns = ["Open", "High", "Low", "Close", "Volume"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise RuntimeError(f"Missing required columns: {missing_columns}")

            logger.info(
                f"Loaded {len(df)} data points from {df.index[0]} to {df.index[-1]}"
            )
            return df

        except Exception as e:
            logger.error(f"Failed to load data for {config.symbol}: {e}")
            raise RuntimeError(f"Data loading failed: {e}") from e

    def _load_strategy_class(self, strategy_class_path: str) -> type[Strategy]:
        """
        Dynamically load strategy class.

        Args:
            strategy_class_path: Fully qualified class path (e.g.,
                'app.strategies.cdc_action_zone.CDCActionZoneBacktestStrategy')

        Returns:
            Strategy class

        Raises:
            ValueError: If strategy class cannot be loaded
        """
        try:
            module_path, class_name = strategy_class_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            strategy_class = getattr(module, class_name)

            # Validate it's a Strategy subclass
            if not issubclass(strategy_class, Strategy):
                raise ValueError(f"Class {class_name} is not a Strategy subclass")

            logger.info(f"Loaded strategy class: {strategy_class_path}")
            return strategy_class

        except Exception as e:
            logger.error(f"Failed to load strategy class {strategy_class_path}: {e}")
            raise ValueError(f"Invalid strategy class: {strategy_class_path}") from e

    def _configure_strategy(
        self, strategy_class: type[Strategy], parameters: dict[str, Any]
    ) -> type[Strategy]:
        """
        Configure strategy class with parameters.

        Args:
            strategy_class: Base strategy class
            parameters: Strategy parameters to set

        Returns:
            Configured strategy class
        """

        # Create a new class that inherits from the strategy class
        class ConfiguredStrategy(strategy_class):
            pass

        # Set parameters as class attributes
        for param_name, param_value in parameters.items():
            if hasattr(strategy_class, param_name):
                setattr(ConfiguredStrategy, param_name, param_value)
                logger.debug(f"Set strategy parameter {param_name} = {param_value}")
            else:
                logger.warning(f"Unknown strategy parameter: {param_name}")

        return ConfiguredStrategy

    async def _process_results(
        self,
        config: BacktestConfig,
        bt_result: Any,
        data: pd.DataFrame,
        buy_hold_return: float,
        execution_time: float,
        started_at: datetime,
    ) -> BacktestResult:
        """
        Process backtesting.py results into our format.

        Args:
            config: Backtest configuration
            bt_result: Raw backtesting.py result
            data: Original OHLCV data
            buy_hold_return: Buy and hold return for comparison
            execution_time: Backtest execution time
            started_at: Backtest start timestamp

        Returns:
            Processed backtest result
        """
        completed_at = datetime.now()

        # Extract equity curve
        equity_curve = bt_result._equity_curve["Equity"]

        # Extract trades
        trades = self._extract_trades(bt_result._trades, data.index)

        # Calculate comprehensive metrics
        metrics_calculator = MetricsCalculator(
            trades=trades,
            equity_curve=equity_curve,
            initial_cash=config.initial_cash,
        )

        metrics = metrics_calculator.calculate_all_metrics(buy_hold_return)

        # Create equity curve data
        equity_curve_data = [
            {
                "date": timestamp,
                "equity": float(equity),
                "drawdown": float(
                    (equity - equity_curve[: i + 1].max()) / equity_curve[: i + 1].max()
                    if i > 0
                    else 0
                ),
            }
            for i, (timestamp, equity) in enumerate(equity_curve.items())
        ]

        # Create drawdown curve data
        running_max = equity_curve.expanding().max()
        drawdown_pct = (equity_curve - running_max) / running_max * 100

        drawdown_curve_data = [
            {
                "date": timestamp,
                "drawdown": float(drawdown),
            }
            for timestamp, drawdown in drawdown_pct.items()
        ]

        return BacktestResult(
            config=config,
            metrics=metrics,
            trades=trades,
            equity_curve=equity_curve_data,
            drawdown_curve=drawdown_curve_data,
            execution_time=execution_time,
            data_points=len(data),
            started_at=started_at,
            completed_at=completed_at,
            status="completed",
        )

    def _extract_trades(
        self, bt_trades: pd.DataFrame, data_index: pd.DatetimeIndex
    ) -> list[TradeResult]:
        """
        Extract trade results from backtesting.py format.

        Args:
            bt_trades: Raw trades DataFrame from backtesting.py
            data_index: Original data datetime index

        Returns:
            List of processed trade results
        """
        trades: list[TradeResult] = []

        if bt_trades.empty:
            return trades

        for _, trade in bt_trades.iterrows():
            try:
                # Extract trade information
                entry_bar = int(trade["EntryBar"])
                exit_bar = int(trade["ExitBar"])

                # Get timestamps from data index
                entry_time = data_index[entry_bar]
                exit_time = data_index[exit_bar]

                # Calculate duration in bars (convert to days for consistency)
                duration = exit_bar - entry_bar

                trade_result = TradeResult(
                    entry_bar=entry_bar,
                    exit_bar=exit_bar,
                    entry_time=entry_time,
                    exit_time=exit_time,
                    duration=duration,
                    entry_price=float(trade["EntryPrice"]),
                    exit_price=float(trade["ExitPrice"]),
                    size=float(trade["Size"]),
                    pnl=float(trade["PnL"]),
                    pnl_pct=float(trade["ReturnPct"]),
                    return_pct=float(trade["ReturnPct"]),
                    is_long=float(trade["Size"]) > 0,
                    exit_reason=None,  # backtesting.py doesn't provide this
                )

                trades.append(trade_result)

            except Exception as e:
                logger.warning(f"Failed to process trade: {e}")
                continue

        logger.info(f"Extracted {len(trades)} trades from backtest results")
        return trades

    def _get_empty_metrics(self) -> PerformanceMetrics:
        """Get empty metrics for failed backtests."""
        from app.backtesting.models import PerformanceMetrics

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


class BacktestOptimizer:
    """
    Optimizer for running parameter optimization on strategies.

    This class provides functionality to optimize strategy parameters
    using the backtesting.py optimization capabilities.
    """

    def __init__(self, engine: BacktestEngine) -> None:
        """
        Initialize the optimizer.

        Args:
            engine: Backtest engine instance
        """
        self.engine = engine

    async def optimize_strategy(
        self,
        config: BacktestConfig,
        parameter_ranges: dict[str, tuple[Any, ...]],
        maximize: str = "SQN",  # System Quality Number
        max_tries: int = 100,
        random_state: int | None = None,
    ) -> dict[str, Any]:
        """
        Optimize strategy parameters.

        Args:
            config: Base backtest configuration
            parameter_ranges: Dictionary of parameter names to ranges/values to test
            maximize: Metric to maximize ('Return', 'Sharpe Ratio', 'SQN', etc.)
            max_tries: Maximum optimization iterations
            random_state: Random seed for reproducible results

        Returns:
            Dictionary with optimization results

        Raises:
            RuntimeError: If optimization fails
        """
        try:
            logger.info(f"Starting parameter optimization for {config.strategy_name}")

            # Load data and strategy
            data = await self.engine._load_data(config)
            strategy_class = self.engine._load_strategy_class(config.strategy_class)

            # Create backtest instance
            bt = Backtest(
                data=data,
                strategy=strategy_class,
                cash=config.initial_cash,
                commission=config.commission,
                margin=config.margin,
                trade_on_open=config.trade_on_open,
                hedging=config.hedging,
                exclusive_orders=config.exclusive_orders,
            )

            # Run optimization
            logger.info(
                f"Optimizing {len(parameter_ranges)} parameters with "
                f"max {max_tries} tries"
            )

            optimization_result = bt.optimize(
                **parameter_ranges,
                maximize=maximize,
                max_tries=max_tries,
                random_state=random_state,
            )

            # Extract results
            best_params = {}
            for param_name in parameter_ranges:
                if hasattr(optimization_result._strategy, param_name):
                    best_params[param_name] = getattr(
                        optimization_result._strategy, param_name
                    )

            result = {
                "best_parameters": best_params,
                "best_score": float(optimization_result[maximize]),
                "total_return": float(optimization_result["Return [%]"]),
                "sharpe_ratio": float(optimization_result.get("Sharpe Ratio", 0)),
                "max_drawdown": float(optimization_result.get("Max. Drawdown [%]", 0)),
                "total_trades": int(optimization_result.get("# Trades", 0)),
            }

            logger.info(
                f"Optimization completed. Best {maximize}: {result['best_score']:.4f}"
            )
            logger.info(f"Best parameters: {best_params}")

            return result

        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            raise RuntimeError(f"Parameter optimization failed: {e}") from e
