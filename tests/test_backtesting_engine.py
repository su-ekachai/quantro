"""
Tests for the backtesting engine and related components.

This module provides comprehensive tests for the backtesting functionality,
including engine operations, metrics calculations, and chart generation.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.backtesting.charts import ChartGenerator, generate_all_charts
from app.backtesting.engine import BacktestEngine, BacktestOptimizer
from app.backtesting.metrics import MetricsCalculator, calculate_buy_hold_return
from app.backtesting.models import BacktestConfig, BacktestResult, TradeResult
from app.data_providers.base import OHLCV, AssetInfo, IDataProvider
from app.data_providers.manager import DataProviderManager


class MockStrategy:
    """Mock strategy for testing."""

    def __init__(self) -> None:
        self.trades: list = []

    def init(self) -> None:
        pass

    def next(self) -> None:
        pass


class MockDataProvider(IDataProvider):
    """Mock data provider for testing."""

    def __init__(self) -> None:
        super().__init__("mock", rate_limit_requests=1000, rate_limit_window=1.0)

    async def fetch_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        limit: int | None = None,
    ) -> list[OHLCV]:
        """Return mock OHLCV data."""
        # Generate 100 days of mock data
        dates = pd.date_range(start=start, end=end, freq="D")[:100]
        if limit:
            dates = dates[:limit]

        data = []
        base_price = 100.0

        for i, date in enumerate(dates):
            # Simple trending data with some volatility
            price = base_price + i * 0.5 + (i % 10 - 5) * 2

            ohlcv = OHLCV(
                timestamp=date,
                open=price - 0.5,
                high=price + 1.0,
                low=price - 1.0,
                close=price,
                volume=1000000.0,
            )
            data.append(ohlcv)

        return data

    async def fetch_current_price(self, symbol: str) -> float:
        """Return mock current price."""
        return 100.0

    async def get_asset_info(self, symbol: str) -> AssetInfo:
        """Return mock asset info."""
        return AssetInfo(
            symbol=symbol,
            name=f"Mock {symbol}",
            asset_type="crypto",
            exchange="mock",
        )


class MockDataProviderManager(DataProviderManager):
    """Mock data provider manager."""

    def __init__(self) -> None:
        super().__init__()
        self.provider = MockDataProvider()

    async def get_provider_for_symbol(self, symbol: str) -> IDataProvider:
        return self.provider


@pytest.fixture
def mock_data_provider_manager() -> MockDataProviderManager:
    """Create mock data provider manager."""
    return MockDataProviderManager()


@pytest.fixture
def backtest_config() -> BacktestConfig:
    """Create test backtest configuration."""
    return BacktestConfig(
        strategy_name="TestStrategy",
        strategy_class="app.strategies.cdc_action_zone.CDCActionZoneBacktestStrategy",
        symbol="BTCUSD",
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 12, 31),
        initial_cash=10000.0,
        commission=0.001,
    )


@pytest.fixture
def sample_trades() -> list[TradeResult]:
    """Create sample trade results for testing."""
    base_time = datetime(2023, 1, 1)

    trades = [
        TradeResult(
            entry_bar=10,
            exit_bar=20,
            entry_time=base_time + timedelta(days=10),
            exit_time=base_time + timedelta(days=20),
            duration=10,
            entry_price=100.0,
            exit_price=110.0,
            size=1.0,
            pnl=10.0,
            pnl_pct=0.1,
            return_pct=0.1,
            is_long=True,
        ),
        TradeResult(
            entry_bar=30,
            exit_bar=35,
            entry_time=base_time + timedelta(days=30),
            exit_time=base_time + timedelta(days=35),
            duration=5,
            entry_price=110.0,
            exit_price=105.0,
            size=1.0,
            pnl=-5.0,
            pnl_pct=-0.045,
            return_pct=-0.045,
            is_long=True,
        ),
    ]

    return trades


@pytest.fixture
def sample_equity_curve() -> pd.Series:
    """Create sample equity curve for testing."""
    dates = pd.date_range(start=datetime(2023, 1, 1), periods=100, freq="D")

    # Create trending equity curve with some drawdowns
    values = []
    base_value = 10000.0

    for i in range(100):
        # Add trend with some volatility and occasional drawdowns
        trend = i * 50
        volatility = (i % 20 - 10) * 100
        drawdown = -500 if 40 <= i <= 50 else 0  # Drawdown period

        value = base_value + trend + volatility + drawdown
        values.append(max(value, 5000))  # Prevent going too low

    return pd.Series(values, index=dates)


class TestBacktestConfig:
    """Test backtest configuration model."""

    def test_valid_config(self, backtest_config: BacktestConfig) -> None:
        """Test valid configuration creation."""
        assert backtest_config.strategy_name == "TestStrategy"
        assert backtest_config.symbol == "BTCUSD"
        assert backtest_config.initial_cash == 10000.0

    def test_invalid_date_range(self) -> None:
        """Test validation of invalid date range."""
        with pytest.raises(ValueError, match="End date must be after start date"):
            BacktestConfig(
                strategy_name="Test",
                strategy_class="test.Strategy",
                symbol="BTCUSD",
                start_date=datetime(2023, 12, 31),
                end_date=datetime(2023, 1, 1),  # End before start
            )

    def test_symbol_validation(self) -> None:
        """Test symbol validation."""
        config = BacktestConfig(
            strategy_name="Test",
            strategy_class="test.Strategy",
            symbol="  btcusd  ",  # Should be cleaned and uppercased
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
        )
        assert config.symbol == "BTCUSD"

        with pytest.raises(ValueError, match="Symbol cannot be empty"):
            BacktestConfig(
                strategy_name="Test",
                strategy_class="test.Strategy",
                symbol="",
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2023, 12, 31),
            )


class TestMetricsCalculator:
    """Test performance metrics calculation."""

    def test_calculate_total_return(
        self, sample_trades: list[TradeResult], sample_equity_curve: pd.Series
    ) -> None:
        """Test total return calculation."""
        calculator = MetricsCalculator(
            trades=sample_trades,
            equity_curve=sample_equity_curve,
            initial_cash=10000.0,
        )

        total_return = calculator._calculate_total_return()

        # Should be positive given our trending equity curve
        assert total_return > 0
        assert isinstance(total_return, float)

    def test_calculate_sharpe_ratio(
        self, sample_trades: list[TradeResult], sample_equity_curve: pd.Series
    ) -> None:
        """Test Sharpe ratio calculation."""
        calculator = MetricsCalculator(
            trades=sample_trades,
            equity_curve=sample_equity_curve,
            initial_cash=10000.0,
        )

        sharpe_ratio = calculator._calculate_sharpe_ratio()

        # Should be a reasonable value
        assert isinstance(sharpe_ratio, float)
        assert -5 <= sharpe_ratio <= 5  # Reasonable range

    def test_calculate_max_drawdown(
        self, sample_trades: list[TradeResult], sample_equity_curve: pd.Series
    ) -> None:
        """Test maximum drawdown calculation."""
        calculator = MetricsCalculator(
            trades=sample_trades,
            equity_curve=sample_equity_curve,
            initial_cash=10000.0,
        )

        max_drawdown = calculator._calculate_max_drawdown()

        # Should be negative (drawdown)
        assert max_drawdown <= 0
        assert isinstance(max_drawdown, float)

    def test_calculate_trade_statistics(
        self, sample_trades: list[TradeResult], sample_equity_curve: pd.Series
    ) -> None:
        """Test trade statistics calculation."""
        calculator = MetricsCalculator(
            trades=sample_trades,
            equity_curve=sample_equity_curve,
            initial_cash=10000.0,
        )

        stats = calculator._calculate_trade_statistics()

        assert stats["total_trades"] == 2
        assert stats["winning_trades"] == 1
        assert stats["losing_trades"] == 1
        assert stats["win_rate"] == 0.5
        assert stats["avg_win"] == 10.0
        assert stats["avg_loss"] == -5.0
        assert stats["best_trade"] == 10.0
        assert stats["worst_trade"] == -5.0

    def test_calculate_all_metrics(
        self, sample_trades: list[TradeResult], sample_equity_curve: pd.Series
    ) -> None:
        """Test comprehensive metrics calculation."""
        calculator = MetricsCalculator(
            trades=sample_trades,
            equity_curve=sample_equity_curve,
            initial_cash=10000.0,
        )

        metrics = calculator.calculate_all_metrics()

        # Verify all required fields are present
        assert hasattr(metrics, "total_return")
        assert hasattr(metrics, "sharpe_ratio")
        assert hasattr(metrics, "max_drawdown")
        assert hasattr(metrics, "total_trades")
        assert hasattr(metrics, "win_rate")

        # Verify reasonable values
        assert metrics.total_trades == 2
        assert metrics.win_rate == 50.0  # Converted to percentage
        assert metrics.max_drawdown <= 0  # Should be negative

    def test_empty_data_handling(self) -> None:
        """Test handling of empty data."""
        calculator = MetricsCalculator(
            trades=[],
            equity_curve=pd.Series(dtype=float),
            initial_cash=10000.0,
        )

        metrics = calculator.calculate_all_metrics()

        # Should return default values without errors
        assert metrics.total_trades == 0
        assert metrics.win_rate == 0.0
        assert metrics.total_return == 0.0


class TestBacktestEngine:
    """Test backtesting engine functionality."""

    @pytest.mark.asyncio
    async def test_validate_config(
        self,
        mock_data_provider_manager: MockDataProviderManager,
        backtest_config: BacktestConfig,
    ) -> None:
        """Test configuration validation."""
        engine = BacktestEngine(mock_data_provider_manager)

        # Valid config should not raise
        engine._validate_config(backtest_config)

        # Invalid config should raise
        invalid_config = backtest_config.model_copy()
        invalid_config.initial_cash = -1000

        with pytest.raises(ValueError, match="Initial cash must be positive"):
            engine._validate_config(invalid_config)

    @pytest.mark.asyncio
    async def test_load_data(
        self,
        mock_data_provider_manager: MockDataProviderManager,
        backtest_config: BacktestConfig,
    ) -> None:
        """Test data loading functionality."""
        engine = BacktestEngine(mock_data_provider_manager)

        data = await engine._load_data(backtest_config)

        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        assert all(
            col in data.columns for col in ["Open", "High", "Low", "Close", "Volume"]
        )
        assert len(data) > 0

    def test_load_strategy_class(
        self, mock_data_provider_manager: MockDataProviderManager
    ) -> None:
        """Test strategy class loading."""
        engine = BacktestEngine(mock_data_provider_manager)

        # Test loading existing strategy class
        strategy_class = engine._load_strategy_class(
            "app.strategies.cdc_action_zone.CDCActionZoneBacktestStrategy"
        )

        assert strategy_class is not None

        # Test loading non-existent class
        with pytest.raises(ValueError, match="Invalid strategy class"):
            engine._load_strategy_class("non.existent.Strategy")

    def test_configure_strategy(
        self, mock_data_provider_manager: MockDataProviderManager
    ) -> None:
        """Test strategy configuration."""
        engine = BacktestEngine(mock_data_provider_manager)

        # Load base strategy class
        base_class = engine._load_strategy_class(
            "app.strategies.cdc_action_zone.CDCActionZoneBacktestStrategy"
        )

        # Configure with parameters
        parameters = {"fast_ma_period": 10, "slow_ma_period": 30}
        configured_class = engine._configure_strategy(base_class, parameters)

        # Verify parameters are set
        assert hasattr(configured_class, "fast_ma_period")
        assert configured_class.fast_ma_period == 10
        assert configured_class.slow_ma_period == 30

    @pytest.mark.asyncio
    async def test_run_backtest_success(
        self,
        mock_data_provider_manager: MockDataProviderManager,
        backtest_config: BacktestConfig,
    ) -> None:
        """Test successful backtest execution."""
        engine = BacktestEngine(mock_data_provider_manager)

        # Mock the backtesting.py Backtest class
        with patch("app.backtesting.engine.Backtest") as mock_backtest:
            # Create mock result
            mock_result = MagicMock()
            mock_result._equity_curve = {
                "Equity": pd.Series(
                    [10000, 10500, 11000], index=pd.date_range("2023-01-01", periods=3)
                )
            }
            mock_result._trades = pd.DataFrame()  # Empty trades for simplicity

            mock_bt_instance = MagicMock()
            mock_bt_instance.run.return_value = mock_result
            mock_backtest.return_value = mock_bt_instance

            result = await engine.run_backtest(backtest_config)

            assert isinstance(result, BacktestResult)
            assert result.status == "completed"
            assert result.execution_time > 0
            assert result.data_points > 0

    @pytest.mark.asyncio
    async def test_run_backtest_failure(
        self,
        mock_data_provider_manager: MockDataProviderManager,
        backtest_config: BacktestConfig,
    ) -> None:
        """Test backtest failure handling."""
        engine = BacktestEngine(mock_data_provider_manager)

        # Mock data loading to fail
        with patch.object(
            engine, "_load_data", side_effect=RuntimeError("Data loading failed")
        ):
            result = await engine.run_backtest(backtest_config)

            assert result.status == "failed"
            assert result.error_message == "Data loading failed"
            assert result.metrics.total_trades == 0


class TestBacktestOptimizer:
    """Test backtest optimization functionality."""

    @pytest.mark.asyncio
    async def test_optimize_strategy(
        self,
        mock_data_provider_manager: MockDataProviderManager,
        backtest_config: BacktestConfig,
    ) -> None:
        """Test strategy parameter optimization."""
        engine = BacktestEngine(mock_data_provider_manager)
        optimizer = BacktestOptimizer(engine)

        parameter_ranges = {
            "fast_ma_period": tuple(range(5, 15, 2)),
            "slow_ma_period": tuple(range(20, 30, 2)),
        }

        # Mock the optimization process
        with patch("app.backtesting.engine.Backtest") as mock_backtest:
            mock_result = MagicMock()
            mock_result._strategy = MagicMock()
            mock_result._strategy.fast_ma_period = 8
            mock_result._strategy.slow_ma_period = 21
            mock_result.__getitem__ = lambda self, key: {
                "SQN": 1.5,
                "Return [%]": 15.0,
                "Sharpe Ratio": 1.2,
                "Max. Drawdown [%]": -5.0,
                "# Trades": 10,
            }[key]
            mock_result.get = lambda key, default=0: {
                "Sharpe Ratio": 1.2,
                "Max. Drawdown [%]": -5.0,
                "# Trades": 10,
            }.get(key, default)

            mock_bt_instance = MagicMock()
            mock_bt_instance.optimize.return_value = mock_result
            mock_backtest.return_value = mock_bt_instance

            result = await optimizer.optimize_strategy(
                config=backtest_config,
                parameter_ranges=parameter_ranges,
                maximize="SQN",
                max_tries=10,
            )

            assert "best_parameters" in result
            assert "best_score" in result
            assert result["best_parameters"]["fast_ma_period"] == 8
            assert result["best_parameters"]["slow_ma_period"] == 21
            assert result["best_score"] == 1.5


class TestChartGenerator:
    """Test chart generation functionality."""

    def test_generate_equity_curve_chart(self, sample_equity_curve: pd.Series) -> None:
        """Test equity curve chart generation."""
        # Create mock backtest result
        result = MagicMock()
        result.equity_curve = [
            {"date": datetime(2023, 1, 1), "equity": 10000},
            {"date": datetime(2023, 1, 2), "equity": 10500},
            {"date": datetime(2023, 1, 3), "equity": 11000},
        ]
        result.config.initial_cash = 10000
        result.config.strategy_name = "TestStrategy"
        result.config.symbol = "BTCUSD"
        result.config.start_date = datetime(2023, 1, 1)
        result.config.end_date = datetime(2023, 12, 31)
        result.metrics.buy_hold_return = 5.0

        generator = ChartGenerator()
        chart = generator.generate_equity_curve_chart(result)

        assert chart["type"] == "line"
        assert "TestStrategy" in chart["title"]
        assert len(chart["data"]["datasets"]) == 2  # Strategy + Buy & Hold
        assert len(chart["data"]["labels"]) == 3

    def test_generate_drawdown_chart(self) -> None:
        """Test drawdown chart generation."""
        result = MagicMock()
        result.drawdown_curve = [
            {"date": datetime(2023, 1, 1), "drawdown": 0},
            {"date": datetime(2023, 1, 2), "drawdown": -2.5},
            {"date": datetime(2023, 1, 3), "drawdown": -1.0},
        ]
        result.metrics.max_drawdown = -5.0

        generator = ChartGenerator()
        chart = generator.generate_drawdown_chart(result)

        assert chart["type"] == "line"
        assert "Drawdown" in chart["title"]
        assert len(chart["data"]["datasets"]) == 1
        assert len(chart["data"]["labels"]) == 3

    def test_generate_trade_analysis_chart(
        self, sample_trades: list[TradeResult]
    ) -> None:
        """Test trade analysis chart generation."""
        result = MagicMock()
        result.trades = sample_trades
        result.metrics.win_rate = 50.0
        result.metrics.profit_factor = 2.0

        generator = ChartGenerator()
        chart = generator.generate_trade_analysis_chart(result)

        assert chart["type"] == "scatter"
        assert "Trade Analysis" in chart["title"]
        assert len(chart["data"]["datasets"]) == 2  # Winning + Losing

    def test_generate_all_charts(
        self, sample_trades: list[TradeResult], sample_equity_curve: pd.Series
    ) -> None:
        """Test generation of all chart types."""
        result = MagicMock()
        result.equity_curve = [
            {"date": datetime(2023, 1, 1), "equity": 10000},
            {"date": datetime(2023, 1, 2), "equity": 10500},
        ]
        result.drawdown_curve = [
            {"date": datetime(2023, 1, 1), "drawdown": 0},
            {"date": datetime(2023, 1, 2), "drawdown": -1.0},
        ]
        result.trades = sample_trades
        result.config.initial_cash = 10000
        result.config.strategy_name = "TestStrategy"
        result.config.symbol = "BTCUSD"
        result.config.start_date = datetime(2023, 1, 1)
        result.config.end_date = datetime(2023, 12, 31)
        result.metrics.buy_hold_return = 5.0
        result.metrics.max_drawdown = -5.0
        result.metrics.win_rate = 50.0
        result.metrics.profit_factor = 2.0
        result.metrics.total_return = 10.0
        result.metrics.sharpe_ratio = 1.5
        result.metrics.exposure_time = 75.0
        result.get_monthly_returns.return_value = {"2023-01": 2.5, "2023-02": -1.0}

        charts = generate_all_charts(result)

        expected_charts = [
            "equity_curve",
            "drawdown",
            "monthly_returns",
            "trade_analysis",
            "performance_summary",
        ]
        assert all(chart_type in charts for chart_type in expected_charts)
        assert len(charts) == len(expected_charts)


class TestBuyHoldCalculation:
    """Test buy and hold return calculation."""

    def test_calculate_buy_hold_return(self) -> None:
        """Test buy and hold return calculation."""
        # Create sample data
        dates = pd.date_range(start=datetime(2023, 1, 1), periods=10, freq="D")
        data = pd.DataFrame(
            {"Close": [100, 102, 98, 105, 110, 108, 115, 120, 118, 125]}, index=dates
        )

        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 10)

        buy_hold_return = calculate_buy_hold_return(data, start_date, end_date)

        # Should be (125 - 100) / 100 = 0.25
        assert abs(buy_hold_return - 0.25) < 0.001

    def test_calculate_buy_hold_return_empty_data(self) -> None:
        """Test buy and hold calculation with empty data."""
        data = pd.DataFrame({"Close": []})
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 10)

        buy_hold_return = calculate_buy_hold_return(data, start_date, end_date)

        assert buy_hold_return == 0.0


# Integration tests
class TestBacktestingIntegration:
    """Integration tests for the complete backtesting workflow."""

    @pytest.mark.asyncio
    async def test_complete_backtest_workflow(
        self, mock_data_provider_manager: MockDataProviderManager
    ) -> None:
        """Test complete backtesting workflow from config to results."""
        config = BacktestConfig(
            strategy_name="CDC Action Zone Test",
            strategy_class="app.strategies.cdc_action_zone.CDCActionZoneBacktestStrategy",
            symbol="BTCUSD",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 3, 31),
            initial_cash=10000.0,
            commission=0.001,
            strategy_parameters={
                "fast_ma_period": 8,
                "slow_ma_period": 21,
                "min_confidence": 0.6,
            },
        )

        engine = BacktestEngine(mock_data_provider_manager)

        # Mock the backtesting library
        with patch("app.backtesting.engine.Backtest") as mock_backtest:
            # Create realistic mock result
            equity_dates = pd.date_range("2023-01-01", periods=50, freq="D")
            equity_values = [10000 + i * 100 + (i % 10 - 5) * 50 for i in range(50)]

            mock_result = MagicMock()
            mock_result._equity_curve = {
                "Equity": pd.Series(equity_values, index=equity_dates)
            }
            mock_result._trades = pd.DataFrame(
                {
                    "EntryBar": [10, 30],
                    "ExitBar": [20, 40],
                    "EntryPrice": [100.0, 110.0],
                    "ExitPrice": [110.0, 105.0],
                    "Size": [1.0, 1.0],
                    "PnL": [10.0, -5.0],
                    "ReturnPct": [0.1, -0.045],
                }
            )

            mock_bt_instance = MagicMock()
            mock_bt_instance.run.return_value = mock_result
            mock_backtest.return_value = mock_bt_instance

            # Run backtest
            result = await engine.run_backtest(config)

            # Verify result structure
            assert isinstance(result, BacktestResult)
            assert result.status == "completed"
            assert result.config.strategy_name == "CDC Action Zone Test"
            assert len(result.trades) == 2
            assert len(result.equity_curve) == 50
            assert result.metrics.total_trades == 2

            # Generate charts
            charts = generate_all_charts(result)
            assert len(charts) == 5

            # Verify chart data
            equity_chart = charts["equity_curve"]
            assert equity_chart["type"] == "line"
            assert len(equity_chart["data"]["datasets"]) == 2

            trade_chart = charts["trade_analysis"]
            assert trade_chart["type"] == "scatter"
            assert len(trade_chart["data"]["datasets"]) == 2


if __name__ == "__main__":
    pytest.main([__file__])
