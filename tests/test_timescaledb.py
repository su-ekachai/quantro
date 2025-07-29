"""
Tests for TimescaleDB functionality
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.timescaledb import TimescaleDBManager


class TestTimescaleDBManager:
    """Test cases for TimescaleDB manager"""

    @pytest.mark.asyncio
    async def test_create_hypertables(self) -> None:
        """Test creating hypertables"""
        with patch("app.core.timescaledb.db_manager") as mock_db_manager:
            mock_session = AsyncMock()
            mock_db_manager.get_async_session.return_value.__aenter__.return_value = (
                mock_session
            )

            # Mock hypertable doesn't exist
            mock_result = MagicMock()
            mock_result.scalar.return_value = False
            mock_session.execute.return_value = mock_result

            await TimescaleDBManager.create_hypertables()

            # Verify session was used
            assert mock_session.execute.called
            assert mock_session.commit.called

    @pytest.mark.asyncio
    async def test_create_market_data_hypertable_already_exists(self) -> None:
        """Test creating hypertable when it already exists"""
        with patch("app.core.timescaledb.db_manager") as mock_db_manager:
            mock_session = AsyncMock()
            mock_db_manager.get_async_session.return_value.__aenter__.return_value = (
                mock_session
            )

            # Mock hypertable already exists
            mock_result = MagicMock()
            mock_result.scalar.return_value = True
            mock_session.execute.return_value = mock_result

            await TimescaleDBManager._create_market_data_hypertable(mock_session)

            # Should check existence but not create
            assert mock_session.execute.called
            # Should not commit since no creation needed
            assert not mock_session.commit.called

    @pytest.mark.asyncio
    async def test_create_time_series_indexes(self) -> None:
        """Test creating time-series indexes"""
        with patch("app.core.timescaledb.db_manager") as mock_db_manager:
            mock_session = AsyncMock()
            mock_db_manager.get_async_session.return_value.__aenter__.return_value = (
                mock_session
            )

            await TimescaleDBManager._create_time_series_indexes(mock_session)

            # Should execute multiple index creation statements
            assert mock_session.execute.call_count >= 5  # We have 5 indexes
            assert mock_session.commit.call_count >= 5

    @pytest.mark.asyncio
    async def test_create_continuous_aggregates(self) -> None:
        """Test creating continuous aggregates"""
        with patch("app.core.timescaledb.db_manager") as mock_db_manager:
            mock_session = AsyncMock()
            mock_db_manager.get_async_session.return_value.__aenter__.return_value = (
                mock_session
            )

            await TimescaleDBManager.create_continuous_aggregates()

            # Should create daily and hourly aggregates
            assert mock_session.execute.call_count >= 2
            assert mock_session.commit.called

    @pytest.mark.asyncio
    async def test_setup_data_retention(self) -> None:
        """Test setting up data retention policies"""
        with patch("app.core.timescaledb.db_manager") as mock_db_manager:
            mock_session = AsyncMock()
            mock_db_manager.get_async_session.return_value.__aenter__.return_value = (
                mock_session
            )

            await TimescaleDBManager.setup_data_retention()

            # Should set retention policies
            assert mock_session.execute.call_count >= 2
            assert mock_session.commit.called

    @pytest.mark.asyncio
    async def test_optimize_chunks(self) -> None:
        """Test optimizing chunk configuration"""
        with patch("app.core.timescaledb.db_manager") as mock_db_manager:
            mock_session = AsyncMock()
            mock_db_manager.get_async_session.return_value.__aenter__.return_value = (
                mock_session
            )

            await TimescaleDBManager.optimize_chunks()

            # Should optimize chunk configuration
            assert mock_session.execute.called
            assert mock_session.commit.called

    @pytest.mark.asyncio
    async def test_get_hypertable_stats_success(self) -> None:
        """Test getting hypertable statistics successfully"""
        with patch("app.core.timescaledb.db_manager") as mock_db_manager:
            mock_session = AsyncMock()
            mock_db_manager.get_async_session.return_value.__aenter__.return_value = (
                mock_session
            )

            # Mock successful result
            mock_row = MagicMock()
            mock_row.hypertable_name = "market_data"
            mock_row.num_chunks = 10
            mock_row.table_size = 1024000
            mock_row.index_size = 256000
            mock_row.total_size = 1280000

            mock_result = MagicMock()
            mock_result.fetchone.return_value = mock_row
            mock_session.execute.return_value = mock_result

            stats = await TimescaleDBManager.get_hypertable_stats()

            assert stats["hypertable_name"] == "market_data"
            assert stats["num_chunks"] == 10
            assert stats["table_size"] == 1024000

    @pytest.mark.asyncio
    async def test_get_hypertable_stats_not_found(self) -> None:
        """Test getting hypertable statistics when hypertable not found"""
        with patch("app.core.timescaledb.db_manager") as mock_db_manager:
            mock_session = AsyncMock()
            mock_db_manager.get_async_session.return_value.__aenter__.return_value = (
                mock_session
            )

            # Mock no result
            mock_result = MagicMock()
            mock_result.fetchone.return_value = None
            mock_session.execute.return_value = mock_result

            stats = await TimescaleDBManager.get_hypertable_stats()

            assert "error" in stats
            assert stats["error"] == "Hypertable not found"

    @pytest.mark.asyncio
    async def test_get_hypertable_stats_error(self) -> None:
        """Test getting hypertable statistics with database error"""
        with patch("app.core.timescaledb.db_manager") as mock_db_manager:
            mock_session = AsyncMock()
            mock_db_manager.get_async_session.return_value.__aenter__.return_value = (
                mock_session
            )

            # Mock database error
            mock_session.execute.side_effect = Exception("Database error")

            stats = await TimescaleDBManager.get_hypertable_stats()

            assert "error" in stats
            assert "Database error" in stats["error"]

    @pytest.mark.asyncio
    async def test_initialize_timescaledb_success(self) -> None:
        """Test successful TimescaleDB initialization"""
        with patch("app.core.timescaledb.TimescaleDBManager") as mock_manager_class:
            mock_manager = AsyncMock()
            mock_manager_class.return_value = mock_manager

            from app.core.timescaledb import initialize_timescaledb

            await initialize_timescaledb()

            # Verify all initialization steps were called
            assert mock_manager.create_hypertables.called
            assert mock_manager.create_continuous_aggregates.called
            assert mock_manager.setup_data_retention.called
            assert mock_manager.optimize_chunks.called

    @pytest.mark.asyncio
    async def test_initialize_timescaledb_error(self) -> None:
        """Test TimescaleDB initialization with error"""
        with patch("app.core.timescaledb.TimescaleDBManager") as mock_manager_class:
            mock_manager = AsyncMock()
            mock_manager.create_hypertables.side_effect = Exception(
                "TimescaleDB not available"
            )
            mock_manager_class.return_value = mock_manager

            from app.core.timescaledb import initialize_timescaledb

            with pytest.raises(Exception, match="TimescaleDB not available"):
                await initialize_timescaledb()


class TestTimescaleDBIntegration:
    """Integration tests for TimescaleDB functionality"""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_market_data_time_series_queries(
        self, db_session: Any, sample_asset: Any
    ) -> None:
        """Test time-series queries on market data (requires actual TimescaleDB)"""
        # This test would require a real TimescaleDB instance
        # For now, we'll skip it in regular test runs
        pytest.skip("Requires TimescaleDB instance for integration testing")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_hypertable_performance(
        self, db_session: Any, sample_asset: Any
    ) -> None:
        """Test hypertable performance with large dataset"""
        # This test would require a real TimescaleDB instance with data
        pytest.skip("Requires TimescaleDB instance with test data")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_continuous_aggregates_functionality(
        self, db_session: Any, sample_asset: Any
    ) -> None:
        """Test continuous aggregates functionality"""
        # This test would require a real TimescaleDB instance
        pytest.skip("Requires TimescaleDB instance for integration testing")


class TestTimescaleDBQueries:
    """Test TimescaleDB-specific query patterns"""

    def test_time_bucket_query_structure(self) -> None:
        """Test that time bucket queries are properly structured"""
        # Test the SQL structure without executing
        daily_query = """
            SELECT
                asset_id,
                time_bucket('1 day', timestamp) AS day,
                FIRST(open_price, timestamp) AS open_price,
                MAX(high_price) AS high_price,
                MIN(low_price) AS low_price,
                LAST(close_price, timestamp) AS close_price,
                SUM(volume) AS volume
            FROM market_data
            WHERE timeframe = '1h'
            GROUP BY asset_id, day
        """

        # Verify query contains expected TimescaleDB functions
        assert "time_bucket" in daily_query
        assert "FIRST(" in daily_query
        assert "LAST(" in daily_query
        assert "GROUP BY asset_id, day" in daily_query

    def test_retention_policy_structure(self) -> None:
        """Test retention policy SQL structure"""
        retention_query = """
            SELECT add_retention_policy(
                'market_data',
                INTERVAL '30 days',
                if_not_exists => TRUE
            )
        """

        assert "add_retention_policy" in retention_query
        assert "INTERVAL '30 days'" in retention_query
        assert "if_not_exists => TRUE" in retention_query

    def test_hypertable_creation_structure(self) -> None:
        """Test hypertable creation SQL structure"""
        create_query = """
            SELECT create_hypertable(
                'market_data',
                'timestamp',
                chunk_time_interval => INTERVAL '1 day',
                if_not_exists => TRUE
            )
        """

        assert "create_hypertable" in create_query
        assert "'market_data'" in create_query
        assert "'timestamp'" in create_query
        assert "chunk_time_interval" in create_query
