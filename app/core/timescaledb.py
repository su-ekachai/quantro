"""
TimescaleDB utilities for time-series optimization
"""

from __future__ import annotations

from typing import Any

from loguru import logger
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import db_manager


class TimescaleDBManager:
    """Manager for TimescaleDB hypertables and time-series optimizations"""

    @staticmethod
    async def create_hypertables() -> None:
        """Create hypertables for time-series data optimization"""
        async with db_manager.get_async_session() as session:
            try:
                # Create hypertable for market_data if it doesn't exist
                await TimescaleDBManager._create_market_data_hypertable(session)

                # Create additional indexes for time-series queries
                await TimescaleDBManager._create_time_series_indexes(session)

                logger.info("TimescaleDB hypertables created successfully")

            except Exception as e:
                logger.error(f"Failed to create TimescaleDB hypertables: {e}")
                raise

    @staticmethod
    async def _create_market_data_hypertable(session: AsyncSession) -> None:
        """Create hypertable for market_data table"""
        # Check if hypertable already exists
        check_query = text("""
            SELECT EXISTS (
                SELECT 1 FROM timescaledb_information.hypertables
                WHERE hypertable_name = 'market_data'
            )
        """)

        result = await session.execute(check_query)
        hypertable_exists = result.scalar()

        if not hypertable_exists:
            # Create hypertable with timestamp as the time dimension
            create_hypertable_query = text("""
                SELECT create_hypertable(
                    'market_data',
                    'timestamp',
                    chunk_time_interval => INTERVAL '1 day',
                    if_not_exists => TRUE
                )
            """)

            await session.execute(create_hypertable_query)
            await session.commit()
            logger.info("Created hypertable for market_data")
        else:
            logger.info("Hypertable for market_data already exists")

    @staticmethod
    async def _create_time_series_indexes(session: AsyncSession) -> None:
        """Create additional indexes optimized for time-series queries"""
        indexes = [
            # Composite index for asset + timestamp queries (most common)
            """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_market_data_asset_timestamp_desc
            ON market_data (asset_id, timestamp DESC)
            """,
            # Index for timeframe-specific queries
            """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_market_data_timeframe_timestamp
            ON market_data (timeframe, timestamp DESC)
            """,
            # Index for OHLCV queries by asset and timeframe
            """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS
            idx_market_data_asset_timeframe_timestamp
            ON market_data (asset_id, timeframe, timestamp DESC)
            """,
            # Index for volume-based queries
            """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_market_data_volume
            ON market_data (volume DESC) WHERE volume > 0
            """,
            # Index for source-based queries
            """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_market_data_source_timestamp
            ON market_data (source, timestamp DESC)
            """,
        ]

        for index_query in indexes:
            try:
                await session.execute(text(index_query))
                await session.commit()
            except Exception as e:
                # Index might already exist, log but don't fail
                logger.warning(f"Index creation warning: {e}")

        logger.info("Time-series indexes created/verified")

    @staticmethod
    async def create_continuous_aggregates() -> None:
        """Create continuous aggregates for common time-series queries"""
        async with db_manager.get_async_session() as session:
            try:
                # Daily OHLCV aggregate
                daily_ohlcv_query = text("""
                    CREATE MATERIALIZED VIEW IF NOT EXISTS market_data_daily
                    WITH (timescaledb.continuous) AS
                    SELECT
                        asset_id,
                        time_bucket('1 day', timestamp) AS day,
                        FIRST(open_price, timestamp) AS open_price,
                        MAX(high_price) AS high_price,
                        MIN(low_price) AS low_price,
                        LAST(close_price, timestamp) AS close_price,
                        SUM(volume) AS volume,
                        AVG(close_price) AS avg_price,
                        COUNT(*) AS data_points
                    FROM market_data
                    WHERE timeframe = '1h'  -- Aggregate from hourly data
                    GROUP BY asset_id, day
                """)

                await session.execute(daily_ohlcv_query)

                # Hourly OHLCV aggregate from minute data
                hourly_ohlcv_query = text("""
                    CREATE MATERIALIZED VIEW IF NOT EXISTS market_data_hourly
                    WITH (timescaledb.continuous) AS
                    SELECT
                        asset_id,
                        time_bucket('1 hour', timestamp) AS hour,
                        FIRST(open_price, timestamp) AS open_price,
                        MAX(high_price) AS high_price,
                        MIN(low_price) AS low_price,
                        LAST(close_price, timestamp) AS close_price,
                        SUM(volume) AS volume,
                        AVG(close_price) AS avg_price,
                        COUNT(*) AS data_points
                    FROM market_data
                    WHERE timeframe IN ('1m', '5m', '15m')  -- Aggregate from minute
                    GROUP BY asset_id, hour
                """)

                await session.execute(hourly_ohlcv_query)
                await session.commit()

                logger.info("Continuous aggregates created successfully")

            except Exception as e:
                logger.error(f"Failed to create continuous aggregates: {e}")
                raise

    @staticmethod
    async def setup_data_retention() -> None:
        """Set up data retention policies for old time-series data"""
        async with db_manager.get_async_session() as session:
            try:
                # Keep raw minute data for 30 days
                minute_retention_query = text("""
                    SELECT add_retention_policy(
                        'market_data',
                        INTERVAL '30 days',
                        if_not_exists => TRUE
                    )
                    WHERE EXISTS (
                        SELECT 1 FROM market_data
                        WHERE timeframe IN ('1m', '5m')
                        LIMIT 1
                    )
                """)

                # Keep hourly data for 1 year
                hourly_retention_query = text("""
                    SELECT add_retention_policy(
                        'market_data',
                        INTERVAL '1 year',
                        if_not_exists => TRUE
                    )
                    WHERE EXISTS (
                        SELECT 1 FROM market_data
                        WHERE timeframe = '1h'
                        LIMIT 1
                    )
                """)

                # Keep daily data indefinitely (no retention policy)

                await session.execute(minute_retention_query)
                await session.execute(hourly_retention_query)
                await session.commit()

                logger.info("Data retention policies configured")

            except Exception as e:
                logger.error(f"Failed to setup data retention: {e}")
                raise

    @staticmethod
    async def optimize_chunks() -> None:
        """Optimize chunk configuration for better performance"""
        async with db_manager.get_async_session() as session:
            try:
                # Set chunk time interval based on data frequency
                chunk_config_query = text("""
                    SELECT set_chunk_time_interval('market_data', INTERVAL '1 day')
                """)

                await session.execute(chunk_config_query)
                await session.commit()

                logger.info("Chunk configuration optimized")

            except Exception as e:
                logger.error(f"Failed to optimize chunks: {e}")
                raise

    @staticmethod
    async def get_hypertable_stats() -> dict[str, Any]:
        """Get statistics about hypertables"""
        async with db_manager.get_async_session() as session:
            try:
                stats_query = text("""
                    SELECT
                        hypertable_name,
                        num_chunks,
                        table_size,
                        index_size,
                        total_size
                    FROM timescaledb_information.hypertables h
                    LEFT JOIN timescaledb_information.hypertable_stats s
                        ON h.hypertable_name = s.hypertable_name
                    WHERE h.hypertable_name = 'market_data'
                """)

                result = await session.execute(stats_query)
                row = result.fetchone()

                if row:
                    return {
                        "hypertable_name": row.hypertable_name,
                        "num_chunks": row.num_chunks,
                        "table_size": row.table_size,
                        "index_size": row.index_size,
                        "total_size": row.total_size,
                    }
                else:
                    return {"error": "Hypertable not found"}

            except Exception as e:
                logger.error(f"Failed to get hypertable stats: {e}")
                return {"error": str(e)}


async def initialize_timescaledb() -> None:
    """Initialize TimescaleDB features for the application"""
    try:
        timescale_manager = TimescaleDBManager()

        # Create hypertables and indexes
        await timescale_manager.create_hypertables()

        # Create continuous aggregates for common queries
        await timescale_manager.create_continuous_aggregates()

        # Setup data retention policies
        await timescale_manager.setup_data_retention()

        # Optimize chunk configuration
        await timescale_manager.optimize_chunks()

        logger.info("TimescaleDB initialization completed successfully")

    except Exception as e:
        logger.error(f"TimescaleDB initialization failed: {e}")
        raise
