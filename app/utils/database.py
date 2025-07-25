"""
Database utility functions
"""

from __future__ import annotations

from typing import Any

from loguru import logger
from sqlalchemy import text

from app.core.database import db_manager


async def create_timescale_hypertables() -> None:
    """Create TimescaleDB hypertables for time-series data"""
    try:
        async with db_manager.get_async_session() as session:
            # Check if TimescaleDB extension is available
            extensions_check = await check_database_extensions()
            if not extensions_check.get("timescaledb", False):
                logger.warning(
                    "TimescaleDB extension not found. Skipping hypertable creation."
                )
                return

            # Create hypertable for market_data
            await session.execute(
                text("""
                    SELECT create_hypertable(
                        'market_data',
                        'timestamp',
                        if_not_exists => TRUE,
                        chunk_time_interval => INTERVAL '1 day'
                    );
                """)
            )

            # Create hypertable for trading_signals
            await session.execute(
                text("""
                    SELECT create_hypertable(
                        'trading_signals',
                        'generated_at',
                        if_not_exists => TRUE,
                        chunk_time_interval => INTERVAL '1 day'
                    );
                """)
            )

            await session.commit()
            logger.info("TimescaleDB hypertables created successfully")

    except Exception as e:
        logger.error(f"Failed to create TimescaleDB hypertables: {e}")
        raise


async def setup_database_indexes() -> None:
    """Create additional database indexes for performance"""
    try:
        async with db_manager.get_async_session() as session:
            # Additional indexes for market_data (no CONCURRENTLY for hypertables)
            await session.execute(
                text("""
                    CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timeframe_time
                    ON market_data (symbol_id, timeframe, timestamp DESC);
                """)
            )

            # Additional indexes for trades (regular table, can use CONCURRENTLY)
            await session.execute(
                text("""
                    CREATE INDEX IF NOT EXISTS idx_trades_portfolio_time
                    ON trades (portfolio_id, executed_at DESC);
                """)
            )

            await session.commit()
            logger.info("Additional database indexes created successfully")

    except Exception as e:
        logger.error(f"Failed to create additional indexes: {e}")
        raise


async def check_database_extensions() -> dict[str, bool]:
    """Check if required PostgreSQL extensions are installed"""
    try:
        async with db_manager.get_async_session() as session:
            # Check TimescaleDB extension
            result = await session.execute(
                text("SELECT extname FROM pg_extension WHERE extname = 'timescaledb'")
            )
            timescaledb_installed = result.scalar() is not None

            return {
                "timescaledb": timescaledb_installed,
            }

    except Exception as e:
        logger.error(f"Failed to check database extensions: {e}")
        return {"timescaledb": False}


async def get_database_stats() -> dict[str, Any]:
    """Get basic database statistics"""
    try:
        async with db_manager.get_async_session() as session:
            # Get table row counts
            tables = [
                "users",
                "symbols",
                "market_data",
                "portfolios",
                "strategies",
                "positions",
                "trades",
                "trading_signals",
            ]

            stats = {}
            for table in tables:
                result = await session.execute(text(f"SELECT COUNT(*) FROM {table}"))
                stats[f"{table}_count"] = result.scalar()

            # Get database size
            result = await session.execute(
                text("SELECT pg_size_pretty(pg_database_size(current_database()))")
            )
            stats["database_size"] = result.scalar()

            return stats

    except Exception as e:
        logger.error(f"Failed to get database statistics: {e}")
        return {"error": str(e)}
