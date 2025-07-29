"""
Database configuration and connection management
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from loguru import logger
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import QueuePool

from app.core.config import settings


class Base(DeclarativeBase):
    """Base class for all database models"""

    pass


class DatabaseManager:
    """Database connection and session management"""

    def __init__(self) -> None:
        self._async_engine: AsyncEngine | None = None
        self._sync_engine: Engine | None = None
        self._async_session_factory: async_sessionmaker[AsyncSession] | None = None
        self._sync_session_factory: sessionmaker[Session] | None = None

    def initialize(self) -> None:
        """Initialize database engines and session factories"""
        # Async engine for main application (uses psycopg3 async)
        async_url = settings.DATABASE_URL.replace(
            "postgresql+psycopg://", "postgresql+psycopg_async://"
        )
        self._async_engine = create_async_engine(
            async_url,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            pool_recycle=3600,  # Recycle connections after 1 hour
            echo=settings.DEBUG,
        )

        # Sync engine for migrations and utilities (uses psycopg3)
        self._sync_engine = create_engine(
            settings.DATABASE_URL,
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=settings.DEBUG,
        )

        # Session factories
        self._async_session_factory = async_sessionmaker(
            bind=self._async_engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        self._sync_session_factory = sessionmaker(
            bind=self._sync_engine,
            expire_on_commit=False,
        )

        logger.info("Database engines initialized")

    @property
    def async_engine(self) -> AsyncEngine:
        """Get async database engine"""
        if self._async_engine is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        return self._async_engine

    @property
    def sync_engine(self) -> Engine:
        """Get sync database engine"""
        if self._sync_engine is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        return self._sync_engine

    @asynccontextmanager
    async def get_async_session(self) -> AsyncGenerator[AsyncSession]:
        """Get async database session"""
        if self._async_session_factory is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")

        async with self._async_session_factory() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    def get_sync_session(self) -> Session:
        """Get sync database session"""
        if self._sync_session_factory is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        return self._sync_session_factory()

    async def health_check(self) -> dict[str, str | bool]:
        """Check database connectivity and health"""
        try:
            async with self.get_async_session() as session:
                # Test basic connectivity
                result = await session.execute(text("SELECT 1"))
                result.scalar()

                # Check TimescaleDB extension
                timescale_result = await session.execute(
                    text(
                        "SELECT extname FROM pg_extension WHERE extname = 'timescaledb'"
                    )
                )
                timescale_installed = timescale_result.scalar() is not None

                return {
                    "status": "healthy",
                    "database_connected": True,
                    "timescaledb_enabled": timescale_installed,
                }
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                "status": "unhealthy",
                "database_connected": False,
                "timescaledb_enabled": False,
                "error": str(e),
            }

    async def close(self) -> None:
        """Close database connections"""
        if self._async_engine:
            await self._async_engine.dispose()
        if self._sync_engine:
            self._sync_engine.dispose()
        logger.info("Database connections closed")


# Global database manager instance
db_manager = DatabaseManager()


async def get_db_session() -> AsyncGenerator[AsyncSession]:
    """Dependency for getting database session in FastAPI routes"""
    async with db_manager.get_async_session() as session:
        yield session


async def init_database() -> None:
    """Initialize database on application startup"""
    db_manager.initialize()

    # Initialize TimescaleDB features if available
    try:
        from app.core.timescaledb import initialize_timescaledb

        await initialize_timescaledb()
    except Exception as e:
        logger.warning(f"TimescaleDB initialization skipped: {e}")

    logger.info("Database initialized successfully")


async def close_database() -> None:
    """Close database connections on application shutdown"""
    await db_manager.close()
