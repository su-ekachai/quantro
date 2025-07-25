"""
Market data models for storing price and volume information
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    DateTime,
    Index,
    Numeric,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.core.database import Base


class Symbol(Base):
    """Symbol/instrument definition"""

    __tablename__ = "symbols"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    symbol: Mapped[str] = mapped_column(String(20), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(255))
    asset_class: Mapped[str] = mapped_column(String(20))  # crypto, stock, commodity
    exchange: Mapped[str] = mapped_column(String(50))
    base_currency: Mapped[str | None] = mapped_column(String(10))
    quote_currency: Mapped[str | None] = mapped_column(String(10))

    # Trading parameters
    min_order_size: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))
    max_order_size: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))
    price_precision: Mapped[int | None] = mapped_column()
    quantity_precision: Mapped[int | None] = mapped_column()

    # Status
    is_active: Mapped[bool] = mapped_column(default=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    market_data: Mapped[list[MarketData]] = relationship(
        "MarketData", back_populates="symbol", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Symbol(id={self.id}, symbol='{self.symbol}')>"


class MarketData(Base):
    """Time-series market data (OHLCV)"""

    __tablename__ = "market_data"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    symbol_id: Mapped[int] = mapped_column(index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    timeframe: Mapped[str] = mapped_column(String(10), index=True)  # 1m, 5m, 1h, 1d

    # OHLCV data
    open_price: Mapped[Decimal] = mapped_column(Numeric(20, 8))
    high_price: Mapped[Decimal] = mapped_column(Numeric(20, 8))
    low_price: Mapped[Decimal] = mapped_column(Numeric(20, 8))
    close_price: Mapped[Decimal] = mapped_column(Numeric(20, 8))
    volume: Mapped[Decimal] = mapped_column(Numeric(20, 8))

    # Additional data
    trades_count: Mapped[int | None] = mapped_column()
    vwap: Mapped[Decimal | None] = mapped_column(
        Numeric(20, 8)
    )  # Volume Weighted Average Price

    # Data source tracking
    source: Mapped[str] = mapped_column(String(50))  # ccxt, set_api, yahoo

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    symbol: Mapped[Symbol] = relationship("Symbol", back_populates="market_data")

    # Constraints
    __table_args__ = (
        UniqueConstraint("symbol_id", "timestamp", "timeframe", name="uq_market_data"),
        Index("idx_market_data_symbol_time", "symbol_id", "timestamp"),
        Index("idx_market_data_timeframe", "timeframe"),
    )

    def __repr__(self) -> str:
        return (
            f"<MarketData(symbol_id={self.symbol_id}, "
            f"timestamp={self.timestamp}, timeframe='{self.timeframe}')>"
        )
