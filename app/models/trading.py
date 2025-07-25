"""
Trading-related models for portfolios, positions, trades, and strategies
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    Numeric,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.market import Symbol
    from app.models.user import User


class Portfolio(Base):
    """User portfolio for tracking overall performance"""

    __tablename__ = "portfolios"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    name: Mapped[str] = mapped_column(String(100))
    description: Mapped[str | None] = mapped_column(Text)

    # Portfolio metrics
    initial_balance: Mapped[Decimal] = mapped_column(Numeric(20, 8))
    current_balance: Mapped[Decimal] = mapped_column(Numeric(20, 8))
    total_pnl: Mapped[Decimal] = mapped_column(Numeric(20, 8), default=0)
    total_fees: Mapped[Decimal] = mapped_column(Numeric(20, 8), default=0)

    # Risk management
    max_drawdown: Mapped[Decimal | None] = mapped_column(Numeric(10, 4))
    risk_per_trade: Mapped[Decimal | None] = mapped_column(Numeric(10, 4))  # Percentage
    max_positions: Mapped[int | None] = mapped_column()

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
    user: Mapped[User] = relationship("User", back_populates="portfolios")
    positions: Mapped[list[Position]] = relationship(
        "Position", back_populates="portfolio", cascade="all, delete-orphan"
    )
    trades: Mapped[list[Trade]] = relationship(
        "Trade", back_populates="portfolio", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Portfolio(id={self.id}, name='{self.name}', user_id={self.user_id})>"


class Strategy(Base):
    """Trading strategy configuration and metadata"""

    __tablename__ = "strategies"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    name: Mapped[str] = mapped_column(String(100))
    description: Mapped[str | None] = mapped_column(Text)
    strategy_type: Mapped[str] = mapped_column(
        String(50)
    )  # cdc_action_zone, custom, etc.

    # Strategy parameters (JSON string)
    parameters: Mapped[str | None] = mapped_column(Text)

    # Performance metrics
    total_trades: Mapped[int] = mapped_column(default=0)
    winning_trades: Mapped[int] = mapped_column(default=0)
    total_pnl: Mapped[Decimal] = mapped_column(Numeric(20, 8), default=0)
    max_drawdown: Mapped[Decimal | None] = mapped_column(Numeric(10, 4))
    sharpe_ratio: Mapped[Decimal | None] = mapped_column(Numeric(10, 4))

    # Status
    is_active: Mapped[bool] = mapped_column(default=True)
    is_backtested: Mapped[bool] = mapped_column(default=False)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    user: Mapped[User] = relationship("User", back_populates="strategies")
    signals: Mapped[list[TradingSignal]] = relationship(
        "TradingSignal", back_populates="strategy", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return (
            f"<Strategy(id={self.id}, name='{self.name}', type='{self.strategy_type}')>"
        )


class Position(Base):
    """Current open positions"""

    __tablename__ = "positions"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    portfolio_id: Mapped[int] = mapped_column(ForeignKey("portfolios.id"), index=True)
    symbol_id: Mapped[int] = mapped_column(ForeignKey("symbols.id"), index=True)

    # Position details
    side: Mapped[str] = mapped_column(String(10))  # long, short
    quantity: Mapped[Decimal] = mapped_column(Numeric(20, 8))
    entry_price: Mapped[Decimal] = mapped_column(Numeric(20, 8))
    current_price: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))

    # Risk management
    stop_loss: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))
    take_profit: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))

    # P&L tracking
    unrealized_pnl: Mapped[Decimal] = mapped_column(Numeric(20, 8), default=0)
    realized_pnl: Mapped[Decimal] = mapped_column(Numeric(20, 8), default=0)

    # Status
    status: Mapped[str] = mapped_column(String(20), default="open")  # open, closed

    # Timestamps
    opened_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    portfolio: Mapped[Portfolio] = relationship("Portfolio", back_populates="positions")
    symbol: Mapped[Symbol] = relationship("Symbol")

    # Indexes
    __table_args__ = (
        Index("idx_positions_portfolio_symbol", "portfolio_id", "symbol_id"),
        Index("idx_positions_status", "status"),
    )

    def __repr__(self) -> str:
        return (
            f"<Position(id={self.id}, symbol_id={self.symbol_id}, "
            f"side='{self.side}', quantity={self.quantity})>"
        )


class Trade(Base):
    """Completed trade records"""

    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    portfolio_id: Mapped[int] = mapped_column(ForeignKey("portfolios.id"), index=True)
    symbol_id: Mapped[int] = mapped_column(ForeignKey("symbols.id"), index=True)
    strategy_id: Mapped[int | None] = mapped_column(
        ForeignKey("strategies.id"), index=True
    )

    # Trade details
    side: Mapped[str] = mapped_column(String(10))  # buy, sell
    quantity: Mapped[Decimal] = mapped_column(Numeric(20, 8))
    price: Mapped[Decimal] = mapped_column(Numeric(20, 8))
    fee: Mapped[Decimal] = mapped_column(Numeric(20, 8), default=0)

    # Trade metadata
    order_type: Mapped[str] = mapped_column(String(20))  # market, limit, stop
    trade_type: Mapped[str] = mapped_column(String(20))  # entry, exit, partial

    # P&L (for exit trades)
    pnl: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))

    # External references
    external_order_id: Mapped[str | None] = mapped_column(String(100))
    exchange: Mapped[str | None] = mapped_column(String(50))

    # Timestamps
    executed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    portfolio: Mapped[Portfolio] = relationship("Portfolio", back_populates="trades")
    symbol: Mapped[Symbol] = relationship("Symbol")
    strategy: Mapped[Strategy | None] = relationship("Strategy")

    # Indexes
    __table_args__ = (
        Index("idx_trades_portfolio_symbol", "portfolio_id", "symbol_id"),
        Index("idx_trades_executed_at", "executed_at"),
        Index("idx_trades_strategy", "strategy_id"),
    )

    def __repr__(self) -> str:
        return (
            f"<Trade(id={self.id}, symbol_id={self.symbol_id}, "
            f"side='{self.side}', quantity={self.quantity}, price={self.price})>"
        )


class TradingSignal(Base):
    """Trading signals generated by strategies"""

    __tablename__ = "trading_signals"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    strategy_id: Mapped[int] = mapped_column(ForeignKey("strategies.id"), index=True)
    symbol_id: Mapped[int] = mapped_column(ForeignKey("symbols.id"), index=True)

    # Signal details
    signal_type: Mapped[str] = mapped_column(String(20))  # buy, sell, hold
    strength: Mapped[Decimal] = mapped_column(Numeric(5, 4))  # 0.0 to 1.0
    price: Mapped[Decimal] = mapped_column(Numeric(20, 8))

    # Risk management suggestions
    suggested_stop_loss: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))
    suggested_take_profit: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))
    suggested_position_size: Mapped[Decimal | None] = mapped_column(Numeric(10, 4))

    # Signal metadata
    timeframe: Mapped[str] = mapped_column(String(10))
    confidence: Mapped[Decimal] = mapped_column(Numeric(5, 4))  # 0.0 to 1.0
    notes: Mapped[str | None] = mapped_column(Text)

    # Status
    status: Mapped[str] = mapped_column(
        String(20), default="active"
    )  # active, executed, expired

    # Timestamps
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    strategy: Mapped[Strategy] = relationship("Strategy", back_populates="signals")
    symbol: Mapped[Symbol] = relationship("Symbol")

    # Indexes
    __table_args__ = (
        Index("idx_signals_strategy_symbol", "strategy_id", "symbol_id"),
        Index("idx_signals_generated_at", "generated_at"),
        Index("idx_signals_status", "status"),
    )

    def __repr__(self) -> str:
        return (
            f"<TradingSignal(id={self.id}, strategy_id={self.strategy_id}, "
            f"signal_type='{self.signal_type}', strength={self.strength})>"
        )
