"""
Trading-related Pydantic schemas
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, Field


class PortfolioBase(BaseModel):
    """Base portfolio schema"""

    name: str = Field(..., max_length=100)
    description: str | None = None
    initial_balance: Decimal = Field(..., gt=0)


class PortfolioCreate(PortfolioBase):
    """Schema for portfolio creation"""

    max_drawdown: Decimal | None = Field(None, ge=0, le=1)
    risk_per_trade: Decimal | None = Field(None, ge=0, le=1)
    max_positions: int | None = Field(None, gt=0)


class PortfolioResponse(PortfolioBase):
    """Schema for portfolio data in API responses"""

    id: int
    user_id: int
    current_balance: Decimal
    total_pnl: Decimal
    total_fees: Decimal
    max_drawdown: Decimal | None = None
    risk_per_trade: Decimal | None = None
    max_positions: int | None = None
    is_active: bool
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class StrategyBase(BaseModel):
    """Base strategy schema"""

    name: str = Field(..., max_length=100)
    description: str | None = None
    strategy_type: str = Field(..., max_length=50)


class StrategyCreate(StrategyBase):
    """Schema for strategy creation"""

    parameters: dict | None = None


class StrategyResponse(StrategyBase):
    """Schema for strategy data in API responses"""

    id: int
    user_id: int
    parameters: dict | None = None
    total_trades: int
    winning_trades: int
    total_pnl: Decimal
    max_drawdown: Decimal | None = None
    sharpe_ratio: Decimal | None = None
    is_active: bool
    is_backtested: bool
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class PositionResponse(BaseModel):
    """Schema for position data in API responses"""

    id: int
    portfolio_id: int
    symbol_id: int
    side: str
    quantity: Decimal
    entry_price: Decimal
    current_price: Decimal | None = None
    stop_loss: Decimal | None = None
    take_profit: Decimal | None = None
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    status: str
    opened_at: datetime
    closed_at: datetime | None = None
    updated_at: datetime

    model_config = {"from_attributes": True}


class TradeResponse(BaseModel):
    """Schema for trade data in API responses"""

    id: int
    portfolio_id: int
    symbol_id: int
    strategy_id: int | None = None
    side: str
    quantity: Decimal
    price: Decimal
    fee: Decimal
    order_type: str
    trade_type: str
    pnl: Decimal | None = None
    external_order_id: str | None = None
    exchange: str | None = None
    executed_at: datetime
    created_at: datetime

    model_config = {"from_attributes": True}


class TradingSignalResponse(BaseModel):
    """Schema for trading signal data in API responses"""

    id: int
    strategy_id: int
    symbol_id: int
    signal_type: str
    strength: Decimal
    price: Decimal
    suggested_stop_loss: Decimal | None = None
    suggested_take_profit: Decimal | None = None
    suggested_position_size: Decimal | None = None
    timeframe: str
    confidence: Decimal
    notes: str | None = None
    status: str
    generated_at: datetime
    expires_at: datetime | None = None
    created_at: datetime

    model_config = {"from_attributes": True}
