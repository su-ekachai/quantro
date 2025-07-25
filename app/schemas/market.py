"""
Market data-related Pydantic schemas
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, Field


class SymbolBase(BaseModel):
    """Base symbol schema"""

    symbol: str = Field(..., max_length=20)
    name: str = Field(..., max_length=255)
    asset_class: str = Field(..., max_length=20)
    exchange: str = Field(..., max_length=50)
    base_currency: str | None = Field(None, max_length=10)
    quote_currency: str | None = Field(None, max_length=10)


class SymbolCreate(SymbolBase):
    """Schema for symbol creation"""

    min_order_size: Decimal | None = None
    max_order_size: Decimal | None = None
    price_precision: int | None = None
    quantity_precision: int | None = None


class SymbolResponse(SymbolBase):
    """Schema for symbol data in API responses"""

    id: int
    min_order_size: Decimal | None = None
    max_order_size: Decimal | None = None
    price_precision: int | None = None
    quantity_precision: int | None = None
    is_active: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class MarketDataBase(BaseModel):
    """Base market data schema"""

    timestamp: datetime
    timeframe: str = Field(..., max_length=10)
    open_price: Decimal = Field(..., gt=0)
    high_price: Decimal = Field(..., gt=0)
    low_price: Decimal = Field(..., gt=0)
    close_price: Decimal = Field(..., gt=0)
    volume: Decimal = Field(..., ge=0)


class MarketDataCreate(MarketDataBase):
    """Schema for market data creation"""

    symbol_id: int
    trades_count: int | None = None
    vwap: Decimal | None = None
    source: str = Field(..., max_length=50)


class MarketDataResponse(MarketDataBase):
    """Schema for market data in API responses"""

    id: int
    symbol_id: int
    trades_count: int | None = None
    vwap: Decimal | None = None
    source: str
    created_at: datetime

    class Config:
        from_attributes = True
