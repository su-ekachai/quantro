"""
Pydantic schemas package
"""

from app.schemas.auth import (
    Token,
    TokenData,
    UserCreate,
    UserInDB,
    UserResponse,
    UserUpdate,
)
from app.schemas.market import (
    MarketDataCreate,
    MarketDataResponse,
    SymbolCreate,
    SymbolResponse,
)
from app.schemas.trading import (
    PortfolioCreate,
    PortfolioResponse,
    PositionResponse,
    StrategyCreate,
    StrategyResponse,
    TradeResponse,
    TradingSignalResponse,
)

__all__ = [
    # Auth schemas
    "Token",
    "TokenData",
    "UserCreate",
    "UserInDB",
    "UserResponse",
    "UserUpdate",
    # Market schemas
    "MarketDataCreate",
    "MarketDataResponse",
    "SymbolCreate",
    "SymbolResponse",
    # Trading schemas
    "PortfolioCreate",
    "PortfolioResponse",
    "PositionResponse",
    "StrategyCreate",
    "StrategyResponse",
    "TradeResponse",
    "TradingSignalResponse",
]
