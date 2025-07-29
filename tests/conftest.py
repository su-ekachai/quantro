"""
Test configuration and fixtures
"""

from __future__ import annotations

from collections.abc import Generator
from decimal import Decimal

import pytest
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.database import Base
from app.models.market import Asset
from app.models.trading import Portfolio, Strategy
from app.models.user import User


@pytest.fixture(scope="session")
def engine() -> Engine:
    """Create test database engine"""
    # Use in-memory SQLite for tests
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def db_session(engine: Engine) -> Generator[Session]:
    """Create database session for tests"""
    # Create tables for each test
    Base.metadata.create_all(engine)

    session_factory = sessionmaker(bind=engine)
    session = session_factory()

    try:
        yield session
    finally:
        session.rollback()
        session.close()
        # Clean up tables after each test
        Base.metadata.drop_all(engine)
        Base.metadata.create_all(engine)


@pytest.fixture
def sample_user(db_session: Session) -> User:
    """Create a sample user for testing"""
    import uuid

    unique_id = str(uuid.uuid4())[:8]

    user = User(
        username=f"testuser_{unique_id}",
        email=f"test_{unique_id}@example.com",
        hashed_password="hashed_password_here",
        full_name="Test User",
        is_active=True,
        is_verified=True,
    )

    db_session.add(user)
    db_session.commit()
    return user


@pytest.fixture
def sample_asset(db_session: Session) -> Asset:
    """Create a sample asset for testing"""
    import uuid

    unique_symbol = f"BTC/USDT-{str(uuid.uuid4())[:8]}"

    asset = Asset(
        symbol=unique_symbol,
        name="Bitcoin",
        asset_class="crypto",
        exchange="binance",
        base_currency="BTC",
        quote_currency="USDT",
        min_order_size=Decimal("0.00001"),
        max_order_size=Decimal("1000"),
        price_precision=2,
        quantity_precision=5,
        is_active=True,
    )

    db_session.add(asset)
    db_session.commit()
    return asset


@pytest.fixture
def sample_portfolio(db_session: Session, sample_user: User) -> Portfolio:
    """Create a sample portfolio for testing"""
    portfolio = Portfolio(
        user_id=sample_user.id,
        name="Test Portfolio",
        description="Portfolio for testing",
        initial_balance=Decimal("10000.00"),
        current_balance=Decimal("10000.00"),
        total_pnl=Decimal("0.00"),
        total_fees=Decimal("0.00"),
        is_active=True,
    )

    db_session.add(portfolio)
    db_session.commit()
    return portfolio


@pytest.fixture
def sample_strategy(db_session: Session, sample_user: User) -> Strategy:
    """Create a sample strategy for testing"""
    strategy = Strategy(
        user_id=sample_user.id,
        name="Test Strategy",
        description="Strategy for testing",
        strategy_type="test",
        parameters='{"test": true}',
        total_trades=0,
        winning_trades=0,
        total_pnl=Decimal("0.00"),
        is_active=True,
        is_backtested=False,
    )

    db_session.add(strategy)
    db_session.commit()
    return strategy


@pytest.fixture
def multiple_assets(db_session: Session) -> list[Asset]:
    """Create multiple assets for testing"""
    import uuid

    unique_id = str(uuid.uuid4())[:8]

    assets = [
        Asset(
            symbol=f"BTC/USDT-{unique_id}",
            name="Bitcoin",
            asset_class="crypto",
            exchange="binance",
            is_active=True,
        ),
        Asset(
            symbol=f"ETH/USDT-{unique_id}",
            name="Ethereum",
            asset_class="crypto",
            exchange="binance",
            is_active=True,
        ),
        Asset(
            symbol=f"PTT-{unique_id}",
            name="PTT Public Company Limited",
            asset_class="stock",
            exchange="SET",
            is_active=True,
        ),
        Asset(
            symbol=f"GOLD-{unique_id}",
            name="Gold Futures",
            asset_class="commodity",
            exchange="COMEX",
            is_active=True,
        ),
    ]

    db_session.add_all(assets)
    db_session.commit()
    return assets
