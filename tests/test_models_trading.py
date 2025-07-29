"""
Tests for trading models
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from sqlalchemy.orm import Session

from app.models.market import Asset
from app.models.trading import Portfolio, Position, Signal, Strategy, Trade
from app.models.user import User


class TestPortfolioModel:
    """Test cases for Portfolio model"""

    def test_create_portfolio(self, db_session: Session, sample_user: User) -> None:
        """Test creating a portfolio"""
        portfolio = Portfolio(
            user_id=sample_user.id,
            name="Main Portfolio",
            description="Primary trading portfolio",
            initial_balance=Decimal("10000.00"),
            current_balance=Decimal("10000.00"),
            total_pnl=Decimal("0.00"),
            total_fees=Decimal("0.00"),
            max_drawdown=Decimal("0.05"),  # 5%
            risk_per_trade=Decimal("0.02"),  # 2%
            max_positions=10,
            is_active=True,
        )

        db_session.add(portfolio)
        db_session.commit()

        assert portfolio.id is not None
        assert portfolio.user_id == sample_user.id
        assert portfolio.name == "Main Portfolio"
        assert portfolio.initial_balance == Decimal("10000.00")
        assert portfolio.is_active is True

    def test_portfolio_user_relationship(
        self, db_session: Session, sample_user: User
    ) -> None:
        """Test Portfolio relationship with User"""
        portfolio = Portfolio(
            user_id=sample_user.id,
            name="Test Portfolio",
            initial_balance=Decimal("5000.00"),
            current_balance=Decimal("5000.00"),
        )

        db_session.add(portfolio)
        db_session.commit()

        assert portfolio.user is not None
        assert portfolio.user.id == sample_user.id
        assert portfolio.user.username == sample_user.username

    def test_portfolio_repr(self, db_session: Session, sample_user: User) -> None:
        """Test Portfolio string representation"""
        portfolio = Portfolio(
            user_id=sample_user.id,
            name="Test Portfolio",
            initial_balance=Decimal("5000.00"),
            current_balance=Decimal("5000.00"),
        )

        db_session.add(portfolio)
        db_session.commit()

        expected = (
            f"<Portfolio(id={portfolio.id}, name='Test Portfolio', "
            f"user_id={sample_user.id})>"
        )
        assert repr(portfolio) == expected


class TestStrategyModel:
    """Test cases for Strategy model"""

    def test_create_strategy(self, db_session: Session, sample_user: User) -> None:
        """Test creating a trading strategy"""
        strategy = Strategy(
            user_id=sample_user.id,
            name="CDC Action Zone",
            description="CDC Action Zone trading strategy",
            strategy_type="cdc_action_zone",
            parameters='{"lookback": 20, "threshold": 0.02}',
            total_trades=0,
            winning_trades=0,
            total_pnl=Decimal("0.00"),
            is_active=True,
            is_backtested=False,
        )

        db_session.add(strategy)
        db_session.commit()

        assert strategy.id is not None
        assert strategy.user_id == sample_user.id
        assert strategy.name == "CDC Action Zone"
        assert strategy.strategy_type == "cdc_action_zone"
        assert strategy.is_active is True

    def test_strategy_performance_metrics(
        self, db_session: Session, sample_user: User
    ) -> None:
        """Test strategy performance metrics"""
        strategy = Strategy(
            user_id=sample_user.id,
            name="Test Strategy",
            strategy_type="custom",
            total_trades=100,
            winning_trades=65,
            total_pnl=Decimal("1500.50"),
            max_drawdown=Decimal("0.15"),
            sharpe_ratio=Decimal("1.25"),
            is_backtested=True,
        )

        db_session.add(strategy)
        db_session.commit()

        assert strategy.total_trades == 100
        assert strategy.winning_trades == 65
        assert strategy.total_pnl == Decimal("1500.50")
        assert strategy.max_drawdown == Decimal("0.15")
        assert strategy.sharpe_ratio == Decimal("1.25")
        assert strategy.is_backtested is True

    def test_strategy_repr(self, db_session: Session, sample_user: User) -> None:
        """Test Strategy string representation"""
        strategy = Strategy(
            user_id=sample_user.id,
            name="Test Strategy",
            strategy_type="custom",
        )

        db_session.add(strategy)
        db_session.commit()

        expected = f"<Strategy(id={strategy.id}, name='Test Strategy', type='custom')>"
        assert repr(strategy) == expected


class TestPositionModel:
    """Test cases for Position model"""

    def test_create_long_position(
        self, db_session: Session, sample_portfolio: Portfolio, sample_asset: Asset
    ) -> None:
        """Test creating a long position"""
        position = Position(
            portfolio_id=sample_portfolio.id,
            asset_id=sample_asset.id,
            side="long",
            quantity=Decimal("1.5"),
            entry_price=Decimal("50000.00"),
            current_price=Decimal("51000.00"),
            stop_loss=Decimal("48000.00"),
            take_profit=Decimal("55000.00"),
            unrealized_pnl=Decimal("1500.00"),
            realized_pnl=Decimal("0.00"),
            status="open",
        )

        db_session.add(position)
        db_session.commit()

        assert position.id is not None
        assert position.portfolio_id == sample_portfolio.id
        assert position.asset_id == sample_asset.id
        assert position.side == "long"
        assert position.quantity == Decimal("1.5")
        assert position.status == "open"

    def test_create_short_position(
        self, db_session: Session, sample_portfolio: Portfolio, sample_asset: Asset
    ) -> None:
        """Test creating a short position"""
        position = Position(
            portfolio_id=sample_portfolio.id,
            asset_id=sample_asset.id,
            side="short",
            quantity=Decimal("2.0"),
            entry_price=Decimal("50000.00"),
            current_price=Decimal("49000.00"),
            stop_loss=Decimal("52000.00"),
            take_profit=Decimal("45000.00"),
            unrealized_pnl=Decimal("2000.00"),
            status="open",
        )

        db_session.add(position)
        db_session.commit()

        assert position.side == "short"
        assert position.unrealized_pnl == Decimal("2000.00")

    def test_position_relationships(
        self, db_session: Session, sample_portfolio: Portfolio, sample_asset: Asset
    ) -> None:
        """Test Position relationships"""
        position = Position(
            portfolio_id=sample_portfolio.id,
            asset_id=sample_asset.id,
            side="long",
            quantity=Decimal("1.0"),
            entry_price=Decimal("50000.00"),
        )

        db_session.add(position)
        db_session.commit()

        assert position.portfolio is not None
        assert position.portfolio.id == sample_portfolio.id
        assert position.asset is not None
        assert position.asset.id == sample_asset.id

    def test_position_repr(
        self, db_session: Session, sample_portfolio: Portfolio, sample_asset: Asset
    ) -> None:
        """Test Position string representation"""
        position = Position(
            portfolio_id=sample_portfolio.id,
            asset_id=sample_asset.id,
            side="long",
            quantity=Decimal("1.5"),
            entry_price=Decimal("50000.00"),
        )

        db_session.add(position)
        db_session.commit()

        # Test that repr contains expected components
        repr_str = repr(position)
        assert "Position" in repr_str
        assert f"asset_id={sample_asset.id}" in repr_str
        assert "side='long'" in repr_str
        assert "quantity=" in repr_str


class TestTradeModel:
    """Test cases for Trade model"""

    def test_create_buy_trade(
        self,
        db_session: Session,
        sample_portfolio: Portfolio,
        sample_asset: Asset,
        sample_strategy: Strategy,
    ) -> None:
        """Test creating a buy trade"""
        executed_at = datetime.now(timezone.utc)
        trade = Trade(
            portfolio_id=sample_portfolio.id,
            asset_id=sample_asset.id,
            strategy_id=sample_strategy.id,
            side="buy",
            quantity=Decimal("1.0"),
            price=Decimal("50000.00"),
            fee=Decimal("25.00"),
            order_type="market",
            trade_type="entry",
            external_order_id="12345",
            exchange="binance",
            executed_at=executed_at,
        )

        db_session.add(trade)
        db_session.commit()

        assert trade.id is not None
        assert trade.side == "buy"
        assert trade.quantity == Decimal("1.0")
        assert trade.price == Decimal("50000.00")
        assert trade.fee == Decimal("25.00")
        # Compare timestamps without microseconds and timezone to avoid precision issues
        assert trade.executed_at.replace(microsecond=0) == executed_at.replace(
            microsecond=0, tzinfo=None
        )

    def test_create_sell_trade_with_pnl(
        self, db_session: Session, sample_portfolio: Portfolio, sample_asset: Asset
    ) -> None:
        """Test creating a sell trade with P&L"""
        trade = Trade(
            portfolio_id=sample_portfolio.id,
            asset_id=sample_asset.id,
            side="sell",
            quantity=Decimal("1.0"),
            price=Decimal("51000.00"),
            fee=Decimal("25.50"),
            order_type="limit",
            trade_type="exit",
            pnl=Decimal("975.00"),  # 1000 profit - 25 fees
            executed_at=datetime.now(timezone.utc),
        )

        db_session.add(trade)
        db_session.commit()

        assert trade.side == "sell"
        assert trade.trade_type == "exit"
        assert trade.pnl == Decimal("975.00")

    def test_trade_relationships(
        self,
        db_session: Session,
        sample_portfolio: Portfolio,
        sample_asset: Asset,
        sample_strategy: Strategy,
    ) -> None:
        """Test Trade relationships"""
        trade = Trade(
            portfolio_id=sample_portfolio.id,
            asset_id=sample_asset.id,
            strategy_id=sample_strategy.id,
            side="buy",
            quantity=Decimal("1.0"),
            price=Decimal("50000.00"),
            order_type="market",
            trade_type="entry",
            executed_at=datetime.now(timezone.utc),
        )

        db_session.add(trade)
        db_session.commit()

        assert trade.portfolio is not None
        assert trade.portfolio.id == sample_portfolio.id
        assert trade.asset is not None
        assert trade.asset.id == sample_asset.id
        assert trade.strategy is not None
        assert trade.strategy.id == sample_strategy.id

    def test_trade_repr(
        self, db_session: Session, sample_portfolio: Portfolio, sample_asset: Asset
    ) -> None:
        """Test Trade string representation"""
        trade = Trade(
            portfolio_id=sample_portfolio.id,
            asset_id=sample_asset.id,
            side="buy",
            quantity=Decimal("1.5"),
            price=Decimal("50000.00"),
            order_type="market",
            trade_type="entry",
            executed_at=datetime.now(timezone.utc),
        )

        db_session.add(trade)
        db_session.commit()

        # Test that repr contains expected components
        repr_str = repr(trade)
        assert "Trade" in repr_str
        assert f"asset_id={sample_asset.id}" in repr_str
        assert "side='buy'" in repr_str
        assert "quantity=" in repr_str
        assert "price=" in repr_str


class TestSignalModel:
    """Test cases for Signal model"""

    def test_create_buy_signal(
        self, db_session: Session, sample_strategy: Strategy, sample_asset: Asset
    ) -> None:
        """Test creating a buy signal"""
        generated_at = datetime.now(timezone.utc)
        signal = Signal(
            strategy_id=sample_strategy.id,
            asset_id=sample_asset.id,
            signal_type="buy",
            strength=Decimal("0.85"),
            price=Decimal("50000.00"),
            suggested_stop_loss=Decimal("48000.00"),
            suggested_take_profit=Decimal("55000.00"),
            suggested_position_size=Decimal("0.02"),
            timeframe="1h",
            confidence=Decimal("0.75"),
            notes="Strong bullish momentum",
            market_conditions="trending",
            risk_level="medium",
            expected_duration="short",
            indicators_used='{"rsi": 65, "macd": "bullish", "volume": "high"}',
            status="active",
            generated_at=generated_at,
        )

        db_session.add(signal)
        db_session.commit()

        assert signal.id is not None
        assert signal.signal_type == "buy"
        assert signal.strength == Decimal("0.85")
        assert signal.confidence == Decimal("0.75")
        assert signal.market_conditions == "trending"
        assert signal.risk_level == "medium"

    def test_create_sell_signal(
        self, db_session: Session, sample_strategy: Strategy, sample_asset: Asset
    ) -> None:
        """Test creating a sell signal"""
        signal = Signal(
            strategy_id=sample_strategy.id,
            asset_id=sample_asset.id,
            signal_type="sell",
            strength=Decimal("0.90"),
            price=Decimal("51000.00"),
            timeframe="4h",
            confidence=Decimal("0.80"),
            market_conditions="ranging",
            risk_level="low",
            expected_duration="medium",
            status="active",
            generated_at=datetime.now(timezone.utc),
        )

        db_session.add(signal)
        db_session.commit()

        assert signal.signal_type == "sell"
        assert signal.strength == Decimal("0.90")
        assert signal.market_conditions == "ranging"

    def test_signal_with_expiration(
        self, db_session: Session, sample_strategy: Strategy, sample_asset: Asset
    ) -> None:
        """Test signal with expiration time"""
        generated_at = datetime.now(timezone.utc)
        expires_at = datetime.now(timezone.utc)

        signal = Signal(
            strategy_id=sample_strategy.id,
            asset_id=sample_asset.id,
            signal_type="buy",
            strength=Decimal("0.70"),
            price=Decimal("50000.00"),
            timeframe="1h",
            confidence=Decimal("0.65"),
            status="active",
            generated_at=generated_at,
            expires_at=expires_at,
        )

        db_session.add(signal)
        db_session.commit()

        # Compare timestamps without microseconds and timezone to avoid precision issues
        assert signal.expires_at is not None
        assert signal.expires_at.replace(microsecond=0) == expires_at.replace(
            microsecond=0, tzinfo=None
        )

    def test_signal_relationships(
        self, db_session: Session, sample_strategy: Strategy, sample_asset: Asset
    ) -> None:
        """Test Signal relationships"""
        signal = Signal(
            strategy_id=sample_strategy.id,
            asset_id=sample_asset.id,
            signal_type="buy",
            strength=Decimal("0.75"),
            price=Decimal("50000.00"),
            timeframe="1h",
            confidence=Decimal("0.70"),
            generated_at=datetime.now(timezone.utc),
        )

        db_session.add(signal)
        db_session.commit()

        assert signal.strategy is not None
        assert signal.strategy.id == sample_strategy.id
        assert signal.asset is not None
        assert signal.asset.id == sample_asset.id

    def test_signal_status_transitions(
        self, db_session: Session, sample_strategy: Strategy, sample_asset: Asset
    ) -> None:
        """Test signal status transitions"""
        signal = Signal(
            strategy_id=sample_strategy.id,
            asset_id=sample_asset.id,
            signal_type="buy",
            strength=Decimal("0.80"),
            price=Decimal("50000.00"),
            timeframe="1h",
            confidence=Decimal("0.75"),
            status="active",
            generated_at=datetime.now(timezone.utc),
        )

        db_session.add(signal)
        db_session.commit()

        # Test status change to executed
        signal.status = "executed"
        signal.executed_at = datetime.now(timezone.utc)
        db_session.commit()

        assert signal.status == "executed"
        assert signal.executed_at is not None

    def test_signal_repr(
        self, db_session: Session, sample_strategy: Strategy, sample_asset: Asset
    ) -> None:
        """Test Signal string representation"""
        signal = Signal(
            strategy_id=sample_strategy.id,
            asset_id=sample_asset.id,
            signal_type="buy",
            strength=Decimal("0.85"),
            price=Decimal("50000.00"),
            timeframe="1h",
            confidence=Decimal("0.75"),
            generated_at=datetime.now(timezone.utc),
        )

        db_session.add(signal)
        db_session.commit()

        expected = (
            f"<Signal(id={signal.id}, strategy_id={sample_strategy.id}, "
            f"signal_type='buy', confidence=0.7500)>"
        )
        assert repr(signal) == expected

    def test_signal_metadata_fields(
        self, db_session: Session, sample_strategy: Strategy, sample_asset: Asset
    ) -> None:
        """Test signal metadata fields"""
        signal = Signal(
            strategy_id=sample_strategy.id,
            asset_id=sample_asset.id,
            signal_type="buy",
            strength=Decimal("0.80"),
            price=Decimal("50000.00"),
            timeframe="1h",
            confidence=Decimal("0.75"),
            market_conditions="volatile",
            risk_level="high",
            expected_duration="long",
            indicators_used='{"rsi": 30, "bb_position": "lower", "volume": "low"}',
            generated_at=datetime.now(timezone.utc),
        )

        db_session.add(signal)
        db_session.commit()

        assert signal.market_conditions == "volatile"
        assert signal.risk_level == "high"
        assert signal.expected_duration == "long"
        assert signal.indicators_used is not None and "rsi" in signal.indicators_used
