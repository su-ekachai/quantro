"""
Tests for market data models
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models.market import Asset, MarketData


class TestAssetModel:
    """Test cases for Asset model"""

    def test_create_crypto_asset(self, db_session: Session) -> None:
        """Test creating a cryptocurrency asset"""
        asset = Asset(
            symbol="BTC/USDT",
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

        assert asset.id is not None
        assert asset.symbol == "BTC/USDT"
        assert asset.asset_class == "crypto"
        assert asset.exchange == "binance"
        assert asset.is_active is True

    def test_create_thai_stock_asset(self, db_session: Session) -> None:
        """Test creating a Thai stock asset"""
        asset = Asset(
            symbol="PTT",
            name="PTT Public Company Limited",
            asset_class="stock",
            exchange="SET",
            base_currency="THB",
            quote_currency="THB",
            min_order_size=Decimal("1"),
            max_order_size=Decimal("1000000"),
            price_precision=2,
            quantity_precision=0,
            is_active=True,
        )

        db_session.add(asset)
        db_session.commit()

        assert asset.id is not None
        assert asset.symbol == "PTT"
        assert asset.asset_class == "stock"
        assert asset.exchange == "SET"

    def test_create_commodity_asset(self, db_session: Session) -> None:
        """Test creating a commodity asset"""
        asset = Asset(
            symbol="GOLD",
            name="Gold Futures",
            asset_class="commodity",
            exchange="COMEX",
            base_currency="USD",
            quote_currency="USD",
            min_order_size=Decimal("0.1"),
            max_order_size=Decimal("100"),
            price_precision=2,
            quantity_precision=1,
            is_active=True,
        )

        db_session.add(asset)
        db_session.commit()

        assert asset.id is not None
        assert asset.symbol == "GOLD"
        assert asset.asset_class == "commodity"
        assert asset.exchange == "COMEX"

    def test_asset_symbol_unique_constraint(self, db_session: Session) -> None:
        """Test that asset symbols must be unique"""
        import uuid

        unique_symbol = f"TEST-{str(uuid.uuid4())[:8]}"

        asset1 = Asset(
            symbol=unique_symbol,
            name="Bitcoin",
            asset_class="crypto",
            exchange="binance",
        )

        asset2 = Asset(
            symbol=unique_symbol,  # Same symbol
            name="Bitcoin",
            asset_class="crypto",
            exchange="coinbase",  # Different exchange
        )

        db_session.add(asset1)
        db_session.commit()

        db_session.add(asset2)
        with pytest.raises(IntegrityError):
            db_session.commit()

        # Rollback the failed transaction
        db_session.rollback()

    def test_asset_repr(self, db_session: Session) -> None:
        """Test Asset string representation"""
        asset = Asset(
            symbol="ETH/USDT",
            name="Ethereum",
            asset_class="crypto",
            exchange="binance",
        )

        db_session.add(asset)
        db_session.commit()

        expected = f"<Asset(id={asset.id}, symbol='ETH/USDT', asset_class='crypto')>"
        assert repr(asset) == expected

    def test_asset_relationships(self, db_session: Session) -> None:
        """Test Asset relationships with MarketData"""
        import uuid

        unique_symbol = f"ETH/USDT-{str(uuid.uuid4())[:8]}"

        asset = Asset(
            symbol=unique_symbol,  # Use unique symbol to avoid conflict
            name="Ethereum",
            asset_class="crypto",
            exchange="binance",
        )

        db_session.add(asset)
        db_session.commit()

        # Initially no market data
        assert len(asset.market_data) == 0

        # Add market data
        market_data = MarketData(
            asset_id=asset.id,
            timestamp=datetime.now(timezone.utc),
            timeframe="1h",
            open_price=Decimal("50000"),
            high_price=Decimal("51000"),
            low_price=Decimal("49000"),
            close_price=Decimal("50500"),
            volume=Decimal("100.5"),
            source="binance",
        )

        db_session.add(market_data)
        db_session.commit()

        # Refresh asset to load relationships
        db_session.refresh(asset)
        assert len(asset.market_data) == 1
        assert asset.market_data[0].asset_id == asset.id


class TestMarketDataModel:
    """Test cases for MarketData model"""

    def test_create_market_data(self, db_session: Session, sample_asset: Asset) -> None:
        """Test creating market data"""
        timestamp = datetime.now(timezone.utc)
        market_data = MarketData(
            asset_id=sample_asset.id,
            timestamp=timestamp,
            timeframe="1h",
            open_price=Decimal("50000.00"),
            high_price=Decimal("51000.00"),
            low_price=Decimal("49000.00"),
            close_price=Decimal("50500.00"),
            volume=Decimal("100.50"),
            trades_count=1500,
            vwap=Decimal("50250.00"),
            source="binance",
        )

        db_session.add(market_data)
        db_session.commit()

        assert market_data.id is not None
        assert market_data.asset_id == sample_asset.id
        # SQLite stores datetime without timezone, so compare naive datetimes
        expected_timestamp = timestamp.replace(tzinfo=None, microsecond=0)
        actual_timestamp = market_data.timestamp.replace(microsecond=0)
        assert actual_timestamp == expected_timestamp
        assert market_data.timeframe == "1h"
        assert market_data.open_price == Decimal("50000.00")
        assert market_data.volume == Decimal("100.50")
        assert market_data.source == "binance"

    def test_market_data_unique_constraint(
        self, db_session: Session, sample_asset: Asset
    ) -> None:
        """Test unique constraint on asset_id, timestamp, timeframe"""
        timestamp = datetime.now(timezone.utc)

        market_data1 = MarketData(
            asset_id=sample_asset.id,
            timestamp=timestamp,
            timeframe="1h",
            open_price=Decimal("50000"),
            high_price=Decimal("51000"),
            low_price=Decimal("49000"),
            close_price=Decimal("50500"),
            volume=Decimal("100"),
            source="binance",
        )

        market_data2 = MarketData(
            asset_id=sample_asset.id,
            timestamp=timestamp,  # Same timestamp
            timeframe="1h",  # Same timeframe
            open_price=Decimal("50100"),
            high_price=Decimal("51100"),
            low_price=Decimal("49100"),
            close_price=Decimal("50600"),
            volume=Decimal("200"),
            source="coinbase",
        )

        db_session.add(market_data1)
        db_session.commit()

        db_session.add(market_data2)
        with pytest.raises(IntegrityError):
            db_session.commit()

    def test_market_data_different_timeframes(
        self, db_session: Session, sample_asset: Asset
    ) -> None:
        """Test that same timestamp with different timeframes is allowed"""
        timestamp = datetime.now(timezone.utc)

        market_data_1h = MarketData(
            asset_id=sample_asset.id,
            timestamp=timestamp,
            timeframe="1h",
            open_price=Decimal("50000"),
            high_price=Decimal("51000"),
            low_price=Decimal("49000"),
            close_price=Decimal("50500"),
            volume=Decimal("100"),
            source="binance",
        )

        market_data_1d = MarketData(
            asset_id=sample_asset.id,
            timestamp=timestamp,  # Same timestamp
            timeframe="1d",  # Different timeframe
            open_price=Decimal("49000"),
            high_price=Decimal("52000"),
            low_price=Decimal("48000"),
            close_price=Decimal("50500"),
            volume=Decimal("2400"),
            source="binance",
        )

        db_session.add_all([market_data_1h, market_data_1d])
        db_session.commit()

        assert market_data_1h.id is not None
        assert market_data_1d.id is not None

    def test_market_data_relationship_with_asset(
        self, db_session: Session, sample_asset: Asset
    ) -> None:
        """Test MarketData relationship with Asset"""
        market_data = MarketData(
            asset_id=sample_asset.id,
            timestamp=datetime.now(timezone.utc),
            timeframe="1h",
            open_price=Decimal("50000"),
            high_price=Decimal("51000"),
            low_price=Decimal("49000"),
            close_price=Decimal("50500"),
            volume=Decimal("100"),
            source="binance",
        )

        db_session.add(market_data)
        db_session.commit()

        # Test relationship
        assert market_data.asset is not None
        assert market_data.asset.id == sample_asset.id
        assert market_data.asset.symbol == sample_asset.symbol

    def test_market_data_repr(self, db_session: Session, sample_asset: Asset) -> None:
        """Test MarketData string representation"""
        timestamp = datetime.now(timezone.utc)
        market_data = MarketData(
            asset_id=sample_asset.id,
            timestamp=timestamp,
            timeframe="1h",
            open_price=Decimal("50000"),
            high_price=Decimal("51000"),
            low_price=Decimal("49000"),
            close_price=Decimal("50500"),
            volume=Decimal("100"),
            source="binance",
        )

        db_session.add(market_data)
        db_session.commit()

        # Test that repr contains expected components
        repr_str = repr(market_data)
        assert "MarketData" in repr_str
        assert f"asset_id={sample_asset.id}" in repr_str
        assert "timeframe='1h'" in repr_str

    def test_market_data_ohlcv_validation(
        self, db_session: Session, sample_asset: Asset
    ) -> None:
        """Test OHLCV data logical validation"""
        # This would typically be handled by application logic, not database constraints
        # But we can test that the data is stored correctly
        market_data = MarketData(
            asset_id=sample_asset.id,
            timestamp=datetime.now(timezone.utc),
            timeframe="1h",
            open_price=Decimal("50000.00"),
            high_price=Decimal("51000.00"),  # High should be >= open, close
            low_price=Decimal("49000.00"),  # Low should be <= open, close
            close_price=Decimal("50500.00"),
            volume=Decimal("100.50"),
            source="binance",
        )

        db_session.add(market_data)
        db_session.commit()

        # Verify OHLC relationships
        assert market_data.high_price >= market_data.open_price
        assert market_data.high_price >= market_data.close_price
        assert market_data.low_price <= market_data.open_price
        assert market_data.low_price <= market_data.close_price

    def test_market_data_precision(
        self, db_session: Session, sample_asset: Asset
    ) -> None:
        """Test decimal precision handling"""
        market_data = MarketData(
            asset_id=sample_asset.id,
            timestamp=datetime.now(timezone.utc),
            timeframe="1h",
            open_price=Decimal("50000.12345678"),  # 8 decimal places
            high_price=Decimal("51000.87654321"),
            low_price=Decimal("49000.11111111"),
            close_price=Decimal("50500.99999999"),
            volume=Decimal("100.12345678"),
            vwap=Decimal("50250.55555555"),
            source="binance",
        )

        db_session.add(market_data)
        db_session.commit()

        # Verify precision is maintained
        assert market_data.open_price == Decimal("50000.12345678")
        assert market_data.volume == Decimal("100.12345678")
        assert market_data.vwap == Decimal("50250.55555555")
