"""Prepare tables for TimescaleDB hypertables

Revision ID: ae3707d3de9a
Revises: b8769e3f7839
Create Date: 2025-07-25 02:22:01.766011

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "ae3707d3de9a"
down_revision: str | Sequence[str] | None = "b8769e3f7839"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # Drop existing primary key constraints and recreate with timestamp included

    # For market_data table - include timestamp in primary key
    op.drop_constraint("market_data_pkey", "market_data", type_="primary")
    op.create_primary_key("market_data_pkey", "market_data", ["id", "timestamp"])

    # For trading_signals table - include generated_at in primary key
    op.drop_constraint("trading_signals_pkey", "trading_signals", type_="primary")
    op.create_primary_key(
        "trading_signals_pkey", "trading_signals", ["id", "generated_at"]
    )


def downgrade() -> None:
    """Downgrade schema."""
    # Revert primary key constraints back to original

    # For market_data table
    op.drop_constraint("market_data_pkey", "market_data", type_="primary")
    op.create_primary_key("market_data_pkey", "market_data", ["id"])

    # For trading_signals table
    op.drop_constraint("trading_signals_pkey", "trading_signals", type_="primary")
    op.create_primary_key("trading_signals_pkey", "trading_signals", ["id"])
