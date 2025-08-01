"""Fix foreign key relationships

Revision ID: 13b999375bcc
Revises: 2502a3f3b67c
Create Date: 2025-07-28 13:01:22.532269

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "13b999375bcc"
down_revision: str | Sequence[str] | None = "2502a3f3b67c"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_foreign_key(None, "login_attempts", "users", ["user_id"], ["id"])
    op.create_foreign_key(None, "market_data", "symbols", ["symbol_id"], ["id"])
    # ### end Alembic commands ###


def downgrade() -> None:
    """Downgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_constraint(None, "market_data", type_="foreignkey")
    op.drop_constraint(None, "login_attempts", type_="foreignkey")
    # ### end Alembic commands ###
