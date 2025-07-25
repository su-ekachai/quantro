"""
User authentication and profile models
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, DateTime, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.trading import Portfolio, Strategy


class User(Base):
    """User model for authentication and profile management"""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    full_name: Mapped[str | None] = mapped_column(String(255))

    # User preferences
    preferred_language: Mapped[str] = mapped_column(String(5), default="en")
    theme: Mapped[str] = mapped_column(String(10), default="light")
    timezone: Mapped[str] = mapped_column(String(50), default="UTC")

    # Account status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False)

    # Notification preferences
    notification_settings: Mapped[str | None] = mapped_column(Text)  # JSON string

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    last_login: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Relationships
    portfolios: Mapped[list[Portfolio]] = relationship(
        "Portfolio", back_populates="user", cascade="all, delete-orphan"
    )
    strategies: Mapped[list[Strategy]] = relationship(
        "Strategy", back_populates="user", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<User(id={self.id}, username='{self.username}')>"
