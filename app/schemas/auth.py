"""
Authentication and user-related Pydantic schemas
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, EmailStr, Field, field_validator

from app.utils.validation import InputSanitizer, SecureBaseModel


class UserBase(SecureBaseModel):
    """Base user schema with common fields"""

    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    full_name: str | None = Field(None, max_length=255)
    preferred_language: str = Field("en", max_length=5)
    theme: str = Field("light", max_length=10)
    timezone: str = Field("UTC", max_length=50)

    @field_validator("username")
    @classmethod
    def validate_username(cls, v: str) -> str:
        if not InputSanitizer.validate_username(v):
            raise ValueError("Username contains invalid characters")
        return v

    @field_validator("email")
    @classmethod
    def validate_email_format(cls, v: str) -> str:
        if not InputSanitizer.validate_email(v):
            raise ValueError("Invalid email format")
        return v


class UserCreate(UserBase):
    """Schema for user creation"""

    password: str = Field(..., min_length=8, max_length=128)

    @field_validator("password")
    @classmethod
    def validate_password_strength(cls, v: str) -> str:
        validation_result = InputSanitizer.validate_password_strength(v)
        if not validation_result["valid"]:
            raise ValueError("; ".join(validation_result["errors"]))
        return v


class UserUpdate(BaseModel):
    """Schema for user updates"""

    full_name: str | None = Field(None, max_length=255)
    preferred_language: str | None = Field(None, max_length=5)
    theme: str | None = Field(None, max_length=10)
    timezone: str | None = Field(None, max_length=50)
    notification_settings: dict | None = None


class UserInDB(UserBase):
    """Schema for user data stored in database"""

    id: int
    hashed_password: str
    is_active: bool
    is_verified: bool
    notification_settings: dict | None = None
    created_at: datetime
    updated_at: datetime
    last_login: datetime | None = None

    model_config = {"from_attributes": True}


class UserResponse(UserBase):
    """Schema for user data in API responses"""

    id: int
    is_active: bool
    is_verified: bool
    created_at: datetime
    updated_at: datetime
    last_login: datetime | None = None

    model_config = {"from_attributes": True}


class Token(BaseModel):
    """JWT token response schema"""

    access_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenData(BaseModel):
    """Token payload data schema"""

    username: str | None = None
    user_id: int | None = None


class LoginRequest(SecureBaseModel):
    """Login request schema with validation"""

    email: EmailStr
    password: str = Field(..., min_length=1, max_length=128)


class LoginResponse(BaseModel):
    """Login response schema"""

    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse


class SecurityEvent(BaseModel):
    """Security event logging schema"""

    event_type: str
    ip_address_hash: str
    username: str | None = None
    user_id: int | None = None
    success: bool
    failure_reason: str | None = None
    user_agent: str | None = None
    timestamp: datetime

    model_config = {"from_attributes": True}
