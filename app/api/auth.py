"""
Authentication API endpoints with security features
"""

from __future__ import annotations

from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from loguru import logger
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db_session
from app.core.middleware import login_failure_tracker
from app.core.security import SecurityError, security_manager
from app.models.user import LoginAttempt, User
from app.schemas.auth import LoginRequest, LoginResponse, UserResponse

router = APIRouter()
security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db_session),
) -> User:
    """Get current authenticated user"""
    try:
        payload = security_manager.verify_token(credentials.credentials)
        user_id = payload.get("user_id")

        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload",
            )

        # Get user from database
        result = await db.execute(
            select(User).where(User.id == user_id, User.is_active)
        )
        user = result.scalar_one_or_none()

        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or inactive",
            )

        return user

    except SecurityError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
        ) from e


@router.post("/login", response_model=LoginResponse)
async def login(
    request: Request,
    login_data: LoginRequest,
    db: AsyncSession = Depends(get_db_session),
) -> LoginResponse:
    """Authenticate user and return JWT token"""
    client_ip = _get_client_ip(request)
    hashed_ip = security_manager.hash_ip_address(client_ip)
    user_agent = request.headers.get("user-agent", "unknown")

    # Check if IP is rate limited
    if login_failure_tracker.is_locked(hashed_ip):
        remaining_time = login_failure_tracker.get_lockout_time_remaining(hashed_ip)
        logger.warning(
            "Login attempt from locked IP",
            extra={
                "ip_hash": hashed_ip,
                "username": login_data.username,
                "remaining_lockout": remaining_time,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Too many failed attempts. Try again in {remaining_time} seconds.",
        )

    # Get user from database
    result = await db.execute(
        select(User).where(User.username == login_data.username, User.is_active)
    )
    user = result.scalar_one_or_none()

    # Check if user exists and password is correct
    if not user or not security_manager.verify_password(
        login_data.password, user.hashed_password
    ):
        # Record failed attempt
        await _record_login_attempt(
            db=db,
            ip_hash=hashed_ip,
            username=login_data.username,
            user_id=user.id if user else None,
            success=False,
            failure_reason="invalid_credentials",
            user_agent=user_agent,
        )

        # Track failure for rate limiting
        login_failure_tracker.record_failure(hashed_ip)

        # Update user failure count if user exists
        if user:
            await _update_user_failure_count(db, user.id)

        logger.warning(
            "Failed login attempt",
            extra={
                "ip_hash": hashed_ip,
                "username": login_data.username,
                "reason": "invalid_credentials",
            },
        )

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    # Check if user account is locked
    if user.locked_until and user.locked_until > datetime.utcnow():
        remaining_time = int((user.locked_until - datetime.utcnow()).total_seconds())
        logger.warning(
            "Login attempt for locked user account",
            extra={
                "ip_hash": hashed_ip,
                "username": login_data.username,
                "user_id": user.id,
                "remaining_lockout": remaining_time,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_423_LOCKED,
            detail=f"Account is locked. Try again in {remaining_time} seconds.",
        )

    # Successful login
    # Reset failure counts
    login_failure_tracker.reset_attempts(hashed_ip)
    await _reset_user_failure_count(db, user.id)

    # Create JWT token
    token_data = {"user_id": user.id, "username": user.username}
    access_token = security_manager.create_access_token(token_data)

    # Update last login time
    await db.execute(
        update(User).where(User.id == user.id).values(last_login=datetime.utcnow())
    )

    # Record successful attempt
    await _record_login_attempt(
        db=db,
        ip_hash=hashed_ip,
        username=login_data.username,
        user_id=user.id,
        success=True,
        user_agent=user_agent,
    )

    await db.commit()

    logger.info(
        "Successful login",
        extra={
            "ip_hash": hashed_ip,
            "username": login_data.username,
            "user_id": user.id,
        },
    )

    return LoginResponse(
        access_token=access_token,
        expires_in=security_manager.expire_minutes * 60,
        user=UserResponse.model_validate(user),
    )


@router.post("/logout")
async def logout(
    request: Request,
    current_user: User = Depends(get_current_user),
) -> dict[str, str]:
    """Logout user (client-side token removal)"""
    client_ip = _get_client_ip(request)
    hashed_ip = security_manager.hash_ip_address(client_ip)

    logger.info(
        "User logout",
        extra={
            "ip_hash": hashed_ip,
            "username": current_user.username,
            "user_id": current_user.id,
        },
    )

    return {"message": "Successfully logged out"}


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_user),
) -> UserResponse:
    """Get current user information"""
    return UserResponse.model_validate(current_user)


def _get_client_ip(request: Request) -> str:
    """Extract client IP address from request"""
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()

    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        return real_ip

    return request.client.host if request.client else "unknown"


async def _record_login_attempt(
    db: AsyncSession,
    ip_hash: str,
    username: str,
    user_id: int | None,
    success: bool,
    failure_reason: str | None = None,
    user_agent: str | None = None,
) -> None:
    """Record login attempt in database"""
    login_attempt = LoginAttempt(
        ip_address_hash=ip_hash,
        username=username,
        user_id=user_id,
        success=success,
        failure_reason=failure_reason,
        user_agent=user_agent,
    )
    db.add(login_attempt)


async def _update_user_failure_count(db: AsyncSession, user_id: int) -> None:
    """Update user failure count and lock if necessary"""
    # Increment failure count
    await db.execute(
        update(User)
        .where(User.id == user_id)
        .values(
            failed_login_attempts=User.failed_login_attempts + 1,
            last_failed_login=datetime.utcnow(),
        )
    )

    # Check if user should be locked (5 failures = 15 minute lock)
    result = await db.execute(
        select(User.failed_login_attempts).where(User.id == user_id)
    )
    failure_count = result.scalar() or 0

    if failure_count >= 5:
        lock_until = datetime.utcnow() + timedelta(minutes=15)
        await db.execute(
            update(User).where(User.id == user_id).values(locked_until=lock_until)
        )


async def _reset_user_failure_count(db: AsyncSession, user_id: int) -> None:
    """Reset user failure count after successful login"""
    await db.execute(
        update(User)
        .where(User.id == user_id)
        .values(failed_login_attempts=0, locked_until=None, last_failed_login=None)
    )
