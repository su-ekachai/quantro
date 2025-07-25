"""
Health check endpoints for Quantro Trading Platform
"""

import os
import sys
from datetime import datetime, timezone

from fastapi import APIRouter
from pydantic import BaseModel

from app.core.database import db_manager

router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response model"""

    status: str
    timestamp: datetime
    version: str
    python_version: str
    environment: str


class DetailedHealthResponse(HealthResponse):
    """Detailed health check response model"""

    database: dict[str, str | bool]
    external_apis: dict[str, str]


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Basic health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc),
        version="0.1.0",
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        environment=os.getenv("ENVIRONMENT", "development"),
    )


@router.get("/health/detailed", response_model=DetailedHealthResponse)
async def detailed_health_check() -> DetailedHealthResponse:
    """Detailed health check endpoint with external dependencies"""
    # Check database health
    db_health = await db_manager.health_check()

    # Determine overall status
    overall_status = (
        "healthy" if db_health.get("database_connected", False) else "unhealthy"
    )

    return DetailedHealthResponse(
        status=overall_status,
        timestamp=datetime.now(timezone.utc),
        version="0.1.0",
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        environment=os.getenv("ENVIRONMENT", "development"),
        database=db_health,
        external_apis={"ccxt": "not_implemented", "set_api": "not_implemented"},
    )


@router.get("/health/database")
async def database_health_check() -> dict[str, str | bool]:
    """Database-specific health check endpoint"""
    return await db_manager.health_check()


@router.get("/ping")
async def ping() -> dict[str, str]:
    """Simple ping endpoint"""
    return {"message": "pong"}
