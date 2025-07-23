"""
Health check endpoints for Quantro Trading Platform
"""

import os
import sys
from datetime import datetime

from fastapi import APIRouter
from pydantic import BaseModel

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

    database_status: str
    external_apis: dict[str, str]


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Basic health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        version="0.1.0",
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        environment=os.getenv("ENVIRONMENT", "development"),
    )


@router.get("/health/detailed", response_model=DetailedHealthResponse)
async def detailed_health_check() -> DetailedHealthResponse:
    """Detailed health check endpoint with external dependencies"""
    # TODO: Add actual database and external API health checks
    return DetailedHealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        version="0.1.0",
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        environment=os.getenv("ENVIRONMENT", "development"),
        database_status="not_implemented",
        external_apis={"ccxt": "not_implemented", "set_api": "not_implemented"},
    )


@router.get("/ping")
async def ping() -> dict[str, str]:
    """Simple ping endpoint"""
    return {"message": "pong"}
