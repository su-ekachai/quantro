"""
Quantro Trading Platform - Main FastAPI Application
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api import auth, health
from app.core.config import settings
from app.core.database import close_database, init_database
from app.core.middleware import SecurityMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Application lifespan manager"""
    # Startup
    await init_database()
    yield
    # Shutdown
    await close_database()


# Create FastAPI application
app = FastAPI(
    title="Quantro Trading Platform",
    description="Lightweight trading platform with backtesting capabilities",
    version="0.1.0",
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    lifespan=lifespan,
)

# Add security middleware
app.add_middleware(SecurityMiddleware)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_hosts_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="frontend/templates")

# Include routers
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(auth.router, prefix="/api/v1/auth", tags=["authentication"])


# Frontend routes
@app.get("/", response_class=HTMLResponse)
async def root(request: Request) -> HTMLResponse:
    """Root endpoint - redirect to dashboard or login"""
    return templates.TemplateResponse(request, "dashboard.html")


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request) -> HTMLResponse:
    """Login page"""
    return templates.TemplateResponse(request, "login.html")


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request) -> HTMLResponse:
    """Dashboard page"""
    return templates.TemplateResponse(request, "dashboard.html")


# API root endpoint
@app.get("/api")
async def api_root() -> dict[str, str]:
    """API root endpoint"""
    return {"message": "Quantro Trading Platform API", "version": "0.1.0"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
    )
