"""
Tests for health check endpoints
"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint() -> None:
    """Test basic health check endpoint"""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert "version" in data


def test_ping_endpoint() -> None:
    """Test ping endpoint"""
    response = client.get("/api/v1/ping")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "pong"


def test_root_endpoint() -> None:
    """Test API root endpoint"""
    response = client.get("/api")
    assert response.status_code == 200
    data = response.json()
    assert "Quantro Trading Platform API" in data["message"]
    assert data["version"] == "0.1.0"
