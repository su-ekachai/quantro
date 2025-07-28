"""
Test configuration for Quantro Trading Platform
"""

import os
import pytest


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment variables"""
    test_env = {
        "SECRET_KEY": "test-secret-key-for-testing-only",
        "DATABASE_URL": "postgresql+psycopg://test:test@localhost:5432/quantro_test",
        "JWT_SECRET_KEY": "test-jwt-secret-key-for-testing-only",
        "JWT_ALGORITHM": "HS256",
        "JWT_EXPIRE_MINUTES": "30",
        "DEBUG": "true",
        "ENVIRONMENT": "testing",
        "ALLOWED_HOSTS": "*",
        "LOG_LEVEL": "DEBUG",
        "CCXT_SANDBOX": "true",
    }
    
    # Set environment variables directly
    for key, value in test_env.items():
        os.environ[key] = value
    
    yield
    
    # Clean up environment variables after tests
    for key in test_env.keys():
        os.environ.pop(key, None)