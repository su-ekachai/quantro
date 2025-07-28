"""
Tests for frontend authentication and theme system
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_login_page_loads():
    """Test that login page loads correctly"""
    response = client.get("/login")
    assert response.status_code == 200
    assert "Quantro" in response.text
    assert "Sign in to your trading account" in response.text


def test_dashboard_page_loads():
    """Test that dashboard page loads correctly"""
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert "Dashboard" in response.text


def test_static_files_served():
    """Test that static files are served correctly"""
    # Test CSS file
    response = client.get("/static/css/main.css")
    assert response.status_code == 200
    assert "text/css" in response.headers.get("content-type", "")
    
    # Test JS file
    response = client.get("/static/js/stores.js")
    assert response.status_code == 200
    assert "javascript" in response.headers.get("content-type", "")
    
    # Test translation files
    response = client.get("/static/translations/en.json")
    assert response.status_code == 200
    assert "application/json" in response.headers.get("content-type", "")
    
    response = client.get("/static/translations/th.json")
    assert response.status_code == 200
    assert "application/json" in response.headers.get("content-type", "")


def test_root_redirects_to_dashboard():
    """Test that root path serves dashboard"""
    response = client.get("/")
    assert response.status_code == 200
    assert "Dashboard" in response.text


def test_login_page_contains_alpine_js():
    """Test that login page includes Alpine.js"""
    response = client.get("/login")
    assert response.status_code == 200
    assert "alpinejs" in response.text
    assert "x-data" in response.text


def test_dashboard_contains_auth_guard():
    """Test that dashboard contains authentication guard"""
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert "authGuard" in response.text
    assert "$store.auth" in response.text


def test_theme_system_css_variables():
    """Test that CSS contains theme system variables"""
    response = client.get("/static/css/main.css")
    assert response.status_code == 200
    content = response.text
    
    # Check for CSS custom properties
    assert "--bg-primary" in content
    assert "--text-primary" in content
    assert ".dark" in content
    
    # Check for theme transition
    assert "transition:" in content


def test_translation_files_structure():
    """Test that translation files have correct structure"""
    # Test English translations
    response = client.get("/static/translations/en.json")
    assert response.status_code == 200
    en_data = response.json()
    
    required_keys = ["dashboard", "login", "logout", "email", "password"]
    for key in required_keys:
        assert key in en_data
    
    # Test Thai translations
    response = client.get("/static/translations/th.json")
    assert response.status_code == 200
    th_data = response.json()
    
    for key in required_keys:
        assert key in th_data
        # Ensure Thai translations are different from English
        assert th_data[key] != en_data[key]


def test_responsive_design_classes():
    """Test that CSS contains responsive design classes"""
    response = client.get("/static/css/main.css")
    assert response.status_code == 200
    content = response.text
    
    # Check for responsive grid
    assert "trading-grid" in content
    assert "@media (min-width:" in content
    
    # Check for mobile/desktop utilities
    assert "mobile-only" in content
    assert "desktop-only" in content


def test_form_validation_styles():
    """Test that CSS contains form validation styles"""
    response = client.get("/static/css/main.css")
    assert response.status_code == 200
    content = response.text
    
    # Check for form styles
    assert "form-input" in content
    assert "form-error" in content
    assert "form-label" in content
    
    # Check for button styles
    assert "btn-primary" in content
    assert "btn-secondary" in content


def test_alpine_stores_structure():
    """Test that Alpine.js stores are properly structured"""
    response = client.get("/static/js/stores.js")
    assert response.status_code == 200
    content = response.text
    
    # Check for required stores
    assert "Alpine.store('auth'" in content
    assert "Alpine.store('theme'" in content
    assert "Alpine.store('i18n'" in content
    assert "Alpine.store('toast'" in content
    
    # Check for authentication methods
    assert "login(" in content
    assert "logout(" in content
    assert "apiCall(" in content
    
    # Check for theme methods
    assert "toggle(" in content
    assert "apply(" in content


def test_alpine_components_structure():
    """Test that Alpine.js components are properly structured"""
    response = client.get("/static/js/components.js")
    assert response.status_code == 200
    content = response.text
    
    # Check for required components
    assert "Alpine.data('loginForm'" in content
    assert "Alpine.data('navigation'" in content
    assert "Alpine.data('toastManager'" in content
    
    # Check for form validation
    assert "formValidator" in content
    assert "validate(" in content