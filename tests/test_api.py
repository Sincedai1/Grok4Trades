"""Test API endpoints."""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert data["message"] == "Grok4Trades API"

def test_health():
    """Test health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data

def test_metrics():
    """Test metrics endpoint."""
    response = client.get("/metrics")
    assert response.status_code == 200
    # Check that prometheus metrics are returned
    assert "g4t_requests_total" in response.text
    assert "g4t_errors_total" in response.text

def test_stream_events():
    """Test SSE streaming endpoint."""
    # Just test that endpoint exists and returns correct headers
    with client as c:
        response = c.get("/stream/events", stream=True)
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
