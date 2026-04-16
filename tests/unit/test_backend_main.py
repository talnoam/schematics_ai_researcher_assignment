"""Unit tests for backend application bootstrap."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.main import app, create_app


def test_create_app_returns_fastapi_instance() -> None:
    """Verify backend app factory creates a FastAPI app."""
    backend_app = create_app()
    assert isinstance(backend_app, FastAPI)


def test_health_endpoint_returns_ok_status() -> None:
    """Verify health endpoint responds with ok status."""
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
