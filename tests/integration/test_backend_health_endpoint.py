"""Integration tests for backend service bootstrap."""

import pytest
from fastapi.testclient import TestClient

from backend.main import app


@pytest.mark.integration
def test_backend_health_endpoint_is_available() -> None:
    """Verify backend health endpoint is available for compose checks."""
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
