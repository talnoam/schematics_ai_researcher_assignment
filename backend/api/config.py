"""Configuration values for API routing and session behavior."""

API_V1_PREFIX: str = "/api/v1"
SESSIONS_PREFIX: str = f"{API_V1_PREFIX}/sessions"

CORS_ALLOW_ORIGINS: list[str] = ["*"]
CORS_ALLOW_METHODS: list[str] = ["*"]
CORS_ALLOW_HEADERS: list[str] = ["*"]
