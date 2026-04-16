"""Configuration values for API routing and session behavior."""

API_V1_PREFIX: str = "/api/v1"
SESSIONS_PREFIX: str = f"{API_V1_PREFIX}/sessions"

CORS_ALLOW_ORIGINS: list[str] = ["*"]
CORS_ALLOW_METHODS: list[str] = ["*"]
CORS_ALLOW_HEADERS: list[str] = ["*"]

COHORT_ZIPCODE_PRIORS: dict[str, dict[str, float]] = {
    "Tech Veterans": {
        "94027": 0.9,
        "90210": 0.1,
    },
    "Middle-aged Suburban Families": {
        "75024": 0.9,
        "30327": 0.1,
    },
    "Young Urban Renters": {
        "19134": 0.9,
        "48201": 0.1,
    },
}
