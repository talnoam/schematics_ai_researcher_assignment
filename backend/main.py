"""Backend application entrypoint for container startup."""

from fastapi import FastAPI
from loguru import logger

from config import settings


def create_app() -> FastAPI:
    """Create the FastAPI application instance."""
    app = FastAPI(title=settings.app_name)

    @app.get("/health")
    def health() -> dict[str, str]:
        """Return a health response for readiness checks."""
        return {"status": "ok"}

    logger.info("Backend app initialized", app_name=settings.app_name)
    return app


app = create_app()
