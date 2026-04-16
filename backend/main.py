"""Backend application entrypoint for container startup."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from backend.api.config import CORS_ALLOW_HEADERS, CORS_ALLOW_METHODS, CORS_ALLOW_ORIGINS
from backend.api.routes import router as api_router
from config import settings


def create_app() -> FastAPI:
    """Create the FastAPI application instance."""
    app = FastAPI(title=settings.app_name)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ALLOW_ORIGINS,
        allow_methods=CORS_ALLOW_METHODS,
        allow_headers=CORS_ALLOW_HEADERS,
    )
    app.include_router(api_router)

    @app.get("/health")
    def health() -> dict[str, str]:
        """Return a health response for readiness checks."""
        return {"status": "ok"}

    logger.info("Backend app initialized", app_name=settings.app_name)
    return app


app = create_app()
