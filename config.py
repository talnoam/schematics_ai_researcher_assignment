"""Global application configuration backed by environment variables."""

from pydantic_settings import BaseSettings


class AppSettings(BaseSettings):
    """Top-level settings shared across all modules.

    Attributes:
        app_name: Display name of the application.
        debug: Enable debug mode for verbose logging.
        ollama_base_url: Base URL for the Ollama LLM API on the host machine.
        ollama_model: Model identifier used for LLM inference.
        backend_url: URL used by frontend when calling backend APIs.
        data_dir: Root directory for generated and raw data files.
        random_seed: Global random seed for reproducibility.
    """

    app_name: str = "Adaptive Questionnaire Agent"
    debug: bool = False
    ollama_base_url: str = "http://host.docker.internal:11434"
    ollama_model: str = "qwen:9b"
    backend_url: str = "http://localhost:8000"
    data_dir: str = "data"
    random_seed: int = 42

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = AppSettings()
