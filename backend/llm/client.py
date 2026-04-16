"""Async client for OpenAI-compatible Ollama chat completions."""

from __future__ import annotations

from typing import Any

import httpx
from loguru import logger

from backend.llm.config import LLM_MAX_TOKENS, LLM_REQUEST_TIMEOUT_SECONDS, LLM_TEMPERATURE
from config import settings


class OllamaClient:
    """Send asynchronous chat-completion requests to Ollama."""

    def __init__(self, base_url: str | None = None, model: str | None = None) -> None:
        """Initialize the Ollama client with endpoint and model defaults."""
        self.base_url: str = (base_url or settings.ollama_base_url).rstrip("/")
        self.model: str = model or settings.ollama_model

    async def generate_json_completion(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a JSON-only completion from the configured Ollama model."""
        endpoint_url: str = f"{self.base_url}/v1/chat/completions"
        # Any is used for dynamic JSON payloads returned by Ollama's API contract.
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": LLM_TEMPERATURE,
            "max_tokens": LLM_MAX_TOKENS,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        async with httpx.AsyncClient(timeout=LLM_REQUEST_TIMEOUT_SECONDS) as client:
            response = await client.post(endpoint_url, json=payload)
            response.raise_for_status()
            # Any is used for dynamic JSON payloads returned by Ollama's API contract.
            response_payload: dict[str, Any] = response.json()

        try:
            completion_text: str = str(response_payload["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as error:
            msg = "Unexpected Ollama response payload structure."
            raise ValueError(msg) from error

        logger.debug(
            "Received Ollama extraction response",
            model=self.model,
            endpoint_url=endpoint_url,
            response_length=len(completion_text),
        )
        return completion_text
