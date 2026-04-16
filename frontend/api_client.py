"""Synchronous frontend client for questionnaire backend endpoints."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast
from uuid import UUID

import httpx
from loguru import logger

from backend.api.schemas import (
    AnswerQuestionRequest,
    AnswerTextRequest,
    QuestionnaireResponse,
    StartSessionRequest,
)
from backend.core_logic.field_mappings import FIELD_NAME_BY_TARGET, coerce_target_field_value
from backend.data_generation.enums import TargetField
from config import settings


class FrontendApiClientError(RuntimeError):
    """Represent frontend API request failures."""


class QuestionnaireApiClient:
    """Call backend questionnaire endpoints and parse typed responses."""

    def __init__(
        self,
        backend_url: str | None = None,
        request_func: (
            Callable[[str, str, dict[str, object] | None], dict[str, object]] | None
        ) = None,
    ) -> None:
        """Initialize the API client with base URL and optional request transport."""
        self.backend_url: str = (backend_url or settings.backend_url).rstrip("/")
        self._request_func: Callable[[str, str, dict[str, object] | None], dict[str, object]] = (
            request_func or self._request_with_httpx
        )

    def start_session(self, cohort_name: str | None = None) -> QuestionnaireResponse:
        """Start a new questionnaire session."""
        request_payload = StartSessionRequest(cohort_name=cohort_name).model_dump(mode="json")
        response_payload = self._request_func("POST", "/api/v1/sessions/start", request_payload)
        return self._parse_questionnaire_response(response_payload)

    def answer_question_explicitly(
        self,
        session_id: UUID,
        target_field: TargetField,
        answer_value: str | bool,
    ) -> QuestionnaireResponse:
        """Submit one explicit structured answer for a session."""
        request_payload = AnswerQuestionRequest(
            target_field=target_field,
            answer_value=answer_value,
        ).model_dump(mode="json")
        response_payload = self._request_func(
            "POST",
            f"/api/v1/sessions/{session_id}/answer",
            request_payload,
        )
        return self._parse_questionnaire_response(response_payload)

    def answer_question_via_text(self, session_id: UUID, user_text: str) -> QuestionnaireResponse:
        """Submit one free-text answer for LLM-based extraction."""
        request_payload = AnswerTextRequest(user_text=user_text).model_dump(mode="json")
        response_payload = self._request_func(
            "POST",
            f"/api/v1/sessions/{session_id}/answer_text",
            request_payload,
        )
        return self._parse_questionnaire_response(response_payload)

    def _request_with_httpx(
        self,
        method: str,
        path: str,
        payload: dict[str, object] | None,
    ) -> dict[str, object]:
        """Perform an HTTP request and return decoded JSON response."""
        url: str = f"{self.backend_url}{path}"
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.request(method=method, url=url, json=payload)
                response.raise_for_status()
                raw_payload: object = response.json()
        except httpx.HTTPError as error:
            logger.error("Backend API request failed", method=method, url=url, error=str(error))
            msg = f"Backend request failed: {error}"
            raise FrontendApiClientError(msg) from error

        if not isinstance(raw_payload, dict):
            msg = "Backend returned a non-object JSON response."
            raise FrontendApiClientError(msg)

        logger.debug("Received backend API response", method=method, url=url)
        return cast(dict[str, object], raw_payload)

    def _parse_questionnaire_response(
        self,
        response_payload: dict[str, object],
    ) -> QuestionnaireResponse:
        """Coerce backend JSON payload into strict QuestionnaireResponse model."""
        payload: dict[str, object] = dict(response_payload)
        session_id_raw: object = payload.get("session_id")
        if not isinstance(session_id_raw, str):
            msg = "Missing or invalid session_id in backend response."
            raise FrontendApiClientError(msg)
        payload["session_id"] = UUID(session_id_raw)

        next_question_raw: object = payload.get("next_question")
        if isinstance(next_question_raw, dict):
            target_field_raw: object = next_question_raw.get("target_field")
            if isinstance(target_field_raw, str):
                next_question_raw = dict(next_question_raw)
                next_question_raw["target_field"] = TargetField(target_field_raw)
                payload["next_question"] = next_question_raw

        current_profile_raw: object = payload.get("current_profile")
        if isinstance(current_profile_raw, dict):
            profile_payload: dict[str, object] = dict(current_profile_raw)
            for target_field, field_name in FIELD_NAME_BY_TARGET.items():
                raw_value: object = profile_payload.get(field_name)
                if raw_value is None:
                    continue
                if isinstance(raw_value, (str, bool)):
                    profile_payload[field_name] = coerce_target_field_value(target_field, raw_value)
            payload["current_profile"] = profile_payload

        return QuestionnaireResponse.model_validate(payload)
