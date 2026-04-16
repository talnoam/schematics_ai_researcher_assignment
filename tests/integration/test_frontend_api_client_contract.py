"""Integration tests for frontend API client against backend routes."""

from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from frontend.api_client import QuestionnaireApiClient

ANSWER_BY_FIELD: dict[str, str | bool] = {
    "credit_score_rate": "good",
    "loan_primary_purpose": "purchase",
    "property_type": "single_family",
    "property_use": "primary_residence",
    "annual_income_band": "100k_to_200k",
    "property_value_band": "600k_to_1m",
    "credit_line_band": "50k_to_100k",
    "age_band": "30_to_44",
    "currently_have_mortgage": True,
    "military_veteran": False,
}


@pytest.mark.integration
def test_frontend_api_client_session_flow() -> None:
    """Verify frontend client can execute start and answer flow against backend app."""
    test_client = TestClient(app)

    def request_func(method: str, path: str, payload: dict[str, object] | None) -> dict[str, object]:
        """Route frontend client calls into in-process FastAPI test client."""
        response = test_client.request(method, path, json=payload)
        response.raise_for_status()
        response_payload: object = response.json()
        assert isinstance(response_payload, dict)
        return response_payload

    frontend_client = QuestionnaireApiClient(request_func=request_func)
    start_response = frontend_client.start_session()
    assert isinstance(start_response.session_id, UUID)

    if start_response.is_complete:
        return

    assert start_response.next_question is not None
    answer_value: str | bool = ANSWER_BY_FIELD[start_response.next_question.target_field.value]
    answer_response = frontend_client.answer_question_explicitly(
        session_id=start_response.session_id,
        target_field=start_response.next_question.target_field,
        answer_value=answer_value,
    )
    assert answer_response.session_id == start_response.session_id
