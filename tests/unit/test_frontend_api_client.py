"""Unit tests for frontend questionnaire API client."""

from uuid import UUID

from backend.data_generation.enums import TargetField
from frontend.api_client import QuestionnaireApiClient


def _build_response_payload() -> dict[str, object]:
    """Build a minimal valid questionnaire response payload."""
    return {
        "session_id": "c21483fb-a12f-489d-9725-8930cbcc6bb7",
        "is_complete": False,
        "next_question": {
            "target_field": "property_use",
            "question_text": "How will the property be used?",
            "friction_cost": 0.16,
        },
        "current_profile": {
            "credit_score_rate": None,
            "loan_primary_purpose": None,
            "property_type": None,
            "property_use": None,
            "annual_income_band": None,
            "property_value_band": None,
            "credit_line_band": None,
            "age_band": None,
            "currently_have_mortgage": None,
            "military_veteran": None,
        },
    }


def test_start_session_uses_expected_path_and_payload() -> None:
    """Verify start-session call uses expected endpoint and returns parsed model."""
    captured_call: dict[str, object] = {}

    def request_func(method: str, path: str, payload: dict[str, object] | None) -> dict[str, object]:
        """Capture request details and return fake backend payload."""
        captured_call["method"] = method
        captured_call["path"] = path
        captured_call["payload"] = payload
        return _build_response_payload()

    client = QuestionnaireApiClient(request_func=request_func)
    response = client.start_session(cohort_name="Tech Veterans")

    assert captured_call["method"] == "POST"
    assert captured_call["path"] == "/api/v1/sessions/start"
    assert captured_call["payload"] == {"cohort_name": "Tech Veterans"}
    assert response.session_id == UUID("c21483fb-a12f-489d-9725-8930cbcc6bb7")


def test_answer_question_explicitly_serializes_target_field() -> None:
    """Verify explicit-answer call serializes target field as enum value string."""
    captured_call: dict[str, object] = {}

    def request_func(method: str, path: str, payload: dict[str, object] | None) -> dict[str, object]:
        """Capture request details and return fake backend payload."""
        captured_call["method"] = method
        captured_call["path"] = path
        captured_call["payload"] = payload
        return _build_response_payload()

    client = QuestionnaireApiClient(request_func=request_func)
    session_id = UUID("c21483fb-a12f-489d-9725-8930cbcc6bb7")
    response = client.answer_question_explicitly(
        session_id=session_id,
        target_field=TargetField.PROPERTY_USE,
        answer_value="investment",
    )

    assert captured_call["method"] == "POST"
    assert captured_call["path"] == f"/api/v1/sessions/{session_id}/answer"
    assert captured_call["payload"] == {
        "target_field": TargetField.PROPERTY_USE.value,
        "answer_value": "investment",
    }
    assert response.next_question is not None
