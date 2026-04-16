"""Integration tests for questionnaire API session lifecycle endpoints."""

import pytest
from fastapi.testclient import TestClient

from backend.data_generation.enums import PropertyUse, TargetField
from backend.main import app

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
def test_session_start_and_answer_lifecycle(mocker) -> None:
    """Verify session start and answer endpoints work as a sequential flow."""
    client = TestClient(app)
    mocker.patch(
        "backend.api.routes.generate_conversational_question",
        new=mocker.AsyncMock(return_value="Could you share that detail for me?"),
    )

    start_response = client.post("/api/v1/sessions/start", json={})
    assert start_response.status_code == 200
    start_payload = start_response.json()
    assert start_payload["session_id"]
    assert "current_profile" in start_payload
    assert "is_complete" in start_payload
    assert "inferred_fields" in start_payload
    assert any(
        inferred_field["field_name"] == TargetField.ZIPCODE.value
        for inferred_field in start_payload["inferred_fields"]
    )

    if start_payload["is_complete"]:
        return

    next_question = start_payload["next_question"]
    assert next_question is not None
    target_field: str = next_question["target_field"]
    answer_value: str | bool = ANSWER_BY_FIELD[target_field]

    answer_response = client.post(
        f"/api/v1/sessions/{start_payload['session_id']}/answer",
        json={
            "target_field": target_field,
            "answer_value": answer_value,
        },
    )
    assert answer_response.status_code == 200
    answer_payload = answer_response.json()
    assert answer_payload["session_id"] == start_payload["session_id"]
    assert "current_profile" in answer_payload


@pytest.mark.integration
def test_answer_text_endpoint_updates_profile_from_mocked_extraction(mocker) -> None:
    """Verify text-answer endpoint applies extracted fields into session profile."""
    client = TestClient(app)
    mocker.patch(
        "backend.api.routes.generate_conversational_question",
        new=mocker.AsyncMock(return_value="Could you clarify this detail for me?"),
    )
    start_response = client.post("/api/v1/sessions/start", json={})
    assert start_response.status_code == 200
    session_id = start_response.json()["session_id"]

    mocker.patch(
        "backend.api.routes.extract_fields_from_text",
        new=mocker.AsyncMock(
            return_value={TargetField.PROPERTY_USE: PropertyUse.INVESTMENT}
        ),
    )
    answer_text_response = client.post(
        f"/api/v1/sessions/{session_id}/answer_text",
        json={"user_text": "This is an investment property."},
    )

    assert answer_text_response.status_code == 200
    payload = answer_text_response.json()
    assert payload["current_profile"]["property_use"] == PropertyUse.INVESTMENT.value
