"""Integration tests for answer_text validation script execution flow."""

from argparse import Namespace
from uuid import uuid4

import pytest

from scripts import validate_answer_text_examples as validator


@pytest.mark.integration
def test_main_logs_expected_and_outcome_fields_for_failures(mocker) -> None:
    """Verify main emits sentence and outcome context on a failing example."""
    mocker.patch(
        "scripts.validate_answer_text_examples.parse_args",
        return_value=Namespace(
            base_url="http://127.0.0.1:8000",
            cohort_name="Middle-aged Suburban Families",
            timeout_seconds=1.0,
        ),
    )
    test_case = validator.ExampleCase(
        sentence="I currently have a mortgage.",
        expected_fields={"currently_have_mortgage": True},
    )
    mocker.patch(
        "scripts.validate_answer_text_examples.EXAMPLE_CASES",
        [test_case],
    )
    mocker.patch(
        "scripts.validate_answer_text_examples.run_case",
        return_value=validator.ValidationResult(
            sentence=test_case.sentence,
            passed=False,
            mismatches=["currently_have_mortgage: expected=True, actual=None"],
            expected_fields=test_case.expected_fields,
            actual_fields={"currently_have_mortgage": None, "loan_primary_purpose": "purchase"},
        ),
    )
    info_spy = mocker.patch("scripts.validate_answer_text_examples.logger.info")
    error_spy = mocker.patch("scripts.validate_answer_text_examples.logger.error")

    with pytest.raises(SystemExit):
        validator.main()

    info_calls = info_spy.call_args_list
    assert any(
        call.args[0] == "Starting answer_text example validation"
        for call in info_calls
    )
    assert any(
        call.args[0] == "Finished answer_text validation"
        for call in info_calls
    )
    error_call = error_spy.call_args_list[0]
    assert error_call.args[0].startswith("FAIL [")
    assert error_call.kwargs["sentence"] == test_case.sentence
    assert error_call.kwargs["expected_fields"] == {"currently_have_mortgage": True}
    assert error_call.kwargs["outcome_fields"] == {"currently_have_mortgage": None}
    assert error_call.kwargs["populated_fields"] == {"loan_primary_purpose": "purchase"}


@pytest.mark.integration
def test_run_case_returns_expected_and_actual_profile_fields() -> None:
    """Verify run_case compares expected values against API profile payload."""
    session_id = str(uuid4())

    def handler(request):
        if request.url.path == "/api/v1/sessions/start":
            return validator.httpx.Response(200, json={"session_id": session_id})
        if request.url.path == f"/api/v1/sessions/{session_id}/answer_text":
            return validator.httpx.Response(
                200,
                json={
                    "current_profile": {
                        "annual_income_band": "50k_to_100k",
                        "property_type": "condo",
                    }
                },
            )
        return validator.httpx.Response(404, json={})

    client = validator.httpx.Client(
        transport=validator.httpx.MockTransport(handler),
        base_url="http://127.0.0.1:8000",
    )
    case = validator.ExampleCase(
        sentence="My household income is around $78,000 per year.",
        expected_fields={"annual_income_band": "50k_to_100k"},
    )

    result = validator.run_case(
        client=client,
        base_url="http://127.0.0.1:8000",
        cohort_name="Middle-aged Suburban Families",
        case=case,
    )

    assert result.passed is True
    assert result.mismatches == []
    assert result.expected_fields == {"annual_income_band": "50k_to_100k"}
    assert result.actual_fields["annual_income_band"] == "50k_to_100k"
