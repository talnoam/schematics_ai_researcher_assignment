"""Unit tests for answer_text validation script logging helpers."""

from scripts.validate_answer_text_examples import (
    ValidationResult,
    build_outcome_fields,
    build_populated_fields,
)


def test_build_outcome_fields_includes_only_expected_field_names() -> None:
    """Verify focused outcome payload is limited to expected fields."""
    result = ValidationResult(
        sentence="sample",
        passed=True,
        expected_fields={"annual_income_band": "50k_to_100k", "property_type": "condo"},
        actual_fields={
            "annual_income_band": "50k_to_100k",
            "property_type": "condo",
            "military_veteran": False,
        },
    )

    outcome_fields = build_outcome_fields(result)

    assert outcome_fields == {
        "annual_income_band": "50k_to_100k",
        "property_type": "condo",
    }


def test_build_populated_fields_drops_null_values() -> None:
    """Verify compact payload includes only populated profile fields."""
    populated_fields = build_populated_fields(
        {
            "annual_income_band": "100k_to_200k",
            "property_type": None,
            "currently_have_mortgage": True,
        }
    )

    assert populated_fields == {
        "annual_income_band": "100k_to_200k",
        "currently_have_mortgage": True,
    }
