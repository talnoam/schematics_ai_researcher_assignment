"""Unit tests for frontend app utility behavior."""

from unittest.mock import MagicMock
from uuid import UUID

from _pytest.monkeypatch import MonkeyPatch

from backend.api.schemas import QuestionnaireResponse
from backend.core_logic.deterministic_rules import PartialUserProfile
from backend.data_generation.enums import PropertyUse, TargetField
from frontend.app import (
    coerce_explicit_option,
    get_explicit_options,
    main,
    split_profile_fields,
)


def test_get_explicit_options_returns_boolean_choices() -> None:
    """Verify boolean fields expose true and false options."""
    options = get_explicit_options(TargetField.CURRENTLY_HAVE_MORTGAGE)
    assert options == ["true", "false"]


def test_coerce_explicit_option_casts_enum_value() -> None:
    """Verify explicit enum selection is coerced to enum member."""
    coerced_value = coerce_explicit_option(TargetField.PROPERTY_USE, "investment")
    assert coerced_value == PropertyUse.INVESTMENT


def test_split_profile_fields_separates_explicit_and_inferred_values() -> None:
    """Verify debug profile split groups explicit and inferred values correctly."""
    response = QuestionnaireResponse(
        session_id=UUID("ff6feec0-f16f-4d62-ae74-c79443ec2d99"),
        is_complete=False,
        next_question=None,
        current_profile=PartialUserProfile(
            property_use=PropertyUse.PRIMARY_RESIDENCE,
            currently_have_mortgage=True,
        ),
    )
    explicit_values, inferred_values = split_profile_fields(
        response,
        explicit_fields={TargetField.PROPERTY_USE},
    )

    assert explicit_values["property_use"] == PropertyUse.PRIMARY_RESIDENCE.value
    assert inferred_values["currently_have_mortgage"] is True


def test_main_calls_render_app(monkeypatch: MonkeyPatch) -> None:
    """Verify main delegates page rendering."""
    from frontend import app as frontend_app

    render_app_mock = MagicMock()
    monkeypatch.setattr(frontend_app, "render_app", render_app_mock)
    main()
    render_app_mock.assert_called_once()
