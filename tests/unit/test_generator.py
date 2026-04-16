"""Unit tests for synthetic mock-data generation."""

import pytest

from backend.data_generation.config import ZIPCODE_PREMIUM_POOL
from backend.data_generation.enums import PropertyValueBand
from backend.data_generation.generator import MockDataGenerator
from backend.data_generation.schemas import GeneratedUserRecord


def test_generate_user_records_returns_expected_count() -> None:
    """Verify generator returns the requested number of typed records."""
    generator = MockDataGenerator(random_seed=123)

    generated_users: list[GeneratedUserRecord] = generator.generate_user_records(user_count=25)

    assert len(generated_users) == 25
    assert isinstance(generated_users[0], GeneratedUserRecord)


def test_generate_dataframe_includes_profile_and_metadata_columns() -> None:
    """Verify generated dataframe includes expected profile and metadata columns."""
    generator = MockDataGenerator(random_seed=44)

    generated_dataframe = generator.generate_dataframe(user_count=10)

    expected_columns: set[str] = {
        "cohort_name",
        "zipcode",
        "affluence_score",
        "risk_profile",
        "credit_score_rate",
        "loan_primary_purpose",
        "property_type",
        "property_use",
        "annual_income_band",
        "property_value_band",
        "credit_line_band",
        "age_band",
        "currently_have_mortgage",
        "military_veteran",
    }
    assert expected_columns.issubset(set(generated_dataframe.columns))


def test_sample_zipcode_returns_premium_pool_for_high_affluence() -> None:
    """Verify high-affluence profiles receive zipcode values from premium pool."""
    generator = MockDataGenerator(random_seed=7)

    zipcode = generator._sample_zipcode(
        property_value_band=PropertyValueBand.UNDER_300K,
        affluence_score=0.95,
    )

    assert zipcode in ZIPCODE_PREMIUM_POOL


def test_validate_cohort_weights_raises_for_unknown_cohort() -> None:
    """Verify unknown cohort names in weight config raise a validation error."""
    with pytest.raises(ValueError):
        MockDataGenerator(cohort_weights={"Unknown Cohort": 1.0}, random_seed=8)
