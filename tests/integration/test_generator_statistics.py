"""Integration tests for end-to-end synthetic data generation statistics."""

import pytest

from backend.data_generation.enums import AnnualIncomeBand, PropertyValueBand
from backend.data_generation.generator import MockDataGenerator


@pytest.mark.integration
def test_generate_dataframe_produces_expected_cohort_mix_and_income_uplift() -> None:
    """Verify generated users preserve cohort mix and latent-factor income uplift."""
    generator = MockDataGenerator(random_seed=22)
    generated_dataframe = generator.generate_dataframe(user_count=1500)

    cohort_frequencies = generated_dataframe["cohort_name"].value_counts(normalize=True)
    assert cohort_frequencies["Tech Veterans"] == pytest.approx(0.4, abs=0.08)
    assert cohort_frequencies["Middle-aged Suburban Families"] == pytest.approx(0.35, abs=0.08)
    assert cohort_frequencies["Young Urban Renters"] == pytest.approx(0.25, abs=0.08)

    baseline_high_income = (
        generated_dataframe["annual_income_band"] == AnnualIncomeBand.ABOVE_200K.value
    ).mean()
    premium_property_mask = (
        generated_dataframe["property_value_band"] == PropertyValueBand.ABOVE_1M.value
    )
    conditional_high_income = (
        generated_dataframe.loc[premium_property_mask, "annual_income_band"]
        == AnnualIncomeBand.ABOVE_200K.value
    ).mean()
    assert conditional_high_income > baseline_high_income
