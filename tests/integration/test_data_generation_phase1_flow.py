"""Integration tests for phase-1 data-generation components."""

import pytest

from backend.data_generation.cohort_loader import load_cohort_definitions
from backend.data_generation.enums import AnnualIncomeBand
from backend.data_generation.latent_factors import LatentFactorModel, LatentFactors


@pytest.mark.integration
def test_cohort_loading_and_latent_adjustment_flow() -> None:
    """Verify real cohort definitions can be loaded and shifted end-to-end."""
    cohorts = load_cohort_definitions()
    target_cohort = next(cohort for cohort in cohorts if cohort.cohort_name == "Tech Veterans")
    model = LatentFactorModel(random_seed=1234)

    adjusted = model.adjust_probabilities(
        base_probabilities=target_cohort.base_probabilities,
        latent_factors=LatentFactors(affluence_score=0.9, risk_profile=0.2),
    )

    assert adjusted.annual_income_band[AnnualIncomeBand.ABOVE_200K] > (
        target_cohort.base_probabilities.annual_income_band[AnnualIncomeBand.ABOVE_200K]
    )
    assert sum(adjusted.credit_score_rate.values()) == pytest.approx(1.0)
    assert sum(adjusted.currently_have_mortgage.values()) == pytest.approx(1.0)
