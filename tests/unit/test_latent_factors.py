"""Unit tests for latent-factor probability adjustments."""

import pytest

from backend.data_generation.enums import (
    AgeBand,
    AnnualIncomeBand,
    CreditLineBand,
    CreditScoreRate,
    LoanPrimaryPurpose,
    PropertyType,
    PropertyUse,
    PropertyValueBand,
)
from backend.data_generation.latent_factors import LatentFactorModel, LatentFactors
from backend.data_generation.schemas import CohortBaseProbabilities


def _build_base_probabilities_for_latent_tests() -> CohortBaseProbabilities:
    """Build a stable base distribution to test latent-factor shifts."""
    return CohortBaseProbabilities(
        credit_score_rate={
            CreditScoreRate.POOR: 0.15,
            CreditScoreRate.FAIR: 0.25,
            CreditScoreRate.GOOD: 0.30,
            CreditScoreRate.VERY_GOOD: 0.20,
            CreditScoreRate.EXCELLENT: 0.10,
        },
        loan_primary_purpose={
            LoanPrimaryPurpose.PURCHASE: 0.3,
            LoanPrimaryPurpose.REFINANCE: 0.3,
            LoanPrimaryPurpose.HOME_IMPROVEMENT: 0.1,
            LoanPrimaryPurpose.DEBT_CONSOLIDATION: 0.2,
            LoanPrimaryPurpose.INVESTMENT: 0.1,
        },
        property_type={
            PropertyType.SINGLE_FAMILY: 0.5,
            PropertyType.CONDO: 0.2,
            PropertyType.TOWNHOUSE: 0.15,
            PropertyType.MULTI_FAMILY: 0.10,
            PropertyType.MOBILE_HOME: 0.05,
        },
        property_use={
            PropertyUse.PRIMARY_RESIDENCE: 0.7,
            PropertyUse.SECOND_HOME: 0.1,
            PropertyUse.INVESTMENT: 0.2,
        },
        annual_income_band={
            AnnualIncomeBand.UNDER_50K: 0.25,
            AnnualIncomeBand.FROM_50K_TO_100K: 0.35,
            AnnualIncomeBand.FROM_100K_TO_200K: 0.25,
            AnnualIncomeBand.ABOVE_200K: 0.15,
        },
        property_value_band={
            PropertyValueBand.UNDER_300K: 0.35,
            PropertyValueBand.FROM_300K_TO_600K: 0.35,
            PropertyValueBand.FROM_600K_TO_1M: 0.20,
            PropertyValueBand.ABOVE_1M: 0.10,
        },
        credit_line_band={
            CreditLineBand.UNDER_10K: 0.2,
            CreditLineBand.FROM_10K_TO_50K: 0.45,
            CreditLineBand.FROM_50K_TO_100K: 0.25,
            CreditLineBand.ABOVE_100K: 0.10,
        },
        age_band={
            AgeBand.UNDER_30: 0.0,
            AgeBand.FROM_30_TO_44: 0.0,
            AgeBand.FROM_45_TO_59: 0.0,
            AgeBand.FROM_60_PLUS: 1.0,
        },
        currently_have_mortgage={True: 0.6, False: 0.4},
        military_veteran={True: 0.2, False: 0.8},
    )


def test_sample_latent_factors_returns_values_in_unit_interval() -> None:
    """Verify sampled latent scores are clipped to the closed unit interval."""
    model = LatentFactorModel(random_seed=123)
    sampled_factors = model.sample_latent_factors()

    assert 0.0 <= sampled_factors.affluence_score <= 1.0
    assert 0.0 <= sampled_factors.risk_profile <= 1.0


def test_adjust_probabilities_shifts_income_and_mortgage_for_affluent_users() -> None:
    """Verify high affluence shifts income upward and lowers mortgage for older cohorts."""
    base_probabilities = _build_base_probabilities_for_latent_tests()
    model = LatentFactorModel(random_seed=7)
    factors = LatentFactors(affluence_score=0.95, risk_profile=0.1)

    adjusted = model.adjust_probabilities(base_probabilities=base_probabilities, latent_factors=factors)

    assert (
        adjusted.annual_income_band[AnnualIncomeBand.ABOVE_200K]
        > base_probabilities.annual_income_band[AnnualIncomeBand.ABOVE_200K]
    )
    assert (
        adjusted.credit_line_band[CreditLineBand.ABOVE_100K]
        > base_probabilities.credit_line_band[CreditLineBand.ABOVE_100K]
    )
    assert (
        adjusted.currently_have_mortgage[True]
        < base_probabilities.currently_have_mortgage[True]
    )


def test_adjust_probabilities_keeps_shifted_distributions_normalized() -> None:
    """Verify shifted probability distributions remain normalized."""
    base_probabilities = _build_base_probabilities_for_latent_tests()
    model = LatentFactorModel(random_seed=77)
    factors = LatentFactors(affluence_score=0.4, risk_profile=0.8)

    adjusted = model.adjust_probabilities(base_probabilities=base_probabilities, latent_factors=factors)

    assert sum(adjusted.annual_income_band.values()) == pytest.approx(1.0)
    assert sum(adjusted.property_value_band.values()) == pytest.approx(1.0)
    assert sum(adjusted.credit_line_band.values()) == pytest.approx(1.0)
    assert sum(adjusted.credit_score_rate.values()) == pytest.approx(1.0)
    assert sum(adjusted.currently_have_mortgage.values()) == pytest.approx(1.0)
