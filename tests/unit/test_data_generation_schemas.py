"""Unit tests for data-generation schemas and validation behavior."""

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
    TargetField,
)
from backend.data_generation.schemas import CohortBaseProbabilities, ObservedAnswer, UserProfile


def _build_valid_base_probabilities() -> CohortBaseProbabilities:
    """Build a valid baseline probability object for schema tests."""
    return CohortBaseProbabilities(
        credit_score_rate={
            CreditScoreRate.POOR: 0.1,
            CreditScoreRate.FAIR: 0.2,
            CreditScoreRate.GOOD: 0.3,
            CreditScoreRate.VERY_GOOD: 0.2,
            CreditScoreRate.EXCELLENT: 0.2,
        },
        loan_primary_purpose={
            LoanPrimaryPurpose.PURCHASE: 0.4,
            LoanPrimaryPurpose.REFINANCE: 0.2,
            LoanPrimaryPurpose.HOME_IMPROVEMENT: 0.2,
            LoanPrimaryPurpose.DEBT_CONSOLIDATION: 0.1,
            LoanPrimaryPurpose.INVESTMENT: 0.1,
        },
        property_type={
            PropertyType.SINGLE_FAMILY: 0.4,
            PropertyType.CONDO: 0.2,
            PropertyType.TOWNHOUSE: 0.2,
            PropertyType.MULTI_FAMILY: 0.1,
            PropertyType.MOBILE_HOME: 0.1,
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
            PropertyValueBand.UNDER_300K: 0.3,
            PropertyValueBand.FROM_300K_TO_600K: 0.4,
            PropertyValueBand.FROM_600K_TO_1M: 0.2,
            PropertyValueBand.ABOVE_1M: 0.1,
        },
        credit_line_band={
            CreditLineBand.UNDER_10K: 0.2,
            CreditLineBand.FROM_10K_TO_50K: 0.4,
            CreditLineBand.FROM_50K_TO_100K: 0.3,
            CreditLineBand.ABOVE_100K: 0.1,
        },
        age_band={
            AgeBand.UNDER_30: 0.2,
            AgeBand.FROM_30_TO_44: 0.3,
            AgeBand.FROM_45_TO_59: 0.3,
            AgeBand.FROM_60_PLUS: 0.2,
        },
        currently_have_mortgage={True: 0.5, False: 0.5},
        military_veteran={True: 0.2, False: 0.8},
    )


def test_user_profile_accepts_strict_enum_values() -> None:
    """Verify UserProfile supports the ten target fields with strict enums."""
    user_profile = UserProfile(
        credit_score_rate=CreditScoreRate.GOOD,
        loan_primary_purpose=LoanPrimaryPurpose.PURCHASE,
        property_type=PropertyType.SINGLE_FAMILY,
        property_use=PropertyUse.PRIMARY_RESIDENCE,
        annual_income_band=AnnualIncomeBand.FROM_100K_TO_200K,
        property_value_band=PropertyValueBand.FROM_300K_TO_600K,
        credit_line_band=CreditLineBand.FROM_50K_TO_100K,
        age_band=AgeBand.FROM_45_TO_59,
        currently_have_mortgage=True,
        military_veteran=False,
    )

    assert user_profile.credit_score_rate == CreditScoreRate.GOOD


def test_observed_answer_rejects_unknown_field_name() -> None:
    """Verify ObservedAnswer validation rejects unknown target field names."""
    with pytest.raises(ValueError):
        ObservedAnswer(field_name="unknown_field", value=True)


def test_cohort_base_probabilities_requires_normalized_distributions() -> None:
    """Verify distribution sums are validated for each target field."""
    valid_probabilities = _build_valid_base_probabilities()
    invalid_credit_score_distribution = {
        CreditScoreRate.POOR: 0.1,
        CreditScoreRate.FAIR: 0.2,
        CreditScoreRate.GOOD: 0.2,
        CreditScoreRate.VERY_GOOD: 0.2,
        CreditScoreRate.EXCELLENT: 0.2,
    }
    invalid_payload = valid_probabilities.model_dump()
    invalid_payload["credit_score_rate"] = invalid_credit_score_distribution

    with pytest.raises(ValueError):
        CohortBaseProbabilities.model_validate(invalid_payload)


def test_observed_answer_accepts_typed_values() -> None:
    """Verify ObservedAnswer accepts target field enum with typed payload."""
    observed_answer = ObservedAnswer(
        field_name=TargetField.CURRENTLY_HAVE_MORTGAGE,
        value=True,
    )

    assert observed_answer.field_name == TargetField.CURRENTLY_HAVE_MORTGAGE
