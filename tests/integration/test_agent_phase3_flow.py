"""Integration tests for phase-3 adaptive agent decision flow."""

import pytest

from backend.core_logic.agent import AdaptiveQuestionnaireAgent
from backend.core_logic.deterministic_rules import PartialUserProfile
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


@pytest.mark.integration
def test_agent_infers_deterministic_field_and_selects_next_question() -> None:
    """Verify agent runs deterministic rules and then chooses a utility-max field."""
    agent = AdaptiveQuestionnaireAgent()
    profile = PartialUserProfile(
        credit_score_rate=CreditScoreRate.GOOD,
        loan_primary_purpose=LoanPrimaryPurpose.REFINANCE,
        property_type=PropertyType.SINGLE_FAMILY,
        property_use=None,
        annual_income_band=AnnualIncomeBand.FROM_100K_TO_200K,
        property_value_band=PropertyValueBand.FROM_600K_TO_1M,
        credit_line_band=CreditLineBand.FROM_50K_TO_100K,
        age_band=AgeBand.FROM_45_TO_59,
        currently_have_mortgage=None,
        military_veteran=False,
    )
    marginal_probabilities: dict[TargetField, dict[str, float]] = {
        TargetField.PROPERTY_USE: {"primary_residence": 0.6, "investment": 0.4},
    }

    decision = agent.get_next_action(profile, marginal_probabilities=marginal_probabilities)

    assert decision.action_type == "ask_question"
    assert decision.selected_field == TargetField.PROPERTY_USE
    assert decision.utility_score > 0.0
