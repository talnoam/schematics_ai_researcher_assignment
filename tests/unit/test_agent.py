"""Unit tests for adaptive questionnaire agent decision logic."""

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


def _build_profile_with_known_values() -> PartialUserProfile:
    """Build a profile with all target fields known."""
    return PartialUserProfile(
        credit_score_rate=CreditScoreRate.GOOD,
        loan_primary_purpose=LoanPrimaryPurpose.PURCHASE,
        property_type=PropertyType.SINGLE_FAMILY,
        property_use=PropertyUse.PRIMARY_RESIDENCE,
        annual_income_band=AnnualIncomeBand.FROM_100K_TO_200K,
        property_value_band=PropertyValueBand.FROM_600K_TO_1M,
        credit_line_band=CreditLineBand.FROM_50K_TO_100K,
        age_band=AgeBand.FROM_45_TO_59,
        currently_have_mortgage=True,
        military_veteran=False,
    )


def test_get_next_action_stops_when_profile_is_fully_known() -> None:
    """Verify agent stops once all fields are known after deterministic pass."""
    agent = AdaptiveQuestionnaireAgent()
    profile = _build_profile_with_known_values()

    decision = agent.get_next_action(profile, marginal_probabilities={})

    assert decision.action_type == "stop_and_infer"
    assert decision.selected_field is None


def test_get_next_action_prefers_low_friction_when_eig_is_close() -> None:
    """Verify low-friction field is preferred when EIG values are relatively close."""
    agent = AdaptiveQuestionnaireAgent()
    profile = _build_profile_with_known_values().model_copy(
        update={
            "credit_score_rate": None,
            "property_use": None,
        }
    )
    marginal_probabilities: dict[TargetField, dict[str, float]] = {
        TargetField.CREDIT_SCORE_RATE: {"good": 0.5, "fair": 0.5},
        TargetField.PROPERTY_USE: {"primary_residence": 0.55, "investment": 0.45},
    }

    decision = agent.get_next_action(profile, marginal_probabilities=marginal_probabilities)

    assert decision.action_type == "ask_question"
    assert decision.selected_field == TargetField.PROPERTY_USE


def test_get_next_action_stops_when_max_utility_below_threshold() -> None:
    """Verify stop decision is returned when utility is below threshold."""
    agent = AdaptiveQuestionnaireAgent()
    profile = _build_profile_with_known_values().model_copy(update={"credit_score_rate": None})
    marginal_probabilities: dict[TargetField, dict[str, float]] = {
        TargetField.CREDIT_SCORE_RATE: {"good": 1.0, "fair": 0.0},
    }

    decision = agent.get_next_action(profile, marginal_probabilities=marginal_probabilities)

    assert decision.action_type == "stop_and_infer"
    assert decision.selected_field is None


def test_get_next_action_runs_deterministic_rule_before_scoring() -> None:
    """Verify deterministic inference can resolve missing fields before utility scoring."""
    agent = AdaptiveQuestionnaireAgent()
    profile = _build_profile_with_known_values().model_copy(
        update={
            "loan_primary_purpose": LoanPrimaryPurpose.REFINANCE,
            "currently_have_mortgage": None,
        }
    )

    decision = agent.get_next_action(profile, marginal_probabilities={})

    assert decision.action_type == "stop_and_infer"
    assert decision.selected_field is None
