"""Integration tests for deterministic inference and information math flow."""

import pytest

from backend.core_logic.deterministic_rules import DeterministicRulesEngine, PartialUserProfile
from backend.core_logic.information_math import (
    calculate_expected_information_gain,
    calculate_shannon_entropy,
)
from backend.data_generation.enums import LoanPrimaryPurpose


@pytest.mark.integration
def test_refinance_rule_and_information_gain_flow() -> None:
    """Verify deterministic inference integrates with entropy-based calculations."""
    rules_engine = DeterministicRulesEngine()
    profile = PartialUserProfile(
        loan_primary_purpose=LoanPrimaryPurpose.REFINANCE,
        currently_have_mortgage=None,
    )
    rule_result = rules_engine.apply_rules(profile)
    assert rule_result.updated_profile.currently_have_mortgage is True
    assert len(rule_result.inferred_fields) == 1

    prior_distribution = {True: 0.5, False: 0.5}
    posterior_distributions = {
        "loan_purpose_refinance": {True: 1.0, False: 0.0},
        "loan_purpose_purchase": {True: 0.4, False: 0.6},
    }
    marginal_probabilities = {
        "loan_purpose_refinance": 0.3,
        "loan_purpose_purchase": 0.7,
    }

    prior_entropy = calculate_shannon_entropy(prior_distribution)
    information_gain = calculate_expected_information_gain(
        prior_distribution=prior_distribution,
        posterior_distributions=posterior_distributions,
        marginal_probabilities=marginal_probabilities,
    )

    assert prior_entropy > 0.0
    assert information_gain > 0.0
    assert information_gain < prior_entropy
