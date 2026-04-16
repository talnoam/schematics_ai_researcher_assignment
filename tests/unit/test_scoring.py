"""Unit tests for adaptive question utility scoring."""

import pytest

from backend.core_logic.scoring import calculate_question_utility


def test_calculate_question_utility_applies_friction_penalty() -> None:
    """Verify utility subtracts weighted friction from expected gain."""
    utility = calculate_question_utility(
        expected_information_gain=0.8,
        friction_cost=0.4,
        friction_weight=0.5,
    )
    assert utility == pytest.approx(0.6)


def test_calculate_question_utility_drops_with_higher_friction_cost() -> None:
    """Verify higher friction yields lower utility for identical information gain."""
    low_friction_utility = calculate_question_utility(
        expected_information_gain=0.7,
        friction_cost=0.2,
        friction_weight=0.5,
    )
    high_friction_utility = calculate_question_utility(
        expected_information_gain=0.7,
        friction_cost=0.9,
        friction_weight=0.5,
    )
    assert low_friction_utility > high_friction_utility
