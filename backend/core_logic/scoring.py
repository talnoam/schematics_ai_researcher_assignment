"""Utility scoring functions for adaptive-question selection."""

from loguru import logger


def calculate_question_utility(
    expected_information_gain: float,
    friction_cost: float,
    friction_weight: float,
) -> float:
    """Calculate utility by balancing information gain against friction penalty."""
    utility: float = expected_information_gain - (friction_weight * friction_cost)
    logger.debug(
        "Calculated question utility",
        expected_information_gain=expected_information_gain,
        friction_cost=friction_cost,
        friction_weight=friction_weight,
        utility=utility,
    )
    return utility
