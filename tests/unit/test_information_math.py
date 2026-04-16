"""Unit tests for information-theory utility functions."""

import pytest

from backend.core_logic.information_math import (
    calculate_expected_information_gain,
    calculate_shannon_entropy,
)


def test_calculate_shannon_entropy_is_maximal_for_uniform_binary_distribution() -> None:
    """Verify binary uniform distribution has maximal one-bit entropy."""
    entropy_value = calculate_shannon_entropy({"yes": 0.5, "no": 0.5})
    assert entropy_value == pytest.approx(1.0)


def test_calculate_shannon_entropy_is_zero_for_certain_outcome() -> None:
    """Verify fully certain distribution has zero entropy."""
    entropy_value = calculate_shannon_entropy({"certain": 1.0, "impossible": 0.0})
    assert entropy_value == pytest.approx(0.0)


def test_calculate_expected_information_gain_for_perfect_question() -> None:
    """Verify perfect split question yields full prior entropy as gain."""
    prior_distribution = {"true": 0.5, "false": 0.5}
    posterior_distributions = {
        "answer_yes": {"true": 1.0, "false": 0.0},
        "answer_no": {"true": 0.0, "false": 1.0},
    }
    marginal_probabilities = {"answer_yes": 0.5, "answer_no": 0.5}

    information_gain = calculate_expected_information_gain(
        prior_distribution=prior_distribution,
        posterior_distributions=posterior_distributions,
        marginal_probabilities=marginal_probabilities,
    )

    assert information_gain == pytest.approx(1.0)


def test_calculate_expected_information_gain_is_zero_without_entropy_reduction() -> None:
    """Verify unchanged posterior uncertainty yields zero information gain."""
    prior_distribution = {"a": 0.7, "b": 0.3}
    posterior_distributions = {
        "x": {"a": 0.7, "b": 0.3},
        "y": {"a": 0.7, "b": 0.3},
    }
    marginal_probabilities = {"x": 0.4, "y": 0.6}

    information_gain = calculate_expected_information_gain(
        prior_distribution=prior_distribution,
        posterior_distributions=posterior_distributions,
        marginal_probabilities=marginal_probabilities,
    )

    assert information_gain == pytest.approx(0.0)


def test_calculate_expected_information_gain_raises_for_mismatched_outcomes() -> None:
    """Verify mismatched posterior and marginal keys are rejected."""
    with pytest.raises(ValueError):
        calculate_expected_information_gain(
            prior_distribution={"a": 0.5, "b": 0.5},
            posterior_distributions={"x": {"a": 0.9, "b": 0.1}},
            marginal_probabilities={"y": 1.0},
        )
