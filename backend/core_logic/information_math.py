"""Information-theoretic utility functions for adaptive question selection."""

from __future__ import annotations

from typing import TypeVar

import numpy as np
from loguru import logger

from backend.core_logic.config import ENTROPY_EPSILON, PROBABILITY_SUM_TOLERANCE

DistributionKeyT = TypeVar("DistributionKeyT")
OutcomeKeyT = TypeVar("OutcomeKeyT")


def _validate_distribution_sum(
    distribution: dict[DistributionKeyT, float],
    label: str,
) -> None:
    """Validate that a probability distribution sums to one."""
    distribution_sum: float = float(sum(distribution.values()))
    if abs(distribution_sum - 1.0) > PROBABILITY_SUM_TOLERANCE:
        msg = f"Distribution '{label}' must sum to 1.0, got {distribution_sum:.6f}."
        raise ValueError(msg)


def calculate_shannon_entropy(probability_distribution: dict[DistributionKeyT, float]) -> float:
    """Calculate Shannon entropy in bits for a categorical distribution."""
    if not probability_distribution:
        msg = "Probability distribution must not be empty."
        raise ValueError(msg)

    if any(probability < 0.0 for probability in probability_distribution.values()):
        msg = "Probability values must be non-negative."
        raise ValueError(msg)

    _validate_distribution_sum(probability_distribution, label="shannon_entropy")

    probabilities: np.ndarray = np.array(list(probability_distribution.values()), dtype=float)
    safe_probabilities: np.ndarray = np.clip(probabilities, ENTROPY_EPSILON, 1.0)
    entropy_components: np.ndarray = -safe_probabilities * np.log2(safe_probabilities)
    entropy_components = np.where(probabilities > 0.0, entropy_components, 0.0)
    entropy_value: float = float(entropy_components.sum())
    return entropy_value


def calculate_expected_information_gain(
    prior_distribution: dict[DistributionKeyT, float],
    posterior_distributions: dict[OutcomeKeyT, dict[DistributionKeyT, float]],
    marginal_probabilities: dict[OutcomeKeyT, float],
) -> float:
    """Calculate expected entropy reduction for a candidate question."""
    if not posterior_distributions:
        msg = "Posterior distributions must not be empty."
        raise ValueError(msg)
    if not marginal_probabilities:
        msg = "Marginal probabilities must not be empty."
        raise ValueError(msg)

    posterior_keys: set[OutcomeKeyT] = set(posterior_distributions)
    marginal_keys: set[OutcomeKeyT] = set(marginal_probabilities)
    if posterior_keys != marginal_keys:
        msg = "Posterior and marginal outcome keys must match."
        raise ValueError(msg)

    if any(probability < 0.0 for probability in marginal_probabilities.values()):
        msg = "Marginal probabilities must be non-negative."
        raise ValueError(msg)

    _validate_distribution_sum(marginal_probabilities, label="marginal_probabilities")

    prior_entropy: float = calculate_shannon_entropy(prior_distribution)
    expected_posterior_entropy: float = 0.0

    for outcome_key, posterior_distribution in posterior_distributions.items():
        posterior_entropy: float = calculate_shannon_entropy(posterior_distribution)
        expected_posterior_entropy += marginal_probabilities[outcome_key] * posterior_entropy

    information_gain: float = prior_entropy - expected_posterior_entropy
    logger.debug(
        "Calculated expected information gain",
        prior_entropy=prior_entropy,
        expected_posterior_entropy=expected_posterior_entropy,
        information_gain=information_gain,
    )
    return information_gain
