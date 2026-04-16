"""Adaptive questionnaire agent that applies rules and utility-based selection."""

from __future__ import annotations

from typing import Literal

from loguru import logger
from pydantic import BaseModel, ConfigDict

from backend.core_logic.config import FRICTION_WEIGHT, MIN_UTILITY_THRESHOLD
from backend.core_logic.deterministic_rules import (
    DeterministicRulesEngine,
    PartialUserProfile,
)
from backend.core_logic.field_mappings import FIELD_NAME_BY_TARGET
from backend.core_logic.information_math import calculate_expected_information_gain
from backend.core_logic.question_bank import QUESTION_BANK
from backend.core_logic.scoring import calculate_question_utility
from backend.data_generation.enums import TargetField


class NextActionDecision(BaseModel):
    """Represent the agent's next-step decision in the adaptive loop."""

    model_config = ConfigDict(extra="forbid", strict=True)

    action_type: Literal["ask_question", "stop_and_infer"]
    selected_field: TargetField | None
    utility_score: float


class AdaptiveQuestionnaireAgent:
    """Select next questions by deterministic inference and utility maximization."""

    def __init__(
        self,
        deterministic_rules_engine: DeterministicRulesEngine | None = None,
        friction_weight: float = FRICTION_WEIGHT,
        min_utility_threshold: float = MIN_UTILITY_THRESHOLD,
    ) -> None:
        """Initialize agent dependencies and scoring thresholds."""
        self._deterministic_rules_engine: DeterministicRulesEngine = (
            deterministic_rules_engine or DeterministicRulesEngine()
        )
        self.friction_weight: float = friction_weight
        self.min_utility_threshold: float = min_utility_threshold

    def get_next_action(
        self,
        partial_profile: PartialUserProfile,
        marginal_probabilities: dict[TargetField, dict[str, float]],
    ) -> NextActionDecision:
        """Compute the next ask-vs-stop decision for a partially known profile."""
        deterministic_result = self._deterministic_rules_engine.apply_rules(partial_profile)
        updated_profile: PartialUserProfile = deterministic_result.updated_profile
        missing_fields: list[TargetField] = self._get_missing_fields(updated_profile)

        if not missing_fields:
            return NextActionDecision(
                action_type="stop_and_infer",
                selected_field=None,
                utility_score=0.0,
            )

        best_field: TargetField | None = None
        best_utility: float = float("-inf")
        for missing_field in missing_fields:
            if missing_field not in marginal_probabilities:
                logger.warning(
                    "Missing marginal probabilities for field",
                    target_field=missing_field.value,
                )
                continue

            field_distribution: dict[str, float] = marginal_probabilities[missing_field]
            expected_information_gain: float = self._calculate_field_eig(field_distribution)
            friction_cost: float = QUESTION_BANK[missing_field].friction_cost
            utility_score: float = calculate_question_utility(
                expected_information_gain=expected_information_gain,
                friction_cost=friction_cost,
                friction_weight=self.friction_weight,
            )
            logger.debug(
                "Scored field utility",
                target_field=missing_field.value,
                expected_information_gain=expected_information_gain,
                friction_cost=friction_cost,
                utility_score=utility_score,
            )
            if utility_score > best_utility:
                best_field = missing_field
                best_utility = utility_score

        if best_field is None:
            return NextActionDecision(
                action_type="stop_and_infer",
                selected_field=None,
                utility_score=0.0,
            )

        if best_utility < self.min_utility_threshold:
            return NextActionDecision(
                action_type="stop_and_infer",
                selected_field=None,
                utility_score=best_utility,
            )

        return NextActionDecision(
            action_type="ask_question",
            selected_field=best_field,
            utility_score=best_utility,
        )

    def _get_missing_fields(self, profile: PartialUserProfile) -> list[TargetField]:
        """Return all target fields that are currently unknown."""
        missing_fields: list[TargetField] = []
        for target_field, field_name in FIELD_NAME_BY_TARGET.items():
            if getattr(profile, field_name) is None:
                missing_fields.append(target_field)
        return missing_fields

    def _calculate_field_eig(self, prior_distribution: dict[str, float]) -> float:
        """Calculate EIG under direct-answer assumption for a candidate field."""
        posterior_distributions: dict[str, dict[str, float]] = {
            outcome: {
                distribution_outcome: 1.0 if distribution_outcome == outcome else 0.0
                for distribution_outcome in prior_distribution
            }
            for outcome in prior_distribution
        }
        return calculate_expected_information_gain(
            prior_distribution=prior_distribution,
            posterior_distributions=posterior_distributions,
            marginal_probabilities=prior_distribution,
        )
