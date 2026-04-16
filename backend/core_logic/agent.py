"""Adaptive questionnaire agent that applies rules and utility-based selection."""

from __future__ import annotations

from typing import Literal

from loguru import logger
from pydantic import BaseModel, ConfigDict

from backend.core_logic.config import AUTO_INFER_THRESHOLD, FRICTION_WEIGHT, MIN_UTILITY_THRESHOLD
from backend.core_logic.deterministic_rules import (
    DeterministicRulesEngine,
    PartialUserProfile,
)
from backend.core_logic.field_mappings import (
    FIELD_NAME_BY_TARGET,
    coerce_marginal_outcome_value,
)
from backend.core_logic.information_math import calculate_expected_information_gain
from backend.core_logic.question_bank import QUESTION_BANK
from backend.core_logic.scoring import calculate_question_utility
from backend.data_generation.enums import TargetField
from backend.data_generation.schemas import InferredField, ProfileFieldValue


class NextActionDecision(BaseModel):
    """Represent the agent's next-step decision in the adaptive loop."""

    model_config = ConfigDict(extra="forbid", strict=True)

    action_type: Literal["ask_question", "stop_and_infer"]
    selected_field: TargetField | None
    utility_score: float
    updated_profile: PartialUserProfile
    inferred_fields: list[InferredField]


class ProbabilisticInferenceResult(BaseModel):
    """Represent profile updates inferred from high-confidence cohort priors."""

    model_config = ConfigDict(extra="forbid", strict=True)

    updated_profile: PartialUserProfile
    inferred_fields: list[InferredField]


class AdaptiveQuestionnaireAgent:
    """Select next questions by deterministic inference and utility maximization."""

    def __init__(
        self,
        deterministic_rules_engine: DeterministicRulesEngine | None = None,
        friction_weight: float = FRICTION_WEIGHT,
        min_utility_threshold: float = MIN_UTILITY_THRESHOLD,
        auto_infer_threshold: float = AUTO_INFER_THRESHOLD,
    ) -> None:
        """Initialize agent dependencies and scoring thresholds."""
        self._deterministic_rules_engine: DeterministicRulesEngine = (
            deterministic_rules_engine or DeterministicRulesEngine()
        )
        self.friction_weight: float = friction_weight
        self.min_utility_threshold: float = min_utility_threshold
        self.auto_infer_threshold: float = auto_infer_threshold

    def get_next_action(
        self,
        partial_profile: PartialUserProfile,
        marginal_probabilities: dict[TargetField, dict[str, float]],
    ) -> ProbabilisticInferenceResult:
        """Compute the next ask-vs-stop decision for a partially known profile."""
        deterministic_result = self._deterministic_rules_engine.apply_rules(partial_profile)
        updated_profile: PartialUserProfile = deterministic_result.updated_profile
        inferred_fields: list[InferredField] = list(deterministic_result.inferred_fields)
        probabilistic_result = self._apply_probabilistic_inference(
            profile=updated_profile,
            marginal_probabilities=marginal_probabilities,
        )
        updated_profile = probabilistic_result.updated_profile
        inferred_fields.extend(probabilistic_result.inferred_fields)
        missing_fields: list[TargetField] = self._get_missing_fields(updated_profile)

        if not missing_fields:
            return NextActionDecision(
                action_type="stop_and_infer",
                selected_field=None,
                utility_score=0.0,
                updated_profile=updated_profile,
                inferred_fields=inferred_fields,
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
                updated_profile=updated_profile,
                inferred_fields=inferred_fields,
            )

        if best_utility < self.min_utility_threshold:
            return NextActionDecision(
                action_type="stop_and_infer",
                selected_field=None,
                utility_score=best_utility,
                updated_profile=updated_profile,
                inferred_fields=inferred_fields,
            )

        return NextActionDecision(
            action_type="ask_question",
            selected_field=best_field,
            utility_score=best_utility,
            updated_profile=updated_profile,
            inferred_fields=inferred_fields,
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

    def _apply_probabilistic_inference(
        self,
        profile: PartialUserProfile,
        marginal_probabilities: dict[TargetField, dict[str, float]],
    ) -> NextActionDecision:
        """Infer missing fields when cohort marginals indicate a dominant outcome."""
        updated_profile: PartialUserProfile = profile.model_copy(deep=True)
        inferred_fields: list[InferredField] = []

        for target_field in self._get_missing_fields(updated_profile):
            field_distribution = marginal_probabilities.get(target_field)
            if field_distribution is None:
                continue
            if not field_distribution:
                continue

            best_outcome, best_probability = max(
                field_distribution.items(),
                key=lambda outcome_probability: outcome_probability[1],
            )
            if best_probability < self.auto_infer_threshold:
                continue

            try:
                inferred_value: ProfileFieldValue = coerce_marginal_outcome_value(
                    target_field=target_field,
                    raw_outcome=best_outcome,
                )
            except ValueError:
                logger.warning(
                    "Skipped probabilistic inference due to coercion failure",
                    target_field=target_field.value,
                    raw_outcome=best_outcome,
                )
                continue

            field_name = FIELD_NAME_BY_TARGET[target_field]
            updated_profile = updated_profile.model_copy(update={field_name: inferred_value})
            inferred_fields.append(
                InferredField(
                    field_name=target_field,
                    inferred_value=inferred_value,
                    confidence=best_probability,
                    inference_reason="Probabilistic inference from strong cohort priors",
                )
            )
            logger.debug(
                "Applied probabilistic inference",
                target_field=target_field.value,
                inferred_value=str(inferred_value),
                confidence=best_probability,
            )

        return ProbabilisticInferenceResult(
            updated_profile=updated_profile,
            inferred_fields=inferred_fields,
        )
