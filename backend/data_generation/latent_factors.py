"""Apply latent continuous factors to cohort probability distributions."""

from __future__ import annotations

from typing import TypeVar

import numpy as np
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from backend.data_generation.config import (
    AFFLUENCE_SHIFT_STRENGTH,
    AGE_MORTGAGE_REDUCTION_WEIGHT,
    LATENT_FACTOR_MEAN,
    LATENT_FACTOR_STD,
    RISK_SHIFT_STRENGTH,
)
from backend.data_generation.enums import AgeBand, AnnualIncomeBand, CreditLineBand, CreditScoreRate, PropertyValueBand
from backend.data_generation.schemas import CohortBaseProbabilities

EnumT = TypeVar("EnumT")
EPSILON: float = 1e-9


class LatentFactors(BaseModel):
    """Represent hidden factors that induce cross-field correlations."""

    model_config = ConfigDict(extra="forbid", strict=True)

    affluence_score: float = Field(ge=0.0, le=1.0)
    risk_profile: float = Field(ge=0.0, le=1.0)


class LatentFactorModel:
    """Shift cohort priors using hidden socioeconomic and risk factors."""

    def __init__(self, random_seed: int | None = None) -> None:
        """Initialize the latent-factor model and its random generator."""
        self.random_seed: int | None = random_seed
        self._rng: np.random.Generator = np.random.default_rng(self.random_seed)

    def sample_latent_factors(self) -> LatentFactors:
        """Sample bounded latent variables for one synthetic user."""
        affluence_score: float = float(
            np.clip(self._rng.normal(loc=LATENT_FACTOR_MEAN, scale=LATENT_FACTOR_STD), 0.0, 1.0)
        )
        risk_profile: float = float(
            np.clip(self._rng.normal(loc=LATENT_FACTOR_MEAN, scale=LATENT_FACTOR_STD), 0.0, 1.0)
        )
        return LatentFactors(affluence_score=affluence_score, risk_profile=risk_profile)

    def adjust_probabilities(
        self,
        base_probabilities: CohortBaseProbabilities,
        latent_factors: LatentFactors | None = None,
    ) -> CohortBaseProbabilities:
        """Shift selected base probabilities according to latent-factor realizations."""
        factors: LatentFactors = latent_factors or self.sample_latent_factors()
        representative_age_band: AgeBand = self._mode_age_band(base_probabilities.age_band)

        adjusted_income: dict[AnnualIncomeBand, float] = self._shift_ordered_distribution(
            base_probabilities.annual_income_band,
            enum_order=[
                AnnualIncomeBand.UNDER_50K,
                AnnualIncomeBand.FROM_50K_TO_100K,
                AnnualIncomeBand.FROM_100K_TO_200K,
                AnnualIncomeBand.ABOVE_200K,
            ],
            directional_shift=factors.affluence_score - 0.5,
            strength=AFFLUENCE_SHIFT_STRENGTH,
        )
        adjusted_property_value: dict[PropertyValueBand, float] = self._shift_ordered_distribution(
            base_probabilities.property_value_band,
            enum_order=[
                PropertyValueBand.UNDER_300K,
                PropertyValueBand.FROM_300K_TO_600K,
                PropertyValueBand.FROM_600K_TO_1M,
                PropertyValueBand.ABOVE_1M,
            ],
            directional_shift=factors.affluence_score - 0.5,
            strength=AFFLUENCE_SHIFT_STRENGTH,
        )
        adjusted_credit_line: dict[CreditLineBand, float] = self._shift_ordered_distribution(
            base_probabilities.credit_line_band,
            enum_order=[
                CreditLineBand.UNDER_10K,
                CreditLineBand.FROM_10K_TO_50K,
                CreditLineBand.FROM_50K_TO_100K,
                CreditLineBand.ABOVE_100K,
            ],
            directional_shift=(factors.affluence_score - 0.5) - (factors.risk_profile - 0.5),
            strength=AFFLUENCE_SHIFT_STRENGTH,
        )
        adjusted_credit_score: dict[CreditScoreRate, float] = self._shift_ordered_distribution(
            base_probabilities.credit_score_rate,
            enum_order=[
                CreditScoreRate.POOR,
                CreditScoreRate.FAIR,
                CreditScoreRate.GOOD,
                CreditScoreRate.VERY_GOOD,
                CreditScoreRate.EXCELLENT,
            ],
            directional_shift=(factors.affluence_score - 0.5) - (factors.risk_profile - 0.5),
            strength=RISK_SHIFT_STRENGTH,
        )
        adjusted_mortgage: dict[bool, float] = self._adjust_mortgage_probability(
            base_distribution=base_probabilities.currently_have_mortgage,
            age_band=representative_age_band,
            factors=factors,
        )

        adjusted_probabilities: CohortBaseProbabilities = CohortBaseProbabilities(
            credit_score_rate=adjusted_credit_score,
            loan_primary_purpose=base_probabilities.loan_primary_purpose,
            property_type=base_probabilities.property_type,
            property_use=base_probabilities.property_use,
            annual_income_band=adjusted_income,
            property_value_band=adjusted_property_value,
            credit_line_band=adjusted_credit_line,
            age_band=base_probabilities.age_band,
            currently_have_mortgage=adjusted_mortgage,
            military_veteran=base_probabilities.military_veteran,
        )

        logger.debug(
            "Applied latent-factor probability adjustments",
            affluence_score=factors.affluence_score,
            risk_profile=factors.risk_profile,
            representative_age_band=representative_age_band.value,
        )
        return adjusted_probabilities

    def _shift_ordered_distribution(
        self,
        distribution: dict[EnumT, float],
        enum_order: list[EnumT],
        directional_shift: float,
        strength: float,
    ) -> dict[EnumT, float]:
        """Shift an ordered categorical distribution with a softmax transformation."""
        order_count: int = len(enum_order)
        logits: list[float] = []
        for index, enum_member in enumerate(enum_order):
            base_probability: float = float(distribution[enum_member])
            relative_position: float = (index / (order_count - 1)) - 0.5
            shift_value: float = strength * directional_shift * relative_position
            logits.append(np.log(base_probability + EPSILON) + shift_value)

        shifted_raw: np.ndarray = np.exp(np.array(logits) - np.max(logits))
        normalized: np.ndarray = shifted_raw / shifted_raw.sum()
        return {enum_order[idx]: float(probability) for idx, probability in enumerate(normalized)}

    def _adjust_mortgage_probability(
        self,
        base_distribution: dict[bool, float],
        age_band: AgeBand,
        factors: LatentFactors,
    ) -> dict[bool, float]:
        """Adjust mortgage ownership with an age-weighted affluence effect."""
        age_mortgage_multiplier: dict[AgeBand, float] = {
            AgeBand.UNDER_30: 0.0,
            AgeBand.FROM_30_TO_44: 0.35,
            AgeBand.FROM_45_TO_59: 0.75,
            AgeBand.FROM_60_PLUS: 1.0,
        }
        base_true_probability: float = float(base_distribution[True])
        affluence_reduction: float = (
            factors.affluence_score
            * age_mortgage_multiplier[age_band]
            * AGE_MORTGAGE_REDUCTION_WEIGHT
        )
        risk_increase: float = (factors.risk_profile - 0.5) * 0.1
        shifted_true_probability: float = float(
            np.clip(base_true_probability - affluence_reduction + risk_increase, 0.01, 0.99)
        )
        return {True: shifted_true_probability, False: 1.0 - shifted_true_probability}

    def _mode_age_band(self, age_distribution: dict[AgeBand, float]) -> AgeBand:
        """Select the most probable age band from the cohort prior."""
        return max(age_distribution.items(), key=lambda item: item[1])[0]
