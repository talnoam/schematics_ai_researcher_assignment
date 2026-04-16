"""Generate synthetic users from cohorts and latent-factor adjustments."""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import numpy as np
import pandas as pd
from loguru import logger

from backend.data_generation.cohort_loader import CohortLoader
from backend.data_generation.config import (
    COHORT_WEIGHT_SUM_TOLERANCE,
    DEFAULT_COHORT_SAMPLING_WEIGHTS,
    PREMIUM_AFFLUENCE_THRESHOLD,
    ZIPCODE_BY_PROPERTY_VALUE_BAND,
    ZIPCODE_PREMIUM_POOL,
)
from backend.data_generation.enums import (
    AgeBand,
    AnnualIncomeBand,
    CreditLineBand,
    CreditScoreRate,
    LoanPrimaryPurpose,
    PropertyType,
    PropertyUse,
    PropertyValueBand,
)
from backend.data_generation.latent_factors import LatentFactorModel, LatentFactors
from backend.data_generation.schemas import CohortBaseProbabilities, CohortDefinition, GeneratedUserRecord, UserProfile

DistributionKeyT = TypeVar("DistributionKeyT")


class MockDataGenerator:
    """Generate synthetic user records from cohorts and latent factors."""

    def __init__(
        self,
        cohort_weights: dict[str, float] | None = None,
        random_seed: int | None = None,
        cohort_definitions_path: Path | None = None,
    ) -> None:
        """Initialize generation dependencies and random state."""
        self.random_seed: int | None = random_seed
        self._rng: np.random.Generator = np.random.default_rng(random_seed)
        self._cohort_loader: CohortLoader = CohortLoader(file_path=cohort_definitions_path)
        self._latent_factor_model: LatentFactorModel = LatentFactorModel(random_seed=random_seed)
        self._cohorts: list[CohortDefinition] = self._cohort_loader.load_definitions()
        self._cohort_by_name: dict[str, CohortDefinition] = {
            cohort_definition.cohort_name: cohort_definition for cohort_definition in self._cohorts
        }
        self._cohort_weights: dict[str, float] = self._validate_cohort_weights(
            cohort_weights=cohort_weights or DEFAULT_COHORT_SAMPLING_WEIGHTS
        )

    def generate_user_records(self, user_count: int) -> list[GeneratedUserRecord]:
        """Generate synthetic users with cohort labels and latent-factor metadata."""
        generated_users: list[GeneratedUserRecord] = []
        for _ in range(user_count):
            cohort_definition: CohortDefinition = self._sample_cohort_definition()
            latent_factors: LatentFactors = self._latent_factor_model.sample_latent_factors()
            shifted_probabilities: CohortBaseProbabilities = self._latent_factor_model.adjust_probabilities(
                base_probabilities=cohort_definition.base_probabilities,
                latent_factors=latent_factors,
            )
            user_profile: UserProfile = self._sample_user_profile(shifted_probabilities)
            zipcode: str = self._sample_zipcode(
                property_value_band=user_profile.property_value_band,
                affluence_score=latent_factors.affluence_score,
            )
            generated_users.append(
                GeneratedUserRecord(
                    cohort_name=cohort_definition.cohort_name,
                    zipcode=zipcode,
                    affluence_score=latent_factors.affluence_score,
                    risk_profile=latent_factors.risk_profile,
                    user_profile=user_profile,
                )
            )

        logger.info("Generated synthetic users", user_count=user_count)
        return generated_users

    def generate_dataframe(self, user_count: int) -> pd.DataFrame:
        """Generate synthetic users and return a tabular dataframe."""
        generated_users: list[GeneratedUserRecord] = self.generate_user_records(user_count=user_count)
        records: list[dict[str, object]] = []
        for generated_user in generated_users:
            profile_data: dict[str, object] = generated_user.user_profile.model_dump(mode="json")
            records.append(
                {
                    "cohort_name": generated_user.cohort_name,
                    "zipcode": generated_user.zipcode,
                    "affluence_score": generated_user.affluence_score,
                    "risk_profile": generated_user.risk_profile,
                    **profile_data,
                }
            )

        dataframe: pd.DataFrame = pd.DataFrame.from_records(records)
        return dataframe

    def _sample_cohort_definition(self) -> CohortDefinition:
        """Sample a cohort definition based on configured cohort weights."""
        cohort_names: list[str] = list(self._cohort_weights)
        cohort_probabilities: np.ndarray = np.array(
            [self._cohort_weights[cohort_name] for cohort_name in cohort_names], dtype=float
        )
        sampled_index: int = int(self._rng.choice(len(cohort_names), p=cohort_probabilities))
        sampled_name: str = cohort_names[sampled_index]
        return self._cohort_by_name[sampled_name]

    def _sample_user_profile(self, probabilities: CohortBaseProbabilities) -> UserProfile:
        """Sample a complete profile from shifted field-level distributions."""
        return UserProfile(
            credit_score_rate=self._sample_from_distribution(probabilities.credit_score_rate),
            loan_primary_purpose=self._sample_from_distribution(probabilities.loan_primary_purpose),
            property_type=self._sample_from_distribution(probabilities.property_type),
            property_use=self._sample_from_distribution(probabilities.property_use),
            annual_income_band=self._sample_from_distribution(probabilities.annual_income_band),
            property_value_band=self._sample_from_distribution(probabilities.property_value_band),
            credit_line_band=self._sample_from_distribution(probabilities.credit_line_band),
            age_band=self._sample_from_distribution(probabilities.age_band),
            currently_have_mortgage=self._sample_from_distribution(probabilities.currently_have_mortgage),
            military_veteran=self._sample_from_distribution(probabilities.military_veteran),
        )

    def _sample_zipcode(self, property_value_band: PropertyValueBand, affluence_score: float) -> str:
        """Sample a zipcode using property value and affluence signals."""
        zipcode_pool: tuple[str, ...]
        if affluence_score >= PREMIUM_AFFLUENCE_THRESHOLD:
            zipcode_pool = ZIPCODE_PREMIUM_POOL
        else:
            zipcode_pool = ZIPCODE_BY_PROPERTY_VALUE_BAND[property_value_band]

        sampled_index: int = int(self._rng.choice(len(zipcode_pool)))
        return zipcode_pool[sampled_index]

    def _sample_from_distribution(self, distribution: dict[DistributionKeyT, float]) -> DistributionKeyT:
        """Sample one key from a categorical probability distribution."""
        values: list[DistributionKeyT] = list(distribution)
        probabilities: np.ndarray = np.array(list(distribution.values()), dtype=float)
        sampled_index: int = int(self._rng.choice(len(values), p=probabilities))
        return values[sampled_index]

    def _validate_cohort_weights(self, cohort_weights: dict[str, float]) -> dict[str, float]:
        """Validate cohort weights against available cohorts and normalization constraints."""
        for cohort_name in cohort_weights:
            if cohort_name not in self._cohort_by_name:
                msg = f"Unknown cohort weight key: {cohort_name}"
                raise ValueError(msg)

        available_cohort_names: set[str] = set(self._cohort_by_name)
        defined_cohort_names: set[str] = set(cohort_weights)
        missing_cohort_names: set[str] = available_cohort_names - defined_cohort_names
        if missing_cohort_names:
            msg = "Missing cohort weights for: " + ", ".join(sorted(missing_cohort_names))
            raise ValueError(msg)

        weight_sum: float = sum(cohort_weights.values())
        if abs(weight_sum - 1.0) > COHORT_WEIGHT_SUM_TOLERANCE:
            msg = f"Cohort weights must sum to 1.0, got {weight_sum:.6f}."
            raise ValueError(msg)

        return cohort_weights
