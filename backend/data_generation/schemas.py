"""Pydantic schemas for data-generation domain models."""

from __future__ import annotations

from typing import Annotated, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.data_generation.config import PROBABILITY_SUM_TOLERANCE
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

Probability = Annotated[float, Field(strict=True, ge=0.0, le=1.0)]
ConfidenceScore = Annotated[float, Field(strict=True, ge=0.0, le=1.0)]
ProfileFieldValue: TypeAlias = (
    CreditScoreRate
    | LoanPrimaryPurpose
    | PropertyType
    | PropertyUse
    | AnnualIncomeBand
    | PropertyValueBand
    | CreditLineBand
    | AgeBand
    | bool
    | str
)


def _validate_distribution_sum(field_name: str, distribution: dict[object, Probability]) -> None:
    """Validate that a categorical distribution sums to one."""
    probability_sum: float = sum(distribution.values())
    if abs(probability_sum - 1.0) > PROBABILITY_SUM_TOLERANCE:
        msg = f"Probabilities for {field_name} must sum to 1.0, got {probability_sum:.6f}."
        raise ValueError(msg)


class UserProfile(BaseModel):
    """Represent a complete synthetic user profile over the ten target fields."""

    model_config = ConfigDict(extra="forbid", strict=True)

    credit_score_rate: CreditScoreRate
    loan_primary_purpose: LoanPrimaryPurpose
    property_type: PropertyType
    property_use: PropertyUse
    annual_income_band: AnnualIncomeBand
    property_value_band: PropertyValueBand
    credit_line_band: CreditLineBand
    age_band: AgeBand
    currently_have_mortgage: bool
    military_veteran: bool


class ObservedAnswer(BaseModel):
    """Represent a user-provided answer observed by the adaptive engine."""

    model_config = ConfigDict(extra="forbid", strict=True)

    field_name: TargetField
    value: ProfileFieldValue


class InferredField(BaseModel):
    """Represent a model-inferred target value and confidence score."""

    model_config = ConfigDict(extra="forbid", strict=True)

    field_name: TargetField
    inferred_value: ProfileFieldValue
    confidence: ConfidenceScore
    inference_reason: str = Field(min_length=1)


class CohortBaseProbabilities(BaseModel):
    """Represent cohort-level base distributions over all target fields."""

    model_config = ConfigDict(extra="forbid")

    credit_score_rate: dict[CreditScoreRate, Probability]
    loan_primary_purpose: dict[LoanPrimaryPurpose, Probability]
    property_type: dict[PropertyType, Probability]
    property_use: dict[PropertyUse, Probability]
    annual_income_band: dict[AnnualIncomeBand, Probability]
    property_value_band: dict[PropertyValueBand, Probability]
    credit_line_band: dict[CreditLineBand, Probability]
    age_band: dict[AgeBand, Probability]
    currently_have_mortgage: dict[bool, Probability]
    military_veteran: dict[bool, Probability]

    @model_validator(mode="after")
    def validate_probability_totals(self) -> CohortBaseProbabilities:
        """Ensure each field-level distribution remains a proper probability vector."""
        _validate_distribution_sum("credit_score_rate", self.credit_score_rate)
        _validate_distribution_sum("loan_primary_purpose", self.loan_primary_purpose)
        _validate_distribution_sum("property_type", self.property_type)
        _validate_distribution_sum("property_use", self.property_use)
        _validate_distribution_sum("annual_income_band", self.annual_income_band)
        _validate_distribution_sum("property_value_band", self.property_value_band)
        _validate_distribution_sum("credit_line_band", self.credit_line_band)
        _validate_distribution_sum("age_band", self.age_band)
        _validate_distribution_sum("currently_have_mortgage", self.currently_have_mortgage)
        _validate_distribution_sum("military_veteran", self.military_veteran)
        return self


class CohortDefinition(BaseModel):
    """Represent a synthetic population cohort with base target distributions."""

    model_config = ConfigDict(extra="forbid")

    cohort_name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    base_probabilities: CohortBaseProbabilities


class CohortCatalog(BaseModel):
    """Wrap all cohort definitions loaded from configuration."""

    model_config = ConfigDict(extra="forbid")

    cohorts: list[CohortDefinition] = Field(min_length=1)


class GeneratedUserRecord(BaseModel):
    """Represent one generated user row with profile, cohort, and latent metadata."""

    model_config = ConfigDict(extra="forbid", strict=True)

    cohort_name: str = Field(min_length=1)
    zipcode: str = Field(min_length=1)
    affluence_score: Probability
    risk_profile: Probability
    user_profile: UserProfile
