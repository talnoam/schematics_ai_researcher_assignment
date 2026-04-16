"""Deterministic business rules for inferring guaranteed profile fields."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict
from loguru import logger

from backend.core_logic.config import DETERMINISTIC_CONFIDENCE, DETERMINISTIC_MAX_PASSES
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
from backend.data_generation.schemas import InferredField


class PartialUserProfile(BaseModel):
    """Represent a partially known user profile for rule-based inference."""

    model_config = ConfigDict(extra="forbid", strict=True)

    credit_score_rate: CreditScoreRate | None = None
    loan_primary_purpose: LoanPrimaryPurpose | None = None
    property_type: PropertyType | None = None
    property_use: PropertyUse | None = None
    annual_income_band: AnnualIncomeBand | None = None
    property_value_band: PropertyValueBand | None = None
    credit_line_band: CreditLineBand | None = None
    age_band: AgeBand | None = None
    currently_have_mortgage: bool | None = None
    military_veteran: bool | None = None
    zipcode: str | None = None


class DeterministicRuleResult(BaseModel):
    """Represent deterministic inference output with trace metadata."""

    model_config = ConfigDict(extra="forbid", strict=True)

    updated_profile: PartialUserProfile
    inferred_fields: list[InferredField]


class DeterministicRulesEngine:
    """Apply deterministic business rules to infer guaranteed field values."""

    def apply_rules(self, profile: PartialUserProfile) -> DeterministicRuleResult:
        """Infer deterministically implied profile fields and return updates."""
        updated_profile: PartialUserProfile = profile.model_copy(deep=True)
        inferred_fields: list[InferredField] = []

        for _ in range(DETERMINISTIC_MAX_PASSES):
            inferences_before_pass: int = len(inferred_fields)
            self._apply_refinance_rule(updated_profile, inferred_fields)
            if len(inferred_fields) == inferences_before_pass:
                break

        logger.info(
            "Applied deterministic rules",
            inferred_count=len(inferred_fields),
        )
        return DeterministicRuleResult(
            updated_profile=updated_profile,
            inferred_fields=inferred_fields,
        )

    def _apply_refinance_rule(
        self,
        profile: PartialUserProfile,
        inferred_fields: list[InferredField],
    ) -> None:
        """Infer mortgage ownership when refinancing is explicitly selected."""
        if profile.loan_primary_purpose != LoanPrimaryPurpose.REFINANCE:
            return

        if profile.currently_have_mortgage is None:
            profile.currently_have_mortgage = True
            inferred_fields.append(
                InferredField(
                    field_name=TargetField.CURRENTLY_HAVE_MORTGAGE,
                    inferred_value=True,
                    confidence=DETERMINISTIC_CONFIDENCE,
                    inference_reason=(
                        "Refinancing requires an existing mortgage, so mortgage ownership is true."
                    ),
                )
            )
            logger.debug(
                "Deterministic inference applied",
                rule_name="refinance_requires_existing_mortgage",
                inferred_field=TargetField.CURRENTLY_HAVE_MORTGAGE.value,
            )
            return

        if profile.currently_have_mortgage is False:
            logger.warning(
                "Deterministic contradiction detected",
                rule_name="refinance_requires_existing_mortgage",
                loan_primary_purpose=profile.loan_primary_purpose.value,
                currently_have_mortgage=profile.currently_have_mortgage,
            )
