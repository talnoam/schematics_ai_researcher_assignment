"""Shared field mappings and coercion helpers for profile values."""

from __future__ import annotations

from enum import StrEnum

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
from backend.data_generation.schemas import ProfileFieldValue

FIELD_NAME_BY_TARGET: dict[TargetField, str] = {
    TargetField.CREDIT_SCORE_RATE: "credit_score_rate",
    TargetField.LOAN_PRIMARY_PURPOSE: "loan_primary_purpose",
    TargetField.PROPERTY_TYPE: "property_type",
    TargetField.PROPERTY_USE: "property_use",
    TargetField.ANNUAL_INCOME_BAND: "annual_income_band",
    TargetField.PROPERTY_VALUE_BAND: "property_value_band",
    TargetField.CREDIT_LINE_BAND: "credit_line_band",
    TargetField.AGE_BAND: "age_band",
    TargetField.CURRENTLY_HAVE_MORTGAGE: "currently_have_mortgage",
    TargetField.MILITARY_VETERAN: "military_veteran",
}

TARGET_VALUE_ENUM_BY_FIELD: dict[TargetField, type[StrEnum]] = {
    TargetField.CREDIT_SCORE_RATE: CreditScoreRate,
    TargetField.LOAN_PRIMARY_PURPOSE: LoanPrimaryPurpose,
    TargetField.PROPERTY_TYPE: PropertyType,
    TargetField.PROPERTY_USE: PropertyUse,
    TargetField.ANNUAL_INCOME_BAND: AnnualIncomeBand,
    TargetField.PROPERTY_VALUE_BAND: PropertyValueBand,
    TargetField.CREDIT_LINE_BAND: CreditLineBand,
    TargetField.AGE_BAND: AgeBand,
}

BOOLEAN_TARGET_FIELDS: set[TargetField] = {
    TargetField.CURRENTLY_HAVE_MORTGAGE,
    TargetField.MILITARY_VETERAN,
}


def coerce_target_field_value(target_field: TargetField, raw_value: str | bool) -> ProfileFieldValue:
    """Coerce one raw answer value into a typed target field value."""
    if target_field in BOOLEAN_TARGET_FIELDS:
        if not isinstance(raw_value, bool):
            msg = f"Answer for '{target_field.value}' must be boolean."
            raise ValueError(msg)
        return raw_value

    if not isinstance(raw_value, str):
        msg = f"Answer for '{target_field.value}' must be a string enum value."
        raise ValueError(msg)

    expected_type: type[StrEnum] = TARGET_VALUE_ENUM_BY_FIELD[target_field]
    try:
        coerced_value: ProfileFieldValue = expected_type(raw_value)
    except ValueError as error:
        msg = f"Invalid value '{raw_value}' for field '{target_field.value}'."
        raise ValueError(msg) from error
    return coerced_value
