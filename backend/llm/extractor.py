"""LLM-based extraction of structured questionnaire fields from free text."""

from __future__ import annotations

import json
import re

from loguru import logger

from backend.core_logic.field_mappings import (
    BOOLEAN_TARGET_FIELDS,
    TARGET_VALUE_ENUM_BY_FIELD,
    coerce_target_field_value,
)
from backend.data_generation.enums import TargetField
from backend.data_generation.enums import (
    AgeBand,
    AnnualIncomeBand,
    CreditLineBand,
    LoanPrimaryPurpose,
    PropertyType,
    PropertyUse,
)
from backend.data_generation.schemas import ProfileFieldValue
from backend.llm.client import OllamaClient

_SYSTEM_PROMPT: str = (
    "You are a strict information extraction assistant. "
    "Extract only explicitly supported field values from the user text. "
    "Return ONLY valid JSON object text without markdown, prose, or comments."
)
_FEW_SHOT_EXAMPLES: str = """
Few-shot examples:
Example 1
User text: "My household income is about $85,000 per year."
Expected JSON: {"annual_income_band":"50k_to_100k"}

Example 2
User text: "I make around 220k annually."
Expected JSON: {"annual_income_band":"above_200k"}

Example 3
User text: "I work in tech and have a stable job."
Expected JSON: {}

Example 4
User text: "I am 33 years old and looking to purchase a condo as my primary residence."
Expected JSON: {"age_band":"30_to_44","loan_primary_purpose":"purchase","property_type":"condo","property_use":"primary_residence"}

Example 5
User text: "I already own a home and want to refinance."
Expected JSON: {"loan_primary_purpose":"refinance","currently_have_mortgage":true}
"""

_AMOUNT_PATTERN: re.Pattern[str] = re.compile(
    r"\$(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*([kK])?|(\d+(?:\.\d+)?)\s*([kK])\b"
)
_AGE_YEARS_OLD_PATTERN: re.Pattern[str] = re.compile(r"\b(\d{1,3})\s+years?\s+old\b", re.IGNORECASE)
_AGE_I_AM_PATTERN: re.Pattern[str] = re.compile(r"\b(?:i am|i'm)\s+(\d{1,3})\b", re.IGNORECASE)


async def extract_fields_from_text(
    user_text: str,
    target_fields: list[TargetField],
    llm_client: OllamaClient | None = None,
) -> dict[TargetField, ProfileFieldValue]:
    """Extract typed target fields from free-form user text."""
    if not target_fields:
        return {}

    client: OllamaClient = llm_client or OllamaClient()
    schema_block: str = _build_schema_block(target_fields)
    user_prompt: str = (
        "Extract values for the following target fields if they are explicitly mentioned.\n"
        f"{schema_block}\n"
        "Rules:\n"
        "- Return ONLY a JSON object where keys are target field names.\n"
        "- Omit fields that are not mentioned or uncertain.\n"
        "- Never invent values.\n"
        "- If annual income is not explicit, omit annual_income_band.\n"
        f"{_FEW_SHOT_EXAMPLES}\n"
        f"User text:\n{user_text}"
    )
    raw_completion: str = await client.generate_json_completion(
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )
    logger.info(f"Raw LLM Extraction Payload: {raw_completion}")
    llm_extracted_fields = _parse_and_coerce_extraction(raw_completion, target_fields)
    rule_based_extracted_fields = _extract_rule_based_fields(user_text, target_fields)

    # Prefer explicit deterministic parsing when text contains clear evidence.
    merged_extracted_fields: dict[TargetField, ProfileFieldValue] = dict(llm_extracted_fields)
    merged_extracted_fields.update(rule_based_extracted_fields)
    logger.info(
        "Merged extracted structured fields",
        llm_extracted_count=len(llm_extracted_fields),
        rule_based_extracted_count=len(rule_based_extracted_fields),
        merged_extracted_count=len(merged_extracted_fields),
    )
    return merged_extracted_fields


def _build_schema_block(target_fields: list[TargetField]) -> str:
    """Build prompt schema text with allowed values per target field."""
    schema_lines: list[str] = ["Allowed fields and values:"]
    for target_field in target_fields:
        if target_field in BOOLEAN_TARGET_FIELDS:
            allowed_values: list[str] = ["true", "false"]
        else:
            enum_class = TARGET_VALUE_ENUM_BY_FIELD[target_field]
            allowed_values = [enum_member.value for enum_member in enum_class]
        schema_lines.append(f"- {target_field.value}: {allowed_values}")
    return "\n".join(schema_lines)


def _parse_and_coerce_extraction(
    raw_json_text: str,
    target_fields: list[TargetField],
) -> dict[TargetField, ProfileFieldValue]:
    """Parse LLM JSON output and coerce extracted values to typed fields."""
    try:
        parsed_payload: object = json.loads(raw_json_text)
    except json.JSONDecodeError:
        logger.warning("Failed to parse extraction JSON payload", raw_response=raw_json_text)
        return {}

    if not isinstance(parsed_payload, dict):
        logger.warning("Extraction payload is not a JSON object", raw_response=raw_json_text)
        return {}

    allowed_fields: set[TargetField] = set(target_fields)
    coerced_result: dict[TargetField, ProfileFieldValue] = {}
    for field_name, raw_value in parsed_payload.items():
        if not isinstance(field_name, str):
            continue

        try:
            target_field: TargetField = TargetField(field_name)
        except ValueError:
            logger.debug("Skipping unsupported extracted field", field_name=field_name)
            continue

        if target_field not in allowed_fields:
            continue

        if not isinstance(raw_value, (str, bool)):
            logger.debug(
                "Skipping extracted field with invalid value type",
                target_field=target_field.value,
            )
            continue

        try:
            coerced_result[target_field] = coerce_target_field_value(target_field, raw_value)
        except ValueError:
            logger.debug(
                "Skipping extracted field with uncoercible value",
                target_field=target_field.value,
                raw_value=raw_value,
            )

    logger.info("Extracted structured fields from text", extracted_count=len(coerced_result))
    return coerced_result


def _extract_rule_based_fields(
    user_text: str,
    target_fields: list[TargetField],
) -> dict[TargetField, ProfileFieldValue]:
    """Extract explicit structured fields using deterministic text patterns."""
    extracted_fields: dict[TargetField, ProfileFieldValue] = {}
    lowered_text: str = user_text.lower()
    allowed_fields: set[TargetField] = set(target_fields)

    if TargetField.LOAN_PRIMARY_PURPOSE in allowed_fields:
        loan_primary_purpose = _extract_loan_primary_purpose(lowered_text)
        if loan_primary_purpose is not None:
            extracted_fields[TargetField.LOAN_PRIMARY_PURPOSE] = loan_primary_purpose

    if TargetField.PROPERTY_USE in allowed_fields:
        property_use = _extract_property_use(lowered_text)
        if property_use is not None:
            extracted_fields[TargetField.PROPERTY_USE] = property_use

    if TargetField.PROPERTY_TYPE in allowed_fields:
        property_type = _extract_property_type(lowered_text)
        if property_type is not None:
            extracted_fields[TargetField.PROPERTY_TYPE] = property_type

    if TargetField.ANNUAL_INCOME_BAND in allowed_fields:
        annual_income_band = _extract_annual_income_band(lowered_text)
        if annual_income_band is not None:
            extracted_fields[TargetField.ANNUAL_INCOME_BAND] = annual_income_band

    if TargetField.CREDIT_LINE_BAND in allowed_fields:
        credit_line_band = _extract_credit_line_band(lowered_text)
        if credit_line_band is not None:
            extracted_fields[TargetField.CREDIT_LINE_BAND] = credit_line_band

    if TargetField.AGE_BAND in allowed_fields:
        age_band = _extract_age_band(lowered_text)
        if age_band is not None:
            extracted_fields[TargetField.AGE_BAND] = age_band

    if TargetField.MILITARY_VETERAN in allowed_fields:
        military_veteran = _extract_military_veteran(lowered_text)
        if military_veteran is not None:
            extracted_fields[TargetField.MILITARY_VETERAN] = military_veteran

    if TargetField.CURRENTLY_HAVE_MORTGAGE in allowed_fields:
        currently_have_mortgage = _extract_currently_have_mortgage(lowered_text)
        if currently_have_mortgage is not None:
            extracted_fields[TargetField.CURRENTLY_HAVE_MORTGAGE] = currently_have_mortgage

    return extracted_fields


def _extract_loan_primary_purpose(text: str) -> LoanPrimaryPurpose | None:
    """Extract loan purpose when explicit purpose phrasing is present."""
    if "refinance" in text:
        return LoanPrimaryPurpose.REFINANCE
    if "home improvement" in text:
        return LoanPrimaryPurpose.HOME_IMPROVEMENT
    if "debt consolidation" in text:
        return LoanPrimaryPurpose.DEBT_CONSOLIDATION
    if "investment" in text and ("property" in text or "home" in text):
        return LoanPrimaryPurpose.INVESTMENT
    purchase_cues: tuple[str, ...] = (
        "purchase",
        "buy",
        "buying",
        "looking to purchase",
        "planning to purchase",
    )
    if any(cue in text for cue in purchase_cues):
        return LoanPrimaryPurpose.PURCHASE
    return None


def _extract_property_use(text: str) -> PropertyUse | None:
    """Extract property use from explicit occupancy intent phrases."""
    if "primary residence" in text:
        return PropertyUse.PRIMARY_RESIDENCE
    if "second home" in text:
        return PropertyUse.SECOND_HOME
    if "investment" in text:
        return PropertyUse.INVESTMENT
    return None


def _extract_property_type(text: str) -> PropertyType | None:
    """Extract property type from explicit property descriptors."""
    if "single family" in text or "single-family" in text:
        return PropertyType.SINGLE_FAMILY
    if "condo" in text:
        return PropertyType.CONDO
    if "townhouse" in text:
        return PropertyType.TOWNHOUSE
    if "multi-family" in text or "multi family" in text:
        return PropertyType.MULTI_FAMILY
    if "mobile home" in text:
        return PropertyType.MOBILE_HOME
    return None


def _extract_annual_income_band(text: str) -> AnnualIncomeBand | None:
    """Extract annual income band from explicit income-related amount mentions."""
    income_cues: tuple[str, ...] = (
        "income",
        "salary",
        "earn",
        "make",
        "annually",
        "per year",
        "a year",
    )
    if not any(cue in text for cue in income_cues):
        return None

    amount = _extract_first_amount(text)
    if amount is None:
        return None
    return _amount_to_annual_income_band(amount)


def _extract_credit_line_band(text: str) -> CreditLineBand | None:
    """Extract credit line band from explicit credit-line amount mentions."""
    credit_line_cues: tuple[str, ...] = (
        "credit line",
        "credit lines",
        "revolving credit",
        "available credit",
    )
    if not any(cue in text for cue in credit_line_cues):
        return None

    amount = _extract_first_amount(text)
    if amount is None:
        return None
    return _amount_to_credit_line_band(amount)


def _extract_age_band(text: str) -> AgeBand | None:
    """Extract age band from explicit age phrases."""
    age_match = _AGE_YEARS_OLD_PATTERN.search(text)
    if age_match is None:
        age_match = _AGE_I_AM_PATTERN.search(text)
    if age_match is None:
        return None
    age_value = int(age_match.group(1))
    if age_value < 30:
        return AgeBand.UNDER_30
    if age_value <= 44:
        return AgeBand.FROM_30_TO_44
    if age_value <= 59:
        return AgeBand.FROM_45_TO_59
    return AgeBand.FROM_60_PLUS


def _extract_military_veteran(text: str) -> bool | None:
    """Extract military veteran flag from explicit veteran statements."""
    if "not a military veteran" in text or "not military veteran" in text:
        return False
    if "military veteran" in text:
        return True
    return None


def _extract_currently_have_mortgage(text: str) -> bool | None:
    """Extract mortgage ownership from explicit mortgage statements."""
    negative_phrases: tuple[str, ...] = (
        "do not have a mortgage",
        "don't have a mortgage",
        "no mortgage",
        "without a mortgage",
    )
    if any(phrase in text for phrase in negative_phrases):
        return False

    positive_phrases: tuple[str, ...] = (
        "currently have a mortgage",
        "already have a mortgage",
        "have a mortgage",
    )
    if any(phrase in text for phrase in positive_phrases):
        return True
    return None


def _extract_first_amount(text: str) -> float | None:
    """Extract the first numeric amount with optional thousand suffix from text."""
    amount_match = _AMOUNT_PATTERN.search(text)
    if amount_match is None:
        return None

    dollar_number = amount_match.group(1)
    dollar_suffix = amount_match.group(2)
    bare_number = amount_match.group(3)
    bare_suffix = amount_match.group(4)

    if dollar_number is not None:
        raw_amount = float(dollar_number.replace(",", ""))
        if dollar_suffix is not None:
            raw_amount *= 1000.0
        return raw_amount

    if bare_number is not None:
        raw_amount = float(bare_number)
        if bare_suffix is not None:
            raw_amount *= 1000.0
        return raw_amount

    return None


def _amount_to_annual_income_band(amount: float) -> AnnualIncomeBand:
    """Map annual income amount to the corresponding band enum."""
    if amount < 50_000:
        return AnnualIncomeBand.UNDER_50K
    if amount < 100_000:
        return AnnualIncomeBand.FROM_50K_TO_100K
    if amount < 200_000:
        return AnnualIncomeBand.FROM_100K_TO_200K
    return AnnualIncomeBand.ABOVE_200K


def _amount_to_credit_line_band(amount: float) -> CreditLineBand:
    """Map credit line amount to the corresponding band enum."""
    if amount < 10_000:
        return CreditLineBand.UNDER_10K
    if amount < 50_000:
        return CreditLineBand.FROM_10K_TO_50K
    if amount < 100_000:
        return CreditLineBand.FROM_50K_TO_100K
    return CreditLineBand.ABOVE_100K
