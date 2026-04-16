"""LLM-based extraction of structured questionnaire fields from free text."""

from __future__ import annotations

import json

from loguru import logger

from backend.core_logic.field_mappings import (
    BOOLEAN_TARGET_FIELDS,
    TARGET_VALUE_ENUM_BY_FIELD,
    coerce_target_field_value,
)
from backend.data_generation.enums import TargetField
from backend.data_generation.schemas import ProfileFieldValue
from backend.llm.client import OllamaClient

_SYSTEM_PROMPT: str = (
    "You are a strict information extraction assistant. "
    "Extract only explicitly supported field values from the user text. "
    "Return ONLY valid JSON object text without markdown, prose, or comments."
)


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
        f"User text:\n{user_text}"
    )
    raw_completion: str = await client.generate_json_completion(
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )
    return _parse_and_coerce_extraction(raw_completion, target_fields)


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
