"""LLM-powered conversational question generation with safe fallback behavior."""

from __future__ import annotations

from loguru import logger

from backend.core_logic.deterministic_rules import PartialUserProfile
from backend.core_logic.question_bank import QUESTION_BANK
from backend.data_generation.enums import TargetField
from backend.llm.client import OllamaClient

_SYSTEM_PROMPT: str = (
    "You are a friendly, professional AI assistant helping a user complete a loan/property profile. "
    "Ask exactly one polite conversational question to gather one missing field. "
    "Output ONLY the question text. "
    "Do not output introductions, JSON, markdown, lists, labels, or explanation."
)


async def generate_conversational_question(
    target_field: TargetField,
    profile: PartialUserProfile,
) -> str:
    """Generate a contextual conversational question for one target field.

    Args:
        target_field: Field that needs to be asked next.
        profile: Current known user profile context.

    Returns:
        A single conversational question string.
    """
    context_text: str = _format_known_profile_context(profile)
    user_prompt: str = (
        f"Known user profile context: {context_text}\n"
        f"Missing target field to ask about: {target_field.value}\n"
        "Ask one concise, polite, conversational question that helps determine this field."
    )
    client = OllamaClient()
    fallback_question: str = QUESTION_BANK[target_field].question_text
    try:
        generated_question: str = await client.generate_text_completion(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
    except Exception as error:  # noqa: BLE001
        logger.warning(
            "Failed to generate conversational question; using fallback",
            target_field=target_field.value,
            error=str(error),
        )
        return fallback_question

    cleaned_question: str = _sanitize_generated_question(generated_question)
    if not cleaned_question:
        logger.warning(
            "Generated conversational question was empty; using fallback",
            target_field=target_field.value,
        )
        return fallback_question
    return cleaned_question


def _format_known_profile_context(profile: PartialUserProfile) -> str:
    """Render non-null profile fields as natural language for prompting."""
    serialized_profile: dict[str, object] = profile.model_dump(mode="json")
    known_field_descriptions: list[str] = []
    for field_name, field_value in serialized_profile.items():
        if field_value is None:
            continue
        label: str = field_name.replace("_", " ")
        formatted_value: str = str(field_value).replace("_", " ")
        known_field_descriptions.append(f"{label} is {formatted_value}")

    if not known_field_descriptions:
        return "No prior profile fields are known yet."
    return "; ".join(known_field_descriptions) + "."


def _sanitize_generated_question(generated_question: str) -> str:
    """Trim common wrapper noise while preserving the core question."""
    cleaned_question: str = generated_question.strip().strip('"').strip("'").strip()
    cleaned_question = cleaned_question.replace("\n", " ").strip()
    return cleaned_question
