"""Unit tests for conversational question generation via LLM."""

from unittest.mock import AsyncMock, patch

import pytest

from backend.core_logic.deterministic_rules import PartialUserProfile
from backend.core_logic.question_bank import QUESTION_BANK
from backend.data_generation.enums import TargetField
from backend.llm.generator import generate_conversational_question


@pytest.mark.asyncio
async def test_generate_conversational_question_returns_dynamic_question_from_llm() -> None:
    """Verify dynamic question text is returned when LLM generation succeeds."""
    mocked_question: str = "Could you share your current zipcode so I can tailor the options?"
    with patch(
        "backend.llm.client.OllamaClient.generate_text_completion",
        new=AsyncMock(return_value=mocked_question),
    ):
        generated_question = await generate_conversational_question(
            target_field=TargetField.ZIPCODE,
            profile=PartialUserProfile(),
        )

    assert generated_question == mocked_question


@pytest.mark.asyncio
async def test_generate_conversational_question_falls_back_to_static_question_on_failure() -> None:
    """Verify static question fallback is used when LLM generation fails."""
    with patch(
        "backend.llm.client.OllamaClient.generate_text_completion",
        new=AsyncMock(side_effect=TimeoutError("simulated timeout")),
    ):
        generated_question = await generate_conversational_question(
            target_field=TargetField.PROPERTY_USE,
            profile=PartialUserProfile(),
        )

    assert generated_question == QUESTION_BANK[TargetField.PROPERTY_USE].question_text
