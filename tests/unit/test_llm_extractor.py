"""Unit tests for LLM-based text-to-field extraction."""

from unittest.mock import AsyncMock, patch

import pytest

from backend.data_generation.enums import PropertyUse, TargetField
from backend.llm.extractor import extract_fields_from_text


@pytest.mark.asyncio
async def test_extract_fields_from_text_coerces_enums_and_bools() -> None:
    """Verify extractor maps JSON response to typed enum and boolean values."""
    mocked_completion = (
        '{"property_use":"investment","currently_have_mortgage":true,"not_a_field":"ignored"}'
    )
    with patch(
        "backend.llm.client.OllamaClient.generate_json_completion",
        new=AsyncMock(return_value=mocked_completion),
    ):
        extracted_fields = await extract_fields_from_text(
            user_text="I am buying an investment property and I already have a mortgage.",
            target_fields=[
                TargetField.PROPERTY_USE,
                TargetField.CURRENTLY_HAVE_MORTGAGE,
            ],
        )

    assert extracted_fields[TargetField.PROPERTY_USE] == PropertyUse.INVESTMENT
    assert extracted_fields[TargetField.CURRENTLY_HAVE_MORTGAGE] is True
    assert len(extracted_fields) == 2


@pytest.mark.asyncio
async def test_extract_fields_from_text_returns_empty_dict_for_invalid_json() -> None:
    """Verify extractor handles non-JSON model output safely."""
    with patch(
        "backend.llm.client.OllamaClient.generate_json_completion",
        new=AsyncMock(return_value="not-json"),
    ):
        extracted_fields = await extract_fields_from_text(
            user_text="my credit score is around good range",
            target_fields=[TargetField.CREDIT_SCORE_RATE],
        )

    assert extracted_fields == {}
