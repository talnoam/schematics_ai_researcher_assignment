"""Unit tests for LLM-based text-to-field extraction."""

from unittest.mock import AsyncMock, patch

import pytest

from backend.data_generation.enums import (
    AgeBand,
    CreditLineBand,
    LoanPrimaryPurpose,
    PropertyUse,
    TargetField,
)
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


@pytest.mark.asyncio
async def test_extract_fields_from_text_includes_few_shot_examples_in_prompt() -> None:
    """Verify extractor prompt contains guidance examples for strict extraction."""
    mocked_completion = AsyncMock(return_value="{}")
    with patch(
        "backend.llm.client.OllamaClient.generate_json_completion",
        new=mocked_completion,
    ):
        extracted_fields = await extract_fields_from_text(
            user_text="I have a stable job but prefer not to share salary.",
            target_fields=[TargetField.ANNUAL_INCOME_BAND],
        )

    assert extracted_fields == {}
    assert mocked_completion.await_count == 1
    awaited_kwargs = mocked_completion.await_args.kwargs
    user_prompt = awaited_kwargs["user_prompt"]
    assert "Few-shot examples:" in user_prompt
    assert 'Expected JSON: {"annual_income_band":"50k_to_100k"}' in user_prompt
    assert 'Expected JSON: {}' in user_prompt
    assert '"loan_primary_purpose":"purchase"' in user_prompt


@pytest.mark.asyncio
async def test_extract_fields_from_text_rule_based_override_for_age_and_credit_line() -> None:
    """Verify deterministic parsing corrects explicit age and credit line evidence."""
    mocked_completion = '{"credit_line_band":"above_100k","age_band":"30_to_44"}'
    with patch(
        "backend.llm.client.OllamaClient.generate_json_completion",
        new=AsyncMock(return_value=mocked_completion),
    ):
        extracted_fields = await extract_fields_from_text(
            user_text="I'm 27 years old and my available credit lines are about $8,000.",
            target_fields=[TargetField.CREDIT_LINE_BAND, TargetField.AGE_BAND],
        )

    assert extracted_fields[TargetField.CREDIT_LINE_BAND] == CreditLineBand.UNDER_10K
    assert extracted_fields[TargetField.AGE_BAND] == AgeBand.UNDER_30


@pytest.mark.asyncio
async def test_extract_fields_from_text_rule_based_extracts_boolean_flags() -> None:
    """Verify deterministic parsing extracts explicit boolean statements reliably."""
    mocked_completion = "{}"
    with patch(
        "backend.llm.client.OllamaClient.generate_json_completion",
        new=AsyncMock(return_value=mocked_completion),
    ):
        extracted_fields = await extract_fields_from_text(
            user_text="I am not a military veteran and I currently have a mortgage.",
            target_fields=[TargetField.MILITARY_VETERAN, TargetField.CURRENTLY_HAVE_MORTGAGE],
        )

    assert extracted_fields[TargetField.MILITARY_VETERAN] is False
    assert extracted_fields[TargetField.CURRENTLY_HAVE_MORTGAGE] is True


@pytest.mark.asyncio
async def test_extract_fields_from_text_rule_based_extracts_purchase_intent() -> None:
    """Verify deterministic parsing extracts purchase intent from explicit phrasing."""
    mocked_completion = "{}"
    with patch(
        "backend.llm.client.OllamaClient.generate_json_completion",
        new=AsyncMock(return_value=mocked_completion),
    ):
        extracted_fields = await extract_fields_from_text(
            user_text="I am looking to purchase a condo to live in.",
            target_fields=[TargetField.LOAN_PRIMARY_PURPOSE, TargetField.PROPERTY_TYPE],
        )

    assert extracted_fields[TargetField.LOAN_PRIMARY_PURPOSE] == LoanPrimaryPurpose.PURCHASE
