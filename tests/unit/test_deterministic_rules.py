"""Unit tests for deterministic profile inference rules."""

from backend.core_logic.deterministic_rules import DeterministicRulesEngine, PartialUserProfile
from backend.data_generation.enums import LoanPrimaryPurpose, TargetField


def test_apply_rules_infers_mortgage_for_refinance_when_unknown() -> None:
    """Verify refinance purpose deterministically infers mortgage ownership."""
    engine = DeterministicRulesEngine()
    profile = PartialUserProfile(
        loan_primary_purpose=LoanPrimaryPurpose.REFINANCE,
        currently_have_mortgage=None,
    )

    result = engine.apply_rules(profile)

    assert result.updated_profile.currently_have_mortgage is True
    assert len(result.inferred_fields) == 1
    assert result.inferred_fields[0].field_name == TargetField.CURRENTLY_HAVE_MORTGAGE
    assert result.inferred_fields[0].confidence == 1.0


def test_apply_rules_keeps_existing_mortgage_value_when_known() -> None:
    """Verify known mortgage state is preserved without extra inferences."""
    engine = DeterministicRulesEngine()
    profile = PartialUserProfile(
        loan_primary_purpose=LoanPrimaryPurpose.REFINANCE,
        currently_have_mortgage=True,
    )

    result = engine.apply_rules(profile)

    assert result.updated_profile.currently_have_mortgage is True
    assert result.inferred_fields == []


def test_apply_rules_returns_no_inference_without_deterministic_trigger() -> None:
    """Verify no deterministic inference is created without a guaranteed rule."""
    engine = DeterministicRulesEngine()
    profile = PartialUserProfile(
        loan_primary_purpose=LoanPrimaryPurpose.PURCHASE,
        currently_have_mortgage=None,
    )

    result = engine.apply_rules(profile)

    assert result.updated_profile.currently_have_mortgage is None
    assert result.inferred_fields == []
