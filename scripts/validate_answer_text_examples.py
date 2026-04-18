"""Validate answer_text extraction quality against curated example prompts."""

from __future__ import annotations

import argparse
from uuid import UUID

import httpx
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field


class ExampleCase(BaseModel):
    """Represent one answer_text validation case and expected filled fields."""

    model_config = ConfigDict(extra="forbid", strict=True)

    sentence: str
    expected_fields: dict[str, str | bool] = Field(default_factory=dict)


class ValidationResult(BaseModel):
    """Represent one validation run outcome for a single example case."""

    model_config = ConfigDict(extra="forbid", strict=True)

    sentence: str
    passed: bool
    mismatches: list[str] = Field(default_factory=list)
    expected_fields: dict[str, str | bool] = Field(default_factory=dict)
    actual_fields: dict[str, object] = Field(default_factory=dict)


EXAMPLE_CASES: list[ExampleCase] = [
    ExampleCase(
        sentence="I earn about $42,000 a year.",
        expected_fields={"annual_income_band": "under_50k"},
    ),
    ExampleCase(
        sentence="My household income is around $78,000 per year.",
        expected_fields={"annual_income_band": "50k_to_100k"},
    ),
    ExampleCase(
        sentence="I make about 150k annually.",
        expected_fields={"annual_income_band": "100k_to_200k"},
    ),
    ExampleCase(
        sentence="Our annual income is roughly $260,000.",
        expected_fields={"annual_income_band": "above_200k"},
    ),
    ExampleCase(
        sentence="I'm looking to purchase a condo to live in as my primary residence.",
        expected_fields={
            "loan_primary_purpose": "purchase",
            "property_type": "condo",
            "property_use": "primary_residence",
        },
    ),
    ExampleCase(
        sentence="I want to refinance my current home.",
        expected_fields={
            "loan_primary_purpose": "refinance",
            "currently_have_mortgage": True,
        },
    ),
    ExampleCase(
        sentence="This is for a townhouse that will be my second home.",
        expected_fields={
            "property_type": "townhouse",
            "property_use": "second_home",
        },
    ),
    ExampleCase(
        sentence="I'm buying a multi-family property as an investment.",
        expected_fields={
            "loan_primary_purpose": "purchase",
            "property_type": "multi_family",
            "property_use": "investment",
        },
    ),
    ExampleCase(
        sentence="I'm 27 years old and my available credit lines are about $8,000.",
        expected_fields={
            "age_band": "under_30",
            "credit_line_band": "under_10k",
        },
    ),
    ExampleCase(
        sentence="I'm 51 and my revolving credit line is around $65,000.",
        expected_fields={
            "age_band": "45_to_59",
            "credit_line_band": "50k_to_100k",
        },
    ),
    ExampleCase(
        sentence="My credit score is in the very good range.",
        expected_fields={"credit_score_rate": "very_good"},
    ),
    ExampleCase(
        sentence="I am not a military veteran.",
        expected_fields={"military_veteran": False},
    ),
    ExampleCase(
        sentence="I currently have a mortgage.",
        expected_fields={"currently_have_mortgage": True},
    ),
    ExampleCase(
        sentence="My property value is around $720,000.",
        expected_fields={"property_value_band": "600k_to_1m"},
    ),
]


def parse_args() -> argparse.Namespace:
    """Parse CLI options for answer_text validation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Base URL for the backend service.",
    )
    parser.add_argument(
        "--cohort-name",
        default="Middle-aged Suburban Families",
        help="Cohort name to force during /start for stable evaluation context.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=45.0,
        help="HTTP timeout for each request.",
    )
    return parser.parse_args()


def start_session(
    client: httpx.Client,
    base_url: str,
    cohort_name: str,
) -> UUID:
    """Create a new session and return the server-generated session ID."""
    response = client.post(
        f"{base_url}/api/v1/sessions/start",
        json={"cohort_name": cohort_name},
    )
    response.raise_for_status()
    payload = response.json()
    session_id = payload["session_id"]
    if not isinstance(session_id, str):
        msg = "Missing session_id in /start response."
        raise ValueError(msg)
    return UUID(session_id)


def run_case(
    client: httpx.Client,
    base_url: str,
    cohort_name: str,
    case: ExampleCase,
) -> ValidationResult:
    """Run one case through /answer_text and compare expected profile fields."""
    session_id = start_session(client, base_url, cohort_name)
    response = client.post(
        f"{base_url}/api/v1/sessions/{session_id}/answer_text",
        json={"user_text": case.sentence},
    )
    response.raise_for_status()
    payload = response.json()
    current_profile = payload.get("current_profile")
    if not isinstance(current_profile, dict):
        msg = "Missing current_profile in /answer_text response."
        raise ValueError(msg)

    mismatches: list[str] = []
    for field_name, expected_value in case.expected_fields.items():
        actual_value = current_profile.get(field_name)
        if actual_value != expected_value:
            mismatches.append(
                f"{field_name}: expected={expected_value!r}, actual={actual_value!r}"
            )

    return ValidationResult(
        sentence=case.sentence,
        passed=not mismatches,
        mismatches=mismatches,
        expected_fields=case.expected_fields,
        actual_fields=current_profile,
    )


def build_outcome_fields(result: ValidationResult) -> dict[str, object]:
    """Build a focused outcome view for only expected field names."""
    return {
        field_name: result.actual_fields.get(field_name)
        for field_name in result.expected_fields
    }


def build_populated_fields(actual_fields: dict[str, object]) -> dict[str, object]:
    """Build a compact view of non-null fields from a full profile payload."""
    return {
        field_name: field_value
        for field_name, field_value in actual_fields.items()
        if field_value is not None
    }


def main() -> None:
    """Execute all answer_text example cases and report pass/fail summary."""
    args = parse_args()
    results: list[ValidationResult] = []

    logger.info(
        "Starting answer_text example validation",
        base_url=args.base_url,
        cohort_name=args.cohort_name,
        total_cases=len(EXAMPLE_CASES),
    )
    with httpx.Client(timeout=args.timeout_seconds) as client:
        for case_index, case in enumerate(EXAMPLE_CASES, start=1):
            result = run_case(client, args.base_url, args.cohort_name, case)
            results.append(result)
            outcome_fields = build_outcome_fields(result)
            populated_fields = build_populated_fields(result.actual_fields)
            if result.passed:
                logger.info(
                    "PASS [{case_index}/{total_cases}] sentence={sentence!r} "
                    "expected={expected_fields} outcome={outcome_fields}",
                    case_index=case_index,
                    total_cases=len(EXAMPLE_CASES),
                    sentence=case.sentence,
                    expected_fields=result.expected_fields,
                    outcome_fields=outcome_fields,
                )
            else:
                logger.error(
                    "FAIL [{case_index}/{total_cases}] sentence={sentence!r} "
                    "expected={expected_fields} outcome={outcome_fields} "
                    "mismatches={mismatches} populated={populated_fields}",
                    case_index=case_index,
                    total_cases=len(EXAMPLE_CASES),
                    sentence=case.sentence,
                    expected_fields=result.expected_fields,
                    outcome_fields=outcome_fields,
                    mismatches=result.mismatches,
                    populated_fields=populated_fields,
                )

    passed_count = sum(1 for result in results if result.passed)
    failed_count = len(results) - passed_count
    logger.info(
        "Finished answer_text validation",
        passed_cases=passed_count,
        failed_cases=failed_count,
        total_cases=len(results),
    )
    if failed_count > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
