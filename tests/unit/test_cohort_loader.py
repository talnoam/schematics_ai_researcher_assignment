"""Unit tests for cohort YAML loading and validation."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from backend.data_generation.cohort_loader import load_cohort_catalog, load_cohort_definitions
from backend.data_generation.schemas import CohortCatalog


def test_load_cohort_catalog_reads_default_file() -> None:
    """Verify default cohort YAML is parsed into a typed catalog."""
    catalog: CohortCatalog = load_cohort_catalog()
    assert len(catalog.cohorts) == 3


def test_load_cohort_definitions_returns_cohort_list() -> None:
    """Verify helper function returns a list of validated cohorts."""
    cohorts = load_cohort_definitions()
    assert [cohort.cohort_name for cohort in cohorts] == [
        "Tech Veterans",
        "Middle-aged Suburban Families",
        "Young Urban Renters",
    ]


def test_load_cohort_catalog_raises_for_missing_file(tmp_path: Path) -> None:
    """Verify loader raises FileNotFoundError when the YAML path is missing."""
    missing_file_path: Path = tmp_path / "missing.yaml"
    with pytest.raises(FileNotFoundError):
        load_cohort_catalog(file_path=missing_file_path)


def test_load_cohort_catalog_raises_for_invalid_probabilities(tmp_path: Path) -> None:
    """Verify loader raises ValidationError for invalid distribution values."""
    invalid_yaml_path: Path = tmp_path / "invalid_cohorts.yaml"
    invalid_yaml_path.write_text(
        "\n".join(
            [
                "cohorts:",
                "  - cohort_name: Broken Cohort",
                "    description: invalid probability sum",
                "    base_probabilities:",
                "      credit_score_rate:",
                "        poor: 0.8",
                "        fair: 0.8",
                "      loan_primary_purpose: {purchase: 1.0}",
                "      property_type: {single_family: 1.0}",
                "      property_use: {primary_residence: 1.0}",
                "      annual_income_band: {under_50k: 1.0}",
                "      property_value_band: {under_300k: 1.0}",
                "      credit_line_band: {under_10k: 1.0}",
                "      age_band: {under_30: 1.0}",
                "      currently_have_mortgage: {true: 1.0}",
                "      military_veteran: {false: 1.0}",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_cohort_catalog(file_path=invalid_yaml_path)
