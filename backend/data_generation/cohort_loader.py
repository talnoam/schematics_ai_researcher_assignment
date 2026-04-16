"""Load cohort definitions from YAML into strongly typed schemas."""

from __future__ import annotations

from pathlib import Path

import yaml
from loguru import logger
from pydantic import ValidationError

from backend.data_generation.config import DEFAULT_COHORT_DEFINITIONS_PATH
from backend.data_generation.schemas import CohortCatalog, CohortDefinition


def _read_yaml_mapping(file_path: Path) -> dict[str, object]:
    """Read a YAML file and return its top-level mapping."""
    yaml_text: str = file_path.read_text(encoding="utf-8")
    raw_data: object = yaml.safe_load(yaml_text)

    if raw_data is None:
        return {}
    if not isinstance(raw_data, dict):
        msg = f"Expected YAML mapping at top level in {file_path}."
        raise ValueError(msg)
    return raw_data


def load_cohort_catalog(file_path: Path | None = None) -> CohortCatalog:
    """Load and validate the complete cohort catalog from disk."""
    resolved_path: Path = file_path if file_path is not None else DEFAULT_COHORT_DEFINITIONS_PATH
    if not resolved_path.exists():
        logger.error("Cohort definition file missing", file_path=str(resolved_path))
        msg = f"Cohort definition file does not exist: {resolved_path}"
        raise FileNotFoundError(msg)

    raw_mapping: dict[str, object] = _read_yaml_mapping(resolved_path)

    try:
        catalog: CohortCatalog = CohortCatalog.model_validate(raw_mapping)
    except ValidationError as validation_error:
        logger.error(
            "Invalid cohort definitions",
            file_path=str(resolved_path),
            error_count=validation_error.error_count(),
        )
        raise

    logger.info(
        "Loaded cohort definitions",
        file_path=str(resolved_path),
        cohort_count=len(catalog.cohorts),
    )
    return catalog


def load_cohort_definitions(file_path: Path | None = None) -> list[CohortDefinition]:
    """Load and return the list of validated cohort definitions."""
    catalog: CohortCatalog = load_cohort_catalog(file_path=file_path)
    return catalog.cohorts
