"""Configuration values for data generation components."""

from pathlib import Path

PROBABILITY_SUM_TOLERANCE: float = 1e-6
DEFAULT_COHORT_DEFINITIONS_PATH: Path = Path("data/cohorts/cohort_definitions.yaml")

LATENT_FACTOR_MEAN: float = 0.5
LATENT_FACTOR_STD: float = 0.2

AFFLUENCE_SHIFT_STRENGTH: float = 1.8
RISK_SHIFT_STRENGTH: float = 1.2
AGE_MORTGAGE_REDUCTION_WEIGHT: float = 0.2
