"""Configuration values for data generation components."""

from pathlib import Path

from backend.data_generation.enums import PropertyValueBand

PROBABILITY_SUM_TOLERANCE: float = 1e-6
DEFAULT_COHORT_DEFINITIONS_PATH: Path = Path("data/cohorts/cohort_definitions.yaml")

LATENT_FACTOR_MEAN: float = 0.5
LATENT_FACTOR_STD: float = 0.2

AFFLUENCE_SHIFT_STRENGTH: float = 1.8
RISK_SHIFT_STRENGTH: float = 1.2
AGE_MORTGAGE_REDUCTION_WEIGHT: float = 0.2

DEFAULT_COHORT_SAMPLING_WEIGHTS: dict[str, float] = {
    "Tech Veterans": 0.4,
    "Middle-aged Suburban Families": 0.35,
    "Young Urban Renters": 0.25,
}
COHORT_WEIGHT_SUM_TOLERANCE: float = 1e-6

ZIPCODE_PREMIUM_POOL: tuple[str, ...] = ("94027", "90210", "10007", "94301")
ZIPCODE_UPPER_POOL: tuple[str, ...] = ("98039", "94105", "10580", "02108")
ZIPCODE_MID_POOL: tuple[str, ...] = ("75024", "78704", "30327", "60614")
ZIPCODE_ENTRY_POOL: tuple[str, ...] = ("48201", "85009", "44105", "19134")

ZIPCODE_BY_PROPERTY_VALUE_BAND: dict[PropertyValueBand, tuple[str, ...]] = {
    PropertyValueBand.UNDER_300K: ZIPCODE_ENTRY_POOL,
    PropertyValueBand.FROM_300K_TO_600K: ZIPCODE_MID_POOL,
    PropertyValueBand.FROM_600K_TO_1M: ZIPCODE_UPPER_POOL,
    PropertyValueBand.ABOVE_1M: ZIPCODE_PREMIUM_POOL,
}
PREMIUM_AFFLUENCE_THRESHOLD: float = 0.85

DEFAULT_MOCK_USER_COUNT: int = 5000
DEFAULT_MOCK_USERS_OUTPUT_PATH: Path = Path("data/generated/mock_users.csv")
DEFAULT_INCOME_BY_COHORT_PLOT_PATH: Path = Path("data/generated/income_by_cohort.html")
DEFAULT_INCOME_BY_COHORT_STACKED_PLOT_PATH: Path = Path("data/generated/income_by_cohort_stacked.html")
DEFAULT_VISUALIZATIONS_DIR: Path = Path("data/visualizations")
DEFAULT_COHORT_FEATURE_DISTRIBUTIONS_DASHBOARD_PATH: Path = (
    DEFAULT_VISUALIZATIONS_DIR / "cohort_feature_distributions_dashboard.html"
)
VISUALIZATION_SUBPLOT_COLUMNS: int = 2
EXCLUDED_FEATURE_COLUMN_KEYWORDS: tuple[str, ...] = ("id", "token", "session")
