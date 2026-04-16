"""Enumerations for synthetic profile field domains."""

from enum import StrEnum


class CreditScoreRate(StrEnum):
    """Represent credit score ranges used for synthetic users."""

    POOR = "poor"
    FAIR = "fair"
    GOOD = "good"
    VERY_GOOD = "very_good"
    EXCELLENT = "excellent"


class LoanPrimaryPurpose(StrEnum):
    """Represent the primary purpose of a loan."""

    PURCHASE = "purchase"
    REFINANCE = "refinance"
    HOME_IMPROVEMENT = "home_improvement"
    DEBT_CONSOLIDATION = "debt_consolidation"
    INVESTMENT = "investment"


class PropertyType(StrEnum):
    """Represent the property type for a borrowing scenario."""

    SINGLE_FAMILY = "single_family"
    CONDO = "condo"
    TOWNHOUSE = "townhouse"
    MULTI_FAMILY = "multi_family"
    MOBILE_HOME = "mobile_home"


class PropertyUse(StrEnum):
    """Represent occupancy intent for the property."""

    PRIMARY_RESIDENCE = "primary_residence"
    SECOND_HOME = "second_home"
    INVESTMENT = "investment"


class AnnualIncomeBand(StrEnum):
    """Represent annual household income buckets."""

    UNDER_50K = "under_50k"
    FROM_50K_TO_100K = "50k_to_100k"
    FROM_100K_TO_200K = "100k_to_200k"
    ABOVE_200K = "above_200k"


class PropertyValueBand(StrEnum):
    """Represent property valuation buckets."""

    UNDER_300K = "under_300k"
    FROM_300K_TO_600K = "300k_to_600k"
    FROM_600K_TO_1M = "600k_to_1m"
    ABOVE_1M = "above_1m"


class CreditLineBand(StrEnum):
    """Represent available revolving credit line buckets."""

    UNDER_10K = "under_10k"
    FROM_10K_TO_50K = "10k_to_50k"
    FROM_50K_TO_100K = "50k_to_100k"
    ABOVE_100K = "above_100k"


class AgeBand(StrEnum):
    """Represent age group buckets used by cohort priors."""

    UNDER_30 = "under_30"
    FROM_30_TO_44 = "30_to_44"
    FROM_45_TO_59 = "45_to_59"
    FROM_60_PLUS = "60_plus"


class TargetField(StrEnum):
    """Represent all target fields for adaptive profiling."""

    CREDIT_SCORE_RATE = "credit_score_rate"
    LOAN_PRIMARY_PURPOSE = "loan_primary_purpose"
    PROPERTY_TYPE = "property_type"
    PROPERTY_USE = "property_use"
    ANNUAL_INCOME_BAND = "annual_income_band"
    PROPERTY_VALUE_BAND = "property_value_band"
    CREDIT_LINE_BAND = "credit_line_band"
    AGE_BAND = "age_band"
    CURRENTLY_HAVE_MORTGAGE = "currently_have_mortgage"
    MILITARY_VETERAN = "military_veteran"
