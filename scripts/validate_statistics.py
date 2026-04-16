"""Validate generated mock-data statistics and save summary plots."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
from loguru import logger

from backend.data_generation.config import (
    DEFAULT_INCOME_BY_COHORT_PLOT_PATH,
    DEFAULT_INCOME_BY_COHORT_STACKED_PLOT_PATH,
    DEFAULT_MOCK_USERS_OUTPUT_PATH,
)
from backend.data_generation.enums import AnnualIncomeBand, PropertyValueBand


def _load_mock_users(csv_path: Path) -> pd.DataFrame:
    """Load generated mock users from CSV."""
    if not csv_path.exists():
        msg = f"Mock user CSV was not found: {csv_path}"
        raise FileNotFoundError(msg)
    return pd.read_csv(csv_path)


def _normalize_boolean_column(values: pd.Series) -> pd.Series:
    """Convert a CSV boolean-like column to normalized bool values."""
    normalized = values.astype(str).str.strip().str.lower().map({"true": True, "false": False})
    if normalized.isna().any():
        msg = "Boolean normalization failed for currently_have_mortgage column."
        raise ValueError(msg)
    return normalized.astype(bool)


def _calculate_income_property_metrics(mock_users: pd.DataFrame) -> dict[str, float]:
    """Calculate baseline and conditional high-income probabilities."""
    high_income_mask = mock_users["annual_income_band"] == AnnualIncomeBand.ABOVE_200K.value
    premium_property_mask = mock_users["property_value_band"] == PropertyValueBand.ABOVE_1M.value

    baseline_high_income_probability: float = float(high_income_mask.mean())
    conditional_high_income_probability: float = float(high_income_mask[premium_property_mask].mean())

    return {
        "baseline_high_income_probability": baseline_high_income_probability,
        "conditional_high_income_probability": conditional_high_income_probability,
    }


def _calculate_mortgage_rates_by_age(mock_users: pd.DataFrame) -> dict[str, float]:
    """Calculate mortgage ownership rates for each age band."""
    normalized_mortgage = _normalize_boolean_column(mock_users["currently_have_mortgage"])
    working_table: pd.DataFrame = mock_users.copy()
    working_table["currently_have_mortgage"] = normalized_mortgage
    rates = (
        working_table.groupby("age_band", dropna=False)["currently_have_mortgage"]
        .mean()
        .sort_values(ascending=False)
    )
    return {str(age_band): float(rate) for age_band, rate in rates.items()}


def _save_income_distribution_plots(
    mock_users: pd.DataFrame,
    grouped_plot_path: Path,
    stacked_plot_path: Path,
) -> None:
    """Save cohort-level income distribution plots as HTML files."""
    grouped_plot_path.parent.mkdir(parents=True, exist_ok=True)
    stacked_plot_path.parent.mkdir(parents=True, exist_ok=True)

    grouped_fig = px.histogram(
        mock_users,
        x="annual_income_band",
        color="cohort_name",
        barmode="group",
        title="Income Distribution by Cohort",
    )
    grouped_fig.write_html(grouped_plot_path)

    stacked_fig = px.histogram(
        mock_users,
        x="annual_income_band",
        color="cohort_name",
        barmode="stack",
        title="Income Distribution by Cohort (Stacked)",
    )
    stacked_fig.write_html(stacked_plot_path)


def main() -> None:
    """Validate generated CSV statistics and output interpretability plots."""
    mock_users: pd.DataFrame = _load_mock_users(DEFAULT_MOCK_USERS_OUTPUT_PATH)
    income_metrics: dict[str, float] = _calculate_income_property_metrics(mock_users)
    mortgage_rates: dict[str, float] = _calculate_mortgage_rates_by_age(mock_users)
    _save_income_distribution_plots(
        mock_users,
        grouped_plot_path=DEFAULT_INCOME_BY_COHORT_PLOT_PATH,
        stacked_plot_path=DEFAULT_INCOME_BY_COHORT_STACKED_PLOT_PATH,
    )

    uplift: float = (
        income_metrics["conditional_high_income_probability"]
        - income_metrics["baseline_high_income_probability"]
    )
    logger.info(
        f"Income-vs-property conditional metric:\n"
        f"  Baseline Probability: {income_metrics['baseline_high_income_probability']:.4f}\n"
        f"  Conditional Probability: {income_metrics['conditional_high_income_probability']:.4f}\n"
        f"  Uplift: {uplift:.4f}"
    )
    logger.info(f"Mortgage ownership rates by age group: {mortgage_rates}")
    logger.info("Saved validation plots.")


if __name__ == "__main__":
    main()
