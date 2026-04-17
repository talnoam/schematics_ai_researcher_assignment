"""Validate generated mock data and visualize cohort-level feature distributions."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from loguru import logger
from plotly.subplots import make_subplots

from backend.data_generation.config import (
    DEFAULT_COHORT_FEATURE_DISTRIBUTIONS_DASHBOARD_PATH,
    EXCLUDED_FEATURE_COLUMN_KEYWORDS,
    DEFAULT_MOCK_USERS_OUTPUT_PATH,
    VISUALIZATION_SUBPLOT_COLUMNS,
)


def _load_mock_users(csv_path: Path) -> pd.DataFrame:
    """Load generated mock users from CSV."""
    if not csv_path.exists():
        msg = f"Mock user CSV was not found: {csv_path}"
        raise FileNotFoundError(msg)
    return pd.read_csv(csv_path)


def _is_categorical_or_boolean(series: pd.Series) -> bool:
    """Return whether a dataframe column should be treated as categorical."""
    if pd.api.types.is_bool_dtype(series):
        return True
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        return True
    if isinstance(series.dtype, pd.CategoricalDtype):
        return True
    return False


def _feature_name_pretty(column_name: str) -> str:
    """Convert a snake_case column name to a presentation label."""
    return column_name.replace("_", " ").title()


def _is_excluded_feature(column_name: str) -> bool:
    """Return whether a column should be excluded from distribution charts."""
    normalized_column_name: str = column_name.lower()
    if normalized_column_name == "cohort_name":
        return True
    return any(keyword in normalized_column_name for keyword in EXCLUDED_FEATURE_COLUMN_KEYWORDS)


def _identify_feature_columns(mock_users: pd.DataFrame) -> list[str]:
    """Identify chartable categorical and boolean columns from generated data."""
    feature_columns: list[str] = []
    for column_name in mock_users.columns:
        if _is_excluded_feature(column_name):
            continue
        series: pd.Series = mock_users[column_name]
        if _is_categorical_or_boolean(series):
            feature_columns.append(column_name)
    feature_columns.sort()
    return feature_columns


def _build_normalized_distribution_table(mock_users: pd.DataFrame, feature_column: str) -> pd.DataFrame:
    """Build within-cohort normalized category percentages for one feature."""
    working_table: pd.DataFrame = mock_users[["cohort_name", feature_column]].copy()
    working_table[feature_column] = working_table[feature_column].fillna("missing").astype(str)
    raw_counts: pd.DataFrame = (
        working_table.groupby(["cohort_name", feature_column], dropna=False)
        .size()
        .rename("count")
        .reset_index()
    )
    raw_counts["percentage"] = (
        raw_counts["count"] / raw_counts.groupby("cohort_name")["count"].transform("sum")
    ) * 100.0
    return raw_counts


def _build_dashboard_figure(
    mock_users: pd.DataFrame,
    feature_columns: list[str],
) -> go.Figure:
    """Create a multi-panel dashboard comparing feature distributions by cohort."""
    subplot_columns: int = VISUALIZATION_SUBPLOT_COLUMNS
    subplot_rows: int = max(1, math.ceil(len(feature_columns) / subplot_columns))
    subplot_titles: list[str] = [_feature_name_pretty(feature_column) for feature_column in feature_columns]

    figure = make_subplots(
        rows=subplot_rows,
        cols=subplot_columns,
        subplot_titles=subplot_titles,
        vertical_spacing=0.1,
        horizontal_spacing=0.08,
    )
    cohort_names: list[str] = sorted(mock_users["cohort_name"].astype(str).unique().tolist())

    for feature_index, feature_column in enumerate(feature_columns):
        row_index: int = (feature_index // subplot_columns) + 1
        col_index: int = (feature_index % subplot_columns) + 1
        distribution_table: pd.DataFrame = _build_normalized_distribution_table(mock_users, feature_column)
        category_values: list[str] = sorted(distribution_table[feature_column].astype(str).unique().tolist())
        for cohort_name in cohort_names:
            cohort_slice = distribution_table[distribution_table["cohort_name"] == cohort_name]
            percentage_by_category: dict[str, float] = {
                str(category): float(percentage)
                for category, percentage in zip(
                    cohort_slice[feature_column].tolist(),
                    cohort_slice["percentage"].tolist(),
                    strict=True,
                )
            }
            ordered_percentages: list[float] = [
                percentage_by_category.get(category_value, 0.0) for category_value in category_values
            ]
            figure.add_trace(
                go.Bar(
                    x=category_values,
                    y=ordered_percentages,
                    name=cohort_name,
                    showlegend=feature_index == 0,
                    hovertemplate=(
                        "Cohort: %{fullData.name}<br>"
                        "Category: %{x}<br>"
                        "Percentage: %{y:.2f}%<extra></extra>"
                    ),
                ),
                row=row_index,
                col=col_index,
            )
        figure.update_xaxes(title_text=_feature_name_pretty(feature_column), row=row_index, col=col_index)
        figure.update_yaxes(
            title_text="Percentage Within Cohort (%)",
            range=[0.0, 100.0],
            row=row_index,
            col=col_index,
        )

    figure.update_layout(
        title=(
            "Cohort-Level Categorical Feature Distributions "
            "(Normalized Percentages Within Each Cohort)"
        ),
        barmode="group",
        template="plotly_white",
        legend_title_text="Cohort",
        height=350 * subplot_rows,
        width=1450,
    )
    return figure


def _save_dashboard(figure: go.Figure, output_path: Path) -> None:
    """Save the interactive dashboard figure to a single HTML report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pio.write_html(
        figure,
        file=str(output_path),
        include_plotlyjs="cdn",
        full_html=True,
        auto_open=False,
    )


def main() -> None:
    """Validate synthetic generation quality and write visual cohort report."""
    mock_users: pd.DataFrame = _load_mock_users(DEFAULT_MOCK_USERS_OUTPUT_PATH)
    feature_columns: list[str] = _identify_feature_columns(mock_users)
    if not feature_columns:
        msg = "No categorical/boolean feature columns were identified for visualization."
        raise ValueError(msg)

    logger.info(
        "Identified feature columns for cohort distribution validation",
        feature_count=len(feature_columns),
        feature_columns=feature_columns,
    )
    dashboard_figure: go.Figure = _build_dashboard_figure(
        mock_users=mock_users,
        feature_columns=feature_columns,
    )
    _save_dashboard(
        figure=dashboard_figure,
        output_path=DEFAULT_COHORT_FEATURE_DISTRIBUTIONS_DASHBOARD_PATH,
    )
    logger.info(
        "Saved cohort distribution dashboard",
        output_path=str(DEFAULT_COHORT_FEATURE_DISTRIBUTIONS_DASHBOARD_PATH),
    )


if __name__ == "__main__":
    main()
