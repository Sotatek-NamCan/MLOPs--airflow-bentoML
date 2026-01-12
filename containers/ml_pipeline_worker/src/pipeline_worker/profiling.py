"""Profiling and visualization utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

import pandas as pd
from pandas.api.types import is_numeric_dtype

from .config_validation import validate_config_keys

def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def build_profile_summary(df: pd.DataFrame, config: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    """Return descriptive statistics for each feature in the dataframe."""
    config = config or {}
    validate_config_keys(
        config,
        {"top_value_count", "include_percentiles"},
        context="profile_config",
    )
    top_value_count = int(config.get("top_value_count", 5) or 5)
    percentiles = config.get("include_percentiles", [0.25, 0.5, 0.75])

    summary: Dict[str, Any] = {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": {},
    }

    for column in df.columns:
        series = df[column]
        info: Dict[str, Any] = {
            "dtype": str(series.dtype),
            "non_null_count": int(series.notna().sum()),
            "null_count": int(series.isna().sum()),
        }
        info["null_ratio"] = float(info["null_count"]) / summary["row_count"] if summary["row_count"] else 0.0
        info["unique_values"] = int(series.nunique(dropna=True))

        if is_numeric_dtype(series):
            stats = {
                "min": _as_float(series.min()),
                "max": _as_float(series.max()),
                "mean": _as_float(series.mean()),
                "median": _as_float(series.median()),
                "std": _as_float(series.std(ddof=0)),
                "skew": _as_float(series.skew()),
            }
            percentile_stats = {
                str(p): _as_float(series.quantile(p)) for p in percentiles if isinstance(p, (int, float))
            }
            info["statistics"] = stats
            info["percentiles"] = percentile_stats
        else:
            top_values = series.value_counts(dropna=True).head(top_value_count)
            info["top_values"] = [
                {"value": str(index), "count": int(count)} for index, count in top_values.items()
            ]
        summary["columns"][column] = info

    return summary


def render_visualizations(
    df: pd.DataFrame,
    output_dir: Path,
    config: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Create distribution charts and return metadata about generated assets."""
    import matplotlib  # Imported lazily to avoid heavy dependency when unused

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config = config or {}
    validate_config_keys(
        config,
        {"max_numeric_charts", "max_categorical_charts", "top_value_count", "plot_format"},
        context="visualization_config",
    )
    numeric_limit = int(config.get("max_numeric_charts", 10) or 10)
    categorical_limit = int(config.get("max_categorical_charts", 10) or 10)
    top_value_count = int(config.get("top_value_count", 5) or 5)
    plot_format = (config.get("plot_format") or "png").lower()
    if plot_format not in {"png", "jpg", "jpeg"}:
        plot_format = "png"

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata: Dict[str, Any] = {"charts": []}

    numeric_columns = [col for col in df.columns if is_numeric_dtype(df[col])]
    categorical_columns = [col for col in df.columns if not is_numeric_dtype(df[col])]

    def _save(fig, filename: str, column: str, chart_type: str, extra: Dict[str, Any]) -> None:
        path = output_dir / filename
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        payload = {"column": column, "type": chart_type, "file": path.name}
        payload.update(extra)
        metadata["charts"].append(payload)

    for column in numeric_columns[: numeric_limit or None]:
        series = df[column].dropna()
        if series.empty:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].hist(series, bins=min(30, max(5, series.nunique())), color="#4a90e2", alpha=0.8)
        axes[0].set_title(f"{column} distribution")
        axes[0].set_xlabel(column)
        axes[0].set_ylabel("Frequency")

        axes[1].boxplot(series, vert=True)
        axes[1].set_title(f"{column} boxplot")
        axes[1].set_ylabel(column)

        filename = f"{column}_distribution.{plot_format}"
        _save(
            fig,
            filename,
            column,
            "numeric_distribution",
            {"skew": _as_float(series.skew())},
        )

    for column in categorical_columns[: categorical_limit or None]:
        series = df[column].fillna("missing")
        counts = series.value_counts().head(top_value_count)
        if counts.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 4))
        counts.plot(kind="bar", ax=ax, color="#50e3c2")
        ax.set_title(f"{column} top categories")
        ax.set_ylabel("Count")
        ax.set_xlabel(column)

        filename = f"{column}_categories.{plot_format}"
        _save(
            fig,
            filename,
            column,
            "categorical_distribution",
            {"categories": counts.index.tolist()},
        )

    metadata["total_charts"] = len(metadata["charts"])
    metadata["numeric_columns_plotted"] = min(len(numeric_columns), numeric_limit or len(numeric_columns))
    metadata["categorical_columns_plotted"] = min(
        len(categorical_columns), categorical_limit or len(categorical_columns)
    )
    return metadata
