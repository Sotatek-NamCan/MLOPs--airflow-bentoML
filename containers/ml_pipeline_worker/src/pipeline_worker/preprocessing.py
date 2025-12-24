"""Data cleaning utilities for pipeline tasks."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


@dataclass
class CleaningConfig:
    drop_columns: List[str] = field(default_factory=list)
    deduplicate: bool = True
    column_order: List[str] = field(default_factory=list)
    missing_values: Dict[str, Any] = field(default_factory=dict)
    outliers: Dict[str, Any] = field(default_factory=dict)
    transformations: List[Mapping[str, Any]] = field(default_factory=list)

    @classmethod
    def from_raw(cls, raw: Mapping[str, Any] | None) -> "CleaningConfig":
        raw = raw or {}
        return cls(
            drop_columns=_as_list(raw.get("drop_columns")),
            deduplicate=bool(raw.get("deduplicate", True)),
            column_order=_as_list(raw.get("column_order")),
            missing_values=dict(raw.get("missing_values") or {}),
            outliers=dict(raw.get("outliers") or {}),
            transformations=list(raw.get("transformations") or []),
        )


def _fill_missing_values(df: pd.DataFrame, config: CleaningConfig) -> Dict[str, Any]:
    settings = config.missing_values or {}
    numeric_strategy = (settings.get("numeric_strategy") or "median").lower()
    categorical_strategy = (settings.get("categorical_strategy") or "mode").lower()
    categorical_fill_value = settings.get("categorical_fill_value", "missing")
    column_fill_values = settings.get("column_fill_values") or {}

    applied: Dict[str, str] = {}
    for column, value in column_fill_values.items():
        if column in df.columns:
            df[column] = df[column].fillna(value)
            applied[column] = "custom_value"

    numeric_columns = [col for col in df.columns if is_numeric_dtype(df[col])]
    for column in numeric_columns:
        if column in applied:
            continue
        series = df[column]
        if series.isna().sum() == 0:
            continue
        fill_value = None
        if numeric_strategy == "mean":
            fill_value = series.mean()
        elif numeric_strategy == "zero":
            fill_value = 0.0
        else:  # default median
            fill_value = series.median()
        if pd.isna(fill_value):
            fill_value = 0.0
        df[column] = series.fillna(fill_value)
        applied[column] = numeric_strategy

    categorical_columns = [col for col in df.columns if not is_numeric_dtype(df[col])]
    for column in categorical_columns:
        if column in applied:
            continue
        series = df[column]
        if series.isna().sum() == 0:
            continue
        fill_value = categorical_fill_value
        if categorical_strategy == "mode":
            modes = series.mode(dropna=True)
            if not modes.empty:
                fill_value = modes.iloc[0]
        df[column] = series.fillna(fill_value)
        applied[column] = categorical_strategy

    return {
        "strategies": applied,
        "numeric_strategy": numeric_strategy,
        "categorical_strategy": categorical_strategy,
    }


def _handle_outliers(df: pd.DataFrame, config: CleaningConfig) -> List[Dict[str, Any]]:
    settings = config.outliers or {}
    if not settings:
        return []
    method = (settings.get("method") or "iqr").lower()
    columns = settings.get("columns")
    iqr_factor = float(settings.get("iqr_factor", 1.5))
    if columns:
        target_columns = [col for col in columns if col in df.columns]
    else:
        target_columns = [col for col in df.columns if is_numeric_dtype(df[col])]

    changes: List[Dict[str, Any]] = []
    if method != "iqr":
        return changes

    for column in target_columns:
        series = df[column]
        if not is_numeric_dtype(series):
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if not np.isfinite(iqr) or iqr == 0:
            continue
        lower = q1 - iqr_factor * iqr
        upper = q3 + iqr_factor * iqr
        clipped = series.clip(lower, upper)
        changed = int((series != clipped).sum())
        df[column] = clipped
        changes.append(
            {
                "column": column,
                "lower_bound": float(lower),
                "upper_bound": float(upper),
                "adjusted_values": changed,
            }
        )
    return changes


def _drop_columns(df: pd.DataFrame, config: CleaningConfig) -> List[str]:
    if not config.drop_columns:
        return []
    targets = [col for col in config.drop_columns if col in df.columns]
    if not targets:
        return []
    df.drop(columns=targets, inplace=True)
    return targets


def _reorder_columns(df: pd.DataFrame, config: CleaningConfig) -> tuple[pd.DataFrame, List[str]]:
    if not config.column_order:
        return df, []
    ordered = [col for col in config.column_order if col in df.columns]
    if not ordered:
        return df, []
    remainder = [col for col in df.columns if col not in ordered]
    new_order = ordered + remainder
    reordered = df.loc[:, new_order].copy()
    reordered.reset_index(drop=True, inplace=True)
    return reordered, ordered


def _apply_transformations(df: pd.DataFrame, config: CleaningConfig) -> List[Dict[str, Any]]:
    applied: List[Dict[str, Any]] = []
    for transform in config.transformations:
        t_type = (transform.get("type") or "").lower()
        columns = _as_list(transform.get("columns"))
        if not t_type or not columns:
            continue
        method = (transform.get("method") or "").lower()
        for column in columns:
            if column not in df.columns or not is_numeric_dtype(df[column]):
                continue
            series = df[column].astype(float)
            if t_type in {"standardize", "zscore", "scale"} and method in {"standard", "zscore", ""}:
                mean = series.mean()
                std = series.std(ddof=0)
                if std == 0 or pd.isna(std):
                    continue
                df[column] = (series - mean) / std
                applied.append({"type": "standardize", "column": column})
            elif t_type in {"scale", "minmax"} or method == "minmax":
                min_val = series.min()
                max_val = series.max()
                if max_val == min_val:
                    continue
                df[column] = (series - min_val) / (max_val - min_val)
                applied.append({"type": "minmax", "column": column})
            elif t_type == "log1p":
                shift = float(transform.get("shift", 0))
                adjusted = np.clip(series + shift, a_min=0, a_max=None)
                adjusted = np.log1p(adjusted)
                adjusted = pd.Series(adjusted, index=series.index)
                df[column] = adjusted.replace([-np.inf, np.inf], 0.0)
                applied.append({"type": "log1p", "column": column})
            elif t_type == "power":
                exponent = float(transform.get("exponent", 1.0))
                powered = np.power(series, exponent)
                df[column] = pd.Series(powered, index=series.index)
                applied.append({"type": "power", "column": column, "exponent": exponent})
    return applied


def clean_dataframe(df: pd.DataFrame, config: CleaningConfig) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Apply cleaning rules to the dataframe and return the cleaned frame with a summary."""
    working_df = df.copy()
    summary: Dict[str, Any] = {
        "row_count_before": int(len(df)),
    }

    dropped = _drop_columns(working_df, config)
    if dropped:
        summary["dropped_columns"] = dropped

    missing_summary = _fill_missing_values(working_df, config)
    summary["missing_value_strategies"] = missing_summary

    outlier_summary = _handle_outliers(working_df, config)
    if outlier_summary:
        summary["outlier_adjustments"] = outlier_summary

    if config.deduplicate:
        before = len(working_df)
        working_df = working_df.drop_duplicates().reset_index(drop=True)
        summary["deduplicated_rows"] = int(before - len(working_df))

    working_df, column_order = _reorder_columns(working_df, config)
    if column_order:
        summary["column_order"] = column_order

    transformations = _apply_transformations(working_df, config)
    if transformations:
        summary["transformations"] = transformations

    summary["row_count_after"] = int(len(working_df))
    summary["column_count"] = len(working_df.columns)
    summary["columns"] = list(working_df.columns)
    return working_df, summary
