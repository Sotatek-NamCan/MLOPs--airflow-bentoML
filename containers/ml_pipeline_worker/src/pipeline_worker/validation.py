"""Great Expectations powered dataset validation helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence

import pandas as pd
from great_expectations.dataset import PandasDataset


class DataValidationError(RuntimeError):
    """Raised when one or more expectations fail."""

    def __init__(
        self,
        failures: Sequence[Mapping[str, Any]],
        *,
        results: Sequence[Mapping[str, Any]],
        summary: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__("Data validation failed.")
        self.failures = list(failures)
        self.results = list(results)
        self.summary = dict(summary or {})


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    return [value]


def _dedup_preserve_order(values: Iterable[Any]) -> List[Any]:
    seen: set[Any] = set()
    ordered: List[Any] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


@dataclass
class ValidationConfig:
    min_row_count: int | None = 1
    max_row_count: int | None = None
    required_columns: List[str] | None = None
    non_null_columns: List[str] | None = None
    unique_columns: List[str] | None = None
    value_ranges: Dict[str, Dict[str, Any]] | None = None
    allowed_values: Dict[str, Sequence[Any]] | None = None

    @classmethod
    def from_raw(cls, raw: Mapping[str, Any] | None, *, target_column: str | None) -> "ValidationConfig":
        raw = raw or {}
        config = cls(
            min_row_count=raw.get("min_row_count", 1),
            max_row_count=raw.get("max_row_count"),
            required_columns=_as_list(raw.get("required_columns")),
            non_null_columns=_as_list(raw.get("non_null_columns")),
            unique_columns=_as_list(raw.get("unique_columns")),
            value_ranges=dict(raw.get("value_ranges") or {}),
            allowed_values=dict(raw.get("allowed_values") or {}),
        )
        if target_column:
            config.required_columns = _dedup_preserve_order(
                [target_column, *(config.required_columns or [])]
            )
            config.non_null_columns = _dedup_preserve_order(
                [target_column, *(config.non_null_columns or [])]
            )
        return config


def validate_dataframe(df: pd.DataFrame, config: ValidationConfig) -> Dict[str, Any]:
    """Run a set of expectations against a dataframe and raise if one fails."""
    dataset = PandasDataset(df.copy())
    results: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    def _record(result: Mapping[str, Any], *, expectation: str) -> None:
        payload = {
            "expectation": expectation,
            "success": bool(result.get("success")),
            "result": result.get("result"),
            "meta": result.get("meta"),
        }
        results.append(payload)
        if not payload["success"]:
            failures.append(payload)

    if config.min_row_count is not None or config.max_row_count is not None:
        result = dataset.expect_table_row_count_to_be_between(
            min_value=config.min_row_count,
            max_value=config.max_row_count,
        )
        _record(result, expectation="table_row_count_between")

    for column in config.required_columns or []:
        _record(
            dataset.expect_column_to_exist(column),
            expectation=f"column_exists::{column}",
        )

    for column in config.non_null_columns or []:
        _record(
            dataset.expect_column_values_to_not_be_null(column),
            expectation=f"column_not_null::{column}",
        )

    for column in config.unique_columns or []:
        _record(
            dataset.expect_column_values_to_be_unique(column),
            expectation=f"column_unique::{column}",
        )

    for column, range_config in (config.value_ranges or {}).items():
        range_config = range_config or {}
        _record(
            dataset.expect_column_values_to_be_between(
                column,
                min_value=range_config.get("min"),
                max_value=range_config.get("max"),
                strict_min=range_config.get("strict_min"),
                strict_max=range_config.get("strict_max"),
            ),
            expectation=f"column_between::{column}",
        )

    for column, allowed in (config.allowed_values or {}).items():
        allowed_list = list(allowed or [])
        if not allowed_list:
            continue
        _record(
            dataset.expect_column_values_to_be_in_set(column, allowed_list),
            expectation=f"column_in_set::{column}",
        )

    summary = {
        "statistics": {
            "evaluated_expectations": len(results),
            "successful_expectations": len(results) - len(failures),
            "unsuccessful_expectations": len(failures),
        },
        "results": results,
    }
    if failures:
        raise DataValidationError(failures, results=results, summary=summary)
    return summary
