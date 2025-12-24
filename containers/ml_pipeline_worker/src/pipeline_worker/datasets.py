"""Shared helpers for working with dataset files."""
from __future__ import annotations

from pathlib import Path


def resolve_dataset_extension(dataset_path: Path, data_format: str | None) -> str:
    """Return the file extension for a dataset, honoring explicit format overrides."""
    if data_format:
        normalized = data_format.strip()
        if normalized:
            normalized = normalized.lstrip(".")
            if normalized:
                return f".{normalized}"
            return ""
    suffix = dataset_path.suffix
    if suffix:
        return suffix
    raise ValueError("Unable to determine dataset format automatically. Provide --data-format.")
