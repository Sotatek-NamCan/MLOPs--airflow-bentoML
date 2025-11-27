"""Reusable ML pipeline helpers for DockerOperator tasks."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pipeline-worker")
except PackageNotFoundError:  # pragma: no cover - fallback when package metadata missing
    __version__ = "0.0.0"

__all__ = ["__version__"]

