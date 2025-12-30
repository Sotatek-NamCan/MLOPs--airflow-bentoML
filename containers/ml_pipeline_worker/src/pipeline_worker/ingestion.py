"""Configurable data ingestion utilities."""
from __future__ import annotations

import os
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple
from urllib.parse import urlparse

import pandas as pd

from pipeline_worker.storage import (
    ObjectStorageConfigurationError,
    ObjectStorageOperationError,
    build_storage_client,
)


def _normalize_extension(extension: str) -> str:
    ext = extension.lower().strip()
    if not ext:
        raise ValueError("File extension cannot be empty.")
    if not ext.startswith("."):
        ext = f".{ext}"
    return ext


def _validate_file_exists(file_path: str | Path) -> Path:
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"No file found at path: {path}")
    return path


class DataIngestor(ABC):
    """Abstract base class for data ingest operations."""

    @abstractmethod
    def ingest(self, file_path: str | Path) -> pd.DataFrame:
        """Ingest a file and return a pandas DataFrame."""
        raise NotImplementedError


class PandasReaderIngestor(DataIngestor):
    """Generic ingest wrapper around a pandas reader callable."""

    def __init__(
        self,
        reader: Callable[..., pd.DataFrame],
        supported_extensions: Iterable[str],
        reader_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._reader = reader
        self._extensions = tuple(_normalize_extension(ext) for ext in supported_extensions)
        self._reader_kwargs = reader_kwargs or {}

    def ingest(self, file_path: str | Path) -> pd.DataFrame:
        path = _validate_file_exists(file_path)
        extension = path.suffix
        if not extension:
            raise ValueError(
                f"Cannot determine file extension for '{path.name}'. "
                f"Expected one of {self._extensions}."
            )

        normalized_ext = _normalize_extension(extension)
        if normalized_ext not in self._extensions:
            raise ValueError(
                f"Unsupported file extension '{normalized_ext}' for {self.__class__.__name__}. "
                f"Supported: {self._extensions}."
            )

        return self._reader(path, **self._reader_kwargs)


class CSVDataIngestor(PandasReaderIngestor):
    def __init__(self) -> None:
        super().__init__(reader=pd.read_csv, supported_extensions=(".csv",))


class TSVDataIngestor(PandasReaderIngestor):
    def __init__(self) -> None:
        super().__init__(
            reader=pd.read_csv,
            supported_extensions=(".tsv",),
            reader_kwargs={"sep": "\t"},
        )


class JSONDataIngestor(PandasReaderIngestor):
    def __init__(self) -> None:
        super().__init__(
            reader=pd.read_json,
            supported_extensions=(".json",),
            reader_kwargs={"orient": "records", "encoding": "utf-8-sig"},
        )


class ExcelDataIngestor(PandasReaderIngestor):
    def __init__(self) -> None:
        super().__init__(reader=pd.read_excel, supported_extensions=(".xlsx", ".xls"))


class ParquetDataIngestor(PandasReaderIngestor):
    def __init__(self) -> None:
        super().__init__(reader=pd.read_parquet, supported_extensions=(".parquet",))


class ZipDataIngestor(DataIngestor):
    """Extract ZIP archive and delegate ingestion to the appropriate innermost file."""

    def __init__(self, extract_dir: Optional[str] = None) -> None:
        self._extract_dir = Path(extract_dir or "extracted_data")

    def ingest(self, file_path: str | Path) -> pd.DataFrame:
        path = _validate_file_exists(file_path)
        normalized_ext = _normalize_extension(path.suffix)
        if normalized_ext != ".zip":
            raise ValueError("Provided file is not a .zip archive.")

        self._extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(self._extract_dir)

        supported_exts = set(DataIngestorFactory.supported_extensions(exclude_archive=True))
        extracted_files: list[Path] = []
        for root, _, files in os.walk(self._extract_dir):
            for fname in files:
                p = Path(root) / fname
                ext = p.suffix
                if ext and _normalize_extension(ext) in supported_exts:
                    extracted_files.append(p)

        if not extracted_files:
            raise FileNotFoundError("No supported data file found in the ZIP archive.")
        if len(extracted_files) > 1:
            raise ValueError(
                "Multiple supported files found in the ZIP archive. "
                "Please specify which file to use."
            )

        inner_path = extracted_files[0]
        inner_ingestor = DataIngestorFactory.get_data_ingestor(inner_path.suffix)
        return inner_ingestor.ingest(inner_path)


class DataIngestorFactory:
    """Factory to instantiate the appropriate DataIngestor by file extension."""

    _INGESTOR_REGISTRY: Dict[str, Callable[..., DataIngestor]] = {
        ".csv": CSVDataIngestor,
        ".tsv": TSVDataIngestor,
        ".json": JSONDataIngestor,
        ".xlsx": ExcelDataIngestor,
        ".xls": ExcelDataIngestor,
        ".parquet": ParquetDataIngestor,
        ".zip": ZipDataIngestor,
    }

    @classmethod
    def get_data_ingestor(
        cls, file_extension: str | Path, *, zip_extract_dir: Optional[str] = None
    ) -> DataIngestor:
        normalized_ext = _normalize_extension(str(file_extension))
        try:
            ingestor_cls = cls._INGESTOR_REGISTRY[normalized_ext]
        except KeyError as exc:
            supported = ", ".join(sorted(cls._INGESTOR_REGISTRY))
            raise ValueError(
                f"No ingestor available for file extension: {normalized_ext}. "
                f"Supported: {supported}"
            ) from exc

        if normalized_ext == ".zip":
            return ingestor_cls(zip_extract_dir=zip_extract_dir)
        return ingestor_cls()

    @classmethod
    def supported_extensions(cls, *, exclude_archive: bool = False) -> Tuple[str, ...]:
        if exclude_archive:
            return tuple(ext for ext in cls._INGESTOR_REGISTRY if ext != ".zip")
        return tuple(cls._INGESTOR_REGISTRY)


def _resolve_cache_dir(value: Optional[str], *, project_root: Path) -> Path:
    """
    Resolve cache directory while keeping downloads inside the shared project root.

    DockerOperator tasks run in separate containers, so placing caches under
    arbitrary absolute paths (for example /tmp) makes the artifacts invisible to
    subsequent steps. When an absolute path outside the project root is provided,
    we scope it under the default cache folder.
    """
    default_cache = project_root / "data" / "object_storage_cache"
    if not value:
        return default_cache
    path = Path(value)
    if not path.is_absolute():
        return (project_root / path).resolve()
    project_root_resolved = project_root.resolve()
    path_resolved = path.resolve()
    try:
        path_resolved.relative_to(project_root_resolved)
        return path_resolved
    except ValueError:
        normalized = Path(_strip_leading_slash(str(path)))
        return (default_cache / normalized).resolve()


def _strip_leading_slash(value: str) -> str:
    return value.lstrip("/\\")


def _parse_s3_uri(uri: str) -> tuple[Optional[str], Optional[str]]:
    if not uri:
        return None, None
    parsed = urlparse(uri)
    if parsed.scheme.lower() != "s3":
        return None, None
    bucket = parsed.netloc or None
    key = parsed.path.lstrip("/") or None
    return bucket, key


def ingest_data(
    config: Dict[str, Any], *, project_root: Path
) -> tuple[pd.DataFrame, Path]:
    """
    Load a dataset exclusively from object storage.
    Returns a tuple of (dataframe, local_file_path).
    """
    file_hint = config.get("file_path")
    bucket_from_uri, key_from_uri = _parse_s3_uri(str(file_hint)) if file_hint else (None, None)
    if not bucket_from_uri or not key_from_uri:
        raise ValueError(
            "Object storage ingestion requires file_path to be a full s3://<bucket>/<key> URI."
        )
    dataset_key = key_from_uri
    bucket_override = bucket_from_uri

    extract_dir = config.get("zip_extract_dir")
    if extract_dir and not Path(extract_dir).is_absolute():
        extract_dir = str(project_root / extract_dir)

    cache_dir = _resolve_cache_dir(config.get("cache_dir"), project_root=project_root)
    relative_key = Path(_strip_leading_slash(dataset_key))
    resolved_path = cache_dir / relative_key
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        client = build_storage_client(bucket=bucket_override)
        client.download(object_key=dataset_key, destination=resolved_path, bucket=bucket_override)
    except ObjectStorageConfigurationError as exc:
        raise RuntimeError(
            "Failed to configure object storage client for ingestion. "
            "Ensure OBJECT_STORAGE_* env vars are set."
        ) from exc
    except ObjectStorageOperationError as exc:
        raise RuntimeError(
            f"Unable to download dataset '{dataset_key}' from object storage."
        ) from exc

    extension = config.get("file_extension") or resolved_path.suffix or Path(dataset_key).suffix
    ingestor = DataIngestorFactory.get_data_ingestor(extension, zip_extract_dir=extract_dir)
    dataframe = ingestor.ingest(resolved_path)
    return dataframe, resolved_path
