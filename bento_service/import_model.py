#!/usr/bin/env python
"""
Register a trained Airflow model artifact inside the local BentoML model store.

Example:
    python import_model.py --model-path s3://driver-training/models/random_forest.pkl \
        --bento-tag driver_prediction:2024-11-25
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any, Dict, Tuple
from urllib.parse import urlparse

import bentoml
import boto3
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]

def _load_env_file() -> None:
    """
    Load credentials from the local .envbento file if present.

    Keeping the load logic here avoids depending on the Airflow worker package.
    """

    env_file = PROJECT_ROOT / ".envbento"
    if not env_file.exists():
        return

    load_dotenv(dotenv_path=env_file, override=False)


_load_env_file()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import a scikit-learn artifact into BentoML.")
    parser.add_argument(
        "--model-path",
        required=True,
        help="Local path or s3:// URI of the pickle artifact produced by the Airflow pipeline.",
    )
    parser.add_argument(
        "--bento-tag",
        default=os.getenv("BENTOML_MODEL_TAG", "driver_prediction:latest"),
        help="Tag assigned to the Bento model (format: name:version).",
    )
    parser.add_argument(
        "--metadata",
        default="{}",
        help="Optional JSON string with metadata to store alongside the model.",
    )
    return parser.parse_args()


def _is_s3(uri: str) -> bool:
    return uri.lower().startswith("s3://")


def _parse_s3_uri(uri: str) -> Tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError("S3 URI must be of the form s3://bucket/key")
    return parsed.netloc, parsed.path.lstrip("/")


def _download_from_s3(uri: str) -> Path:
    bucket, object_key = _parse_s3_uri(uri)
    client = _build_s3_client()
    tmp_dir = Path(tempfile.mkdtemp())
    destination = tmp_dir / Path(object_key).name
    try:
        client.download_file(bucket, object_key, str(destination))
    except Exception as exc:  # pragma: no cover - surface boto3 errors
        raise SystemExit(
            f"Failed to download '{object_key}' from bucket '{bucket}': {exc}"
        ) from exc
    return destination


def _build_s3_client():
    endpoint_url = os.getenv("OBJECT_STORAGE_ENDPOINT_URL")
    region_name = os.getenv("OBJECT_STORAGE_REGION")
    access_key = os.getenv("OBJECT_STORAGE_ACCESS_KEY")
    secret_key = os.getenv("OBJECT_STORAGE_SECRET_KEY")
    
    if not access_key or not secret_key:
        raise SystemExit(
            "Missing OBJECT_STORAGE_ACCESS_KEY/OBJECT_STORAGE_SECRET_KEY in .envbento."
        )

    session_kwargs: Dict[str, str] = {}
    if endpoint_url:
        session_kwargs["endpoint_url"] = endpoint_url
    if region_name:
        session_kwargs["region_name"] = region_name
    session_kwargs["aws_access_key_id"] = access_key
    session_kwargs["aws_secret_access_key"] = secret_key

    try:
        return boto3.client("s3", **session_kwargs)
    except Exception as exc:  # pragma: no cover - boto3 misconfiguration
        raise SystemExit(f"Unable to configure boto3 S3 client: {exc}") from exc


def _load_model(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def _build_signatures(model: Any) -> Dict[str, Dict[str, Any]]:
    signatures: Dict[str, Dict[str, Any]] = {"predict": {"batchable": True}}
    if hasattr(model, "predict_proba"):
        signatures["predict_proba"] = {"batchable": True}
    return signatures


def main() -> None:
    args = parse_args()
    artifact_path = Path(args.model_path)
    if _is_s3(args.model_path):
        artifact_path = _download_from_s3(args.model_path)

    model = _load_model(artifact_path)
    signatures = _build_signatures(model)
    try:
        extra_metadata = json.loads(args.metadata)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid metadata JSON: {exc}") from exc

    saved_model = bentoml.picklable_model.save_model(
        name=args.bento_tag,
        model=model,
        signatures=signatures,
        metadata={
            "source_artifact": args.model_path,
            **extra_metadata,
        },
    )
    print(f"[bento] Imported model as {saved_model.tag}")


if __name__ == "__main__":
    main()
