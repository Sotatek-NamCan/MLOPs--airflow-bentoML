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
from typing import Any, Dict
from urllib.parse import urlparse

import bentoml
import boto3
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".envbento"
if ENV_FILE.exists():
    load_dotenv(dotenv_path=ENV_FILE, override=False)


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


def _resolve_artifact_path(model_path: str) -> Path:
    if not model_path.lower().startswith("s3://"):
        return Path(model_path)

    parsed = urlparse(model_path)
    bucket = parsed.netloc
    object_key = parsed.path.lstrip("/")
    destination = Path(tempfile.mkdtemp()) / Path(object_key).name
    _build_s3_client().download_file(bucket, object_key, str(destination))
    return destination


def _build_s3_client():
    settings = {
        "endpoint_url": os.getenv("OBJECT_STORAGE_ENDPOINT_URL"),
        "region_name": os.getenv("OBJECT_STORAGE_REGION"),
        "aws_access_key_id": os.getenv("OBJECT_STORAGE_ACCESS_KEY"),
        "aws_secret_access_key": os.getenv("OBJECT_STORAGE_SECRET_KEY"),
    }
    clean_settings: Dict[str, str] = {k: v for k, v in settings.items() if v}
    return boto3.client("s3", **clean_settings)


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
    artifact_path = _resolve_artifact_path(args.model_path)
    model = _load_model(artifact_path)
    signatures = _build_signatures(model)
    extra_metadata = json.loads(args.metadata)

    saved_model = bentoml.picklable_model.save_model(
        name=args.bento_tag,
        model=model,
        signatures=signatures,
        metadata={"source_artifact": args.model_path, **extra_metadata},
    )
    print(f"[bento] Imported model as {saved_model.tag}")


if __name__ == "__main__":
    main()
