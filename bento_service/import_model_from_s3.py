from __future__ import annotations

import argparse
import os
import pickle
import tempfile
from pathlib import Path
from datetime import datetime

import bentoml
import boto3
from dotenv import load_dotenv


def _load_env() -> None:
    env_path = Path(__file__).parent / ".envbento"
    if env_path.exists():
        load_dotenv(env_path)


def _s3_client():
    session = boto3.session.Session(
        aws_access_key_id=os.getenv("OBJECT_STORAGE_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("OBJECT_STORAGE_SECRET_KEY"),
        region_name=os.getenv("OBJECT_STORAGE_REGION"),
    )
    endpoint = os.getenv("OBJECT_STORAGE_ENDPOINT_URL")
    if endpoint:
        return session.client("s3", endpoint_url=endpoint)
    return session.client("s3")


def import_model_from_s3(
    *,
    s3_uri: str,
    model_name: str,
    model_version: str,
) -> str:
    no_scheme = s3_uri[len("s3://") :]
    bucket, _, key = no_scheme.partition("/")
    client = _s3_client()
    with tempfile.TemporaryDirectory() as tmp_dir:
        local_path = Path(tmp_dir) / "model.pkl"
        client.download_file(bucket, key, str(local_path))
        with open(local_path, "rb") as f:
            model_obj = pickle.load(f)

        resolved_version = model_version
        if model_version.lower() == "latest":
            resolved_version = datetime.utcnow().strftime("%Y%m%d%H%M%S")
            print(
                "[Import] 'latest' is reserved by BentoML. "
                f"Using generated version {resolved_version} and 'model:latest' will point to it automatically."
            )

        save_tag = f"{model_name}:{resolved_version}"
        bentoml.sklearn.save_model(
            save_tag,
            model_obj,
            metadata={"source_s3_uri": s3_uri},
        )
    return save_tag


def main() -> None:
    _load_env()
    parser = argparse.ArgumentParser(description="Import a pickled model from S3 into BentoML store.")
    parser.add_argument("--s3-uri", required=True, help="S3 URI to the model.pkl artifact.")
    parser.add_argument("--model-name", required=True, help="Target Bento model name.")
    parser.add_argument("--model-version", required=True, help="Target Bento model version/tag.")
    args = parser.parse_args()

    tag = import_model_from_s3(
        s3_uri=args.s3_uri,
        model_name=args.model_name,
        model_version=args.model_version,
    )
    print(f"Imported model into Bento store as: {tag}")


if __name__ == "__main__":
    main()
