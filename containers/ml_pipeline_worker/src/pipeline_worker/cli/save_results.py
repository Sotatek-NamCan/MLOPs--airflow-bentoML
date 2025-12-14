"""Finalize model artifact location inside a container task."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pipeline_worker.artifacts import artifact_name, copy_artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy trained model artifact to final destination.")
    parser.add_argument("--model-artifact-path", required=True, help="Source path or URI from training.")
    parser.add_argument("--target-output-path", required=True, help="Destination directory or URI prefix.")

    args = parser.parse_args()
    base = args.target_output_path.rstrip("/")
    name = artifact_name(args.model_artifact_path)
    destination = f"{base}/{name}" if base else name

    print(f"Copying model artifact to {destination}", file=sys.stderr)
    final_location = copy_artifact(args.model_artifact_path, destination)
    print(final_location)


if __name__ == "__main__":
    main()
