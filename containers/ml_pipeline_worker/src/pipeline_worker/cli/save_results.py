"""Finalize model artifact location inside a container task."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute final model artifact destination.")
    parser.add_argument("--model-artifact-path", required=True, help="Path returned by training step.")
    parser.add_argument("--target-output-path", required=True, help="Base prefix/directory for storage.")

    args = parser.parse_args()
    artifact_name = Path(args.model_artifact_path).name
    base = args.target_output_path.rstrip("/")
    if base:
        saved_location = f"{base}/{artifact_name}"
    else:
        saved_location = artifact_name

    print(f"Saving model artifact from {args.model_artifact_path} to {saved_location}", file=sys.stderr)
    print(saved_location)


if __name__ == "__main__":
    main()
