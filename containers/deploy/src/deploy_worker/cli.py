from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

from .github import GitHubActionsClient, WorkflowDispatchPayload


def _parse_inputs(value: str | None) -> Dict[str, Any]:
    if not value:
        return {}
    try:
        data = json.loads(value)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise SystemExit(f"Invalid JSON for --extra-inputs: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit("--extra-inputs must be a JSON object.")
    return data


def _env_or_arg(*, token: str | None, env_var: str) -> str | None:
    if token:
        return token
    return os.getenv(env_var)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Trigger GitHub Action to deploy BentoML service.",
    )
    parser.add_argument("--repository", required=True, help="GitHub repository in owner/name format.")
    parser.add_argument("--workflow", required=True, help="Workflow file name or numeric ID to trigger.")
    parser.add_argument("--ref", default="main", help="Git reference (branch or tag) for the workflow run.")
    parser.add_argument(
        "--token",
        help="Optional GitHub token value (otherwise read from --token-env).",
    )
    parser.add_argument(
        "--token-env",
        default="GITHUB_TOKEN",
        help="Environment variable containing GitHub token. Default: GITHUB_TOKEN",
    )
    parser.add_argument("--model-name", required=True, help="Model identifier registered in BentoML.")
    parser.add_argument("--model-version", required=True, help="Model version to deploy.")
    parser.add_argument("--artifact-uri", required=True, help="S3/URI pointing to trained artifact.")
    parser.add_argument(
        "--bentoml-target",
        required=True,
        help="BentoML deployment target (service or deployment name).",
    )
    parser.add_argument(
        "--deploy-environment",
        default="production",
        help="Deployment environment string passed to workflow inputs.",
    )
    parser.add_argument(
        "--extra-inputs",
        default=None,
        help="Additional workflow inputs as JSON object.",
    )
    parser.add_argument(
        "--api-url",
        default="https://api.github.com",
        help="GitHub API base URL (override for GH Enterprise).",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    token = _env_or_arg(token=args.token, env_var=args.token_env)
    if not token:
        raise SystemExit(
            f"No GitHub token provided. Use --token or set the {args.token_env} environment variable."
        )

    extra_inputs = _parse_inputs(args.extra_inputs)
    workflow_inputs: Dict[str, Any] = {
        "model_name": args.model_name,
        "model_version": args.model_version,
        "artifact_uri": args.artifact_uri,
        "bentoml_target": args.bentoml_target,
        "deploy_environment": args.deploy_environment,
    }
    workflow_inputs.update(extra_inputs)

    client = GitHubActionsClient(token=token, api_url=args.api_url)
    payload = WorkflowDispatchPayload(
        repository=args.repository,
        workflow=args.workflow,
        ref=args.ref,
        inputs={key: str(value) for key, value in workflow_inputs.items()},
    )
    response = client.trigger_workflow(payload)
    print(
        "Triggered deployment workflow "
        f"{args.workflow} on {args.repository}@{args.ref} (status={response.status_code})."
    )


if __name__ == "__main__":
    main()
