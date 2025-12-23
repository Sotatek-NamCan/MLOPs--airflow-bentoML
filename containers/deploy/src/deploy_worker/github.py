from __future__ import annotations

import json
from dataclasses import dataclass

import requests
from requests import Response, Session


class GitHubActionError(RuntimeError):
    """Raised when triggering a GitHub Action workflow fails."""


@dataclass
class WorkflowDispatchPayload:
    repository: str
    workflow: str
    ref: str
    inputs: dict[str, str]


class GitHubActionsClient:
    """Minimal client for triggering GitHub workflow dispatch events."""

    def __init__(self, *, token: str, api_url: str = "https://api.github.com"):
        if not token:
            raise ValueError("GitHub token is required to trigger workflows.")
        self.api_url = api_url.rstrip("/")
        self.session: Session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "Content-Type": "application/json",
                "User-Agent": "deploy-worker/1.0",
            }
        )

    def trigger_workflow(self, payload: WorkflowDispatchPayload) -> Response:
        repo = payload.repository.strip()
        url = f"{self.api_url}/repos/{repo}/.github/workflows/{payload.workflow}/dispatches"
        normalized_inputs = {k: str(v) for k, v in payload.inputs.items() if v is not None}
        body = {"ref": payload.ref, "inputs": normalized_inputs}
        response = self.session.post(url, data=json.dumps(body))
        if response.status_code not in (200, 201, 202, 204):
            raise GitHubActionError(
                f"GitHub workflow dispatch failed ({response.status_code}): {response.text}"
            )
        return response
