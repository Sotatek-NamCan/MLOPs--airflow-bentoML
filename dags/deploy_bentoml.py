from __future__ import annotations

import base64
import json
import os
import subprocess
import tempfile
from datetime import datetime, timedelta, timezone

import requests
from airflow import DAG
from airflow.decorators import task
from airflow.exceptions import AirflowException
from airflow.models import Variable

GITHUB_TOKEN_VAR_KEY = "github_api"
OWNER = "Sotatek-NamCan"
REPO = "MLOPs--airflow-bentoML"
BRANCH = "main"  
FILE_PATH = "bento_service/model.version"  
COMMITTER_NAME = "airflow-bot"
COMMITTER_EMAIL = "airflow-bot@users.noreply.github.com"


def _github_headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "Content-Type": "application/json",
        "User-Agent": "airflow-github-committer",
    }


def _get_github_token_from_variable(var_key: str = GITHUB_TOKEN_VAR_KEY) -> str:
    token = Variable.get(var_key, default_var=None)
    if not token:
        raise AirflowException(
            f"Missing GitHub token in Airflow Variable '{var_key}'. "
            f"Create it in Admin → Variables."
        )
    return token

def _escape_for_sed_replacement(s: str) -> str:
    """
    Escape replacement text for sed (basic safe escaping).
    We use delimiter | so we must escape: \ and | and &
    """
    return s.replace("\\", "\\\\").replace("|", "\\|").replace("&", "\\&")


default_args = {
    "owner": "airflow",
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="github_deploy",
    start_date=datetime(2024, 1, 1),  # nếu muốn timezone-aware thì add tzinfo=timezone.utc
    schedule=None,
    catchup=False,
    render_template_as_native_obj=True,
    params={
        "model_version": "2025-12-22_001",
        "model_path": "my_model",
        "commit_message": "",  # blank
    },
    tags=["deploy", "github"],
) as dag:

    @task
    def update_model_in_github_file(**context) -> dict:
        p = context["params"]  # lấy params từ UI

        model_version = (p.get("model_version") or "").strip()
        model_path = (p.get("model_path") or "").strip()
        commit_message = (p.get("commit_message") or "").strip() or None

        if not model_version or not model_path:
            raise AirflowException("Missing required params: model_version, model_path")

        token = _get_github_token_from_variable()
        headers = _github_headers(token)

        if commit_message is None:
            commit_message = f"chore: bump model to {model_path}:{model_version}"

        content_url = f"https://api.github.com/repos/{OWNER}/{REPO}/contents/{FILE_PATH}"

        # GET file (sha + content)
        r = requests.get(content_url, headers=headers, params={"ref": BRANCH}, timeout=30)
        if r.status_code == 404:
            raise AirflowException(f"File not found: {FILE_PATH} on branch {BRANCH}")
        if r.status_code >= 400:
            raise AirflowException(f"GitHub GET failed: {r.status_code} {r.text}")

        data = r.json()
        sha = data.get("sha")
        content_b64 = data.get("content")
        if not sha or not content_b64:
            raise AirflowException("GitHub response missing sha/content (unexpected).")

        original_text = base64.b64decode(content_b64).decode("utf-8")

        # sed replace: ^model: ... -> model: "<model_path>:<model_version>"
        new_model_value = f'{model_path}:{model_version}'
        new_model_value_escaped = _escape_for_sed_replacement(new_model_value)

        with tempfile.TemporaryDirectory() as td:
            tmp_path = os.path.join(td, "target.yaml")
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(original_text)

            sed_expr = (
                f'0,/^model:[[:space:]]*.*/s|^model:[[:space:]]*.*|model: "{new_model_value_escaped}"|'
            )
            subprocess.run(["sed", "-E", "-i", sed_expr, tmp_path], check=True)

            with open(tmp_path, "r", encoding="utf-8") as f:
                updated_text = f.read()

        if updated_text == original_text:
            raise AirflowException("No change after sed. Does file contain a 'model:' line?")

        updated_b64 = base64.b64encode(updated_text.encode("utf-8")).decode("utf-8")
        payload = {
            "message": commit_message,
            "content": updated_b64,
            "sha": sha,
            "branch": BRANCH,
            "committer": {"name": COMMITTER_NAME, "email": COMMITTER_EMAIL},
        }

        r2 = requests.put(content_url, headers=headers, data=json.dumps(payload), timeout=30)
        if r2.status_code >= 400:
            raise AirflowException(f"GitHub PUT failed: {r2.status_code} {r2.text}")

        out = r2.json()
        return {
            "status": "ok",
            "model": f"{model_path}:{model_version}",
            "commit_url": out.get("commit", {}).get("html_url"),
        }

    update_model_in_github_file()