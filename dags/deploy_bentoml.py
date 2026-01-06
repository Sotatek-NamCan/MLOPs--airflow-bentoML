from __future__ import annotations

import base64
import json
import re
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

def _replace_yaml_key(text: str, key: str, value: str) -> str:
    pattern = rf"^{re.escape(key)}:[ \t]*.*$"
    replacement = f'{key}: "{value}"'
    new_text, count = re.subn(pattern, replacement, text, flags=re.MULTILINE)
    if count == 0:
        # Append missing key at the end of the file.
        suffix = "" if new_text.endswith("\n") or new_text == "" else "\n"
        return f"{new_text}{suffix}{replacement}\n"
    return new_text


default_args = {
    "owner": "airflow",
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="ml_github_deploy_5",
    start_date=datetime(2024, 1, 1),  # nếu muốn timezone-aware thì add tzinfo=timezone.utc
    schedule=None,
    catchup=False,
    render_template_as_native_obj=True,
    params={
        "model-name": "my_model",
        "model-version": "2025-12-22_001",
        "model-path": "models/my_model",
        "description": "",  # blank
    },
    tags=["deploy", "github"],
) as dag:

    @task
    def update_model_in_github_file(**context) -> dict:
        p = context["params"]  # lấy params từ UI

        model_name = (p.get("model-name") or "").strip()
        model_version = (p.get("model-version") or "").strip()
        model_path = (p.get("model-path") or "").strip()
        description = (p.get("description") or "").strip()

        if not model_name or not model_version or not model_path or not description:
            raise AirflowException(
                "Missing required params: model-name, model-version, model-path, description"
            )

        token = _get_github_token_from_variable()
        headers = _github_headers(token)

        commit_message = description

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

        updated_text = original_text
        updated_text = _replace_yaml_key(updated_text, "model-name", model_name)
        updated_text = _replace_yaml_key(updated_text, "model-version", model_version)
        updated_text = _replace_yaml_key(updated_text, "model-path", model_path)
        updated_text = _replace_yaml_key(updated_text, "description", description)

        if updated_text == original_text:
            raise AirflowException("No change after update. Are the values already set?")

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
            "model-name": model_name,
            "model-version": model_version,
            "model-path": model_path,
            "description": description,
            "commit_url": out.get("commit", {}).get("html_url"),
        }

    update_model_in_github_file()
