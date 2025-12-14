from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict

import optuna
from optuna import Trial, TrialPruned
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split

from .train_utils import select_model


@dataclass
class SearchSpec:
    type: str
    low: float | int | None = None
    high: float | int | None = None
    step: float | int | None = None
    log: bool = False
    choices: list[Any] | None = None


MODEL_DIRECTIONS: Dict[str, str] = {
    "random_forest_classifier": "maximize",
    "logistic_regression": "maximize",
    "random_forest_regressor": "minimize",
    "linear_regression": "minimize",
}

DEFAULT_SEARCH_SPACES: Dict[str, Dict[str, SearchSpec]] = {
    "random_forest_classifier": {
        "n_estimators": SearchSpec("int", low=50, high=500, step=10),
        "max_depth": SearchSpec("int", low=2, high=20),
        "min_samples_split": SearchSpec("int", low=2, high=10),
        "min_samples_leaf": SearchSpec("int", low=1, high=5),
    },
    "random_forest_regressor": {
        "n_estimators": SearchSpec("int", low=50, high=400, step=10),
        "max_depth": SearchSpec("int", low=2, high=20),
        "min_samples_split": SearchSpec("int", low=2, high=10),
        "min_samples_leaf": SearchSpec("int", low=1, high=5),
    },
    "logistic_regression": {
        "C": SearchSpec("float", low=0.001, high=10, log=True),
        "max_iter": SearchSpec("int", low=100, high=1000, step=50),
    },
}


def _serialize_spec(raw: Dict[str, Any]) -> SearchSpec:
    spec_type = raw.get("type")
    if not spec_type:
        raise ValueError("Every search space entry must define 'type'.")
    spec_type = spec_type.lower()
    if spec_type not in {"int", "float", "categorical"}:
        raise ValueError(f"Unsupported search space type '{spec_type}'.")
    if spec_type == "categorical":
        choices = raw.get("choices")
        if not choices:
            raise ValueError("Categorical search space requires 'choices'.")
        return SearchSpec(type=spec_type, choices=list(choices))
    low = raw.get("low")
    high = raw.get("high")
    if low is None or high is None:
        raise ValueError(f"Search space '{spec_type}' requires 'low' and 'high'.")
    step = raw.get("step")
    log = bool(raw.get("log", False))
    return SearchSpec(type=spec_type, low=low, high=high, step=step, log=log)


def _combine_search_space(model_name: str, overrides: Dict[str, Any]) -> Dict[str, SearchSpec]:
    base_specs = DEFAULT_SEARCH_SPACES.get(model_name.lower(), {})
    combined: Dict[str, SearchSpec] = dict(base_specs)
    for key, raw_spec in overrides.items():
        combined[key] = _serialize_spec(raw_spec)
    return combined


def _suggest_value(trial: Trial, param_name: str, spec: SearchSpec):
    if spec.type == "int":
        step = int(spec.step) if spec.step else 1
        return trial.suggest_int(param_name, int(spec.low), int(spec.high), step=step)
    if spec.type == "float":
        return trial.suggest_float(
            param_name,
            float(spec.low),
            float(spec.high),
            log=spec.log,
            step=float(spec.step) if spec.step else None,
        )
    return trial.suggest_categorical(param_name, spec.choices)


def _trial_objective(
    trial: Trial,
    *,
    model_name: str,
    base_params: Dict[str, Any],
    search_space: Dict[str, SearchSpec],
    X_train,
    X_valid,
    y_train,
    y_valid,
    task_type: str,
):
    params = dict(base_params)
    for name, spec in search_space.items():
        params[name] = _suggest_value(trial, name, spec)
    try:
        model = select_model(model_name, params)
        model.fit(X_train, y_train)
        preds = model.predict(X_valid)
    except Exception as exc:  # pragma: no cover - defensive
        raise TrialPruned() from exc

    if task_type == "regression":
        metric = mean_squared_error(y_valid, preds)
        return metric
    metric = accuracy_score(y_valid, preds)
    return metric


def _infer_task_type(model_name: str) -> str:
    name = model_name.lower()
    if "regressor" in name or "regression" in name and not name.endswith("classifier"):
        return "regression"
    if name in {"linear_regression"}:
        return "regression"
    return "classification"


def perform_hyperparameter_search(
    *,
    model_name: str,
    base_hyperparameters: Dict[str, Any],
    search_space_overrides: Dict[str, Any],
    features,
    target,
    n_trials: int,
    timeout: int | None,
    test_size: float,
    random_state: int,
) -> dict[str, Any]:
    task_type = _infer_task_type(model_name)
    direction = MODEL_DIRECTIONS.get(model_name.lower(), "maximize")
    combined_space = _combine_search_space(model_name, search_space_overrides)
    if not combined_space:
        return dict(base_hyperparameters)

    X_train, X_valid, y_train, y_valid = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target if task_type == "classification" else None,
    )

    study = optuna.create_study(direction=direction)
    study.optimize(
        lambda trial: _trial_objective(
            trial,
            model_name=model_name,
            base_params=base_hyperparameters,
            search_space=combined_space,
            X_train=X_train,
            X_valid=X_valid,
            y_train=y_train,
            y_valid=y_valid,
            task_type=task_type,
        ),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=False,
    )

    best_params = dict(base_hyperparameters)
    best_params.update(study.best_params)
    return best_params


def parse_json_payload(value: str) -> Dict[str, Any]:
    if not value:
        return {}
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise SystemExit(f"Invalid JSON payload provided to tuner: {exc}") from exc
