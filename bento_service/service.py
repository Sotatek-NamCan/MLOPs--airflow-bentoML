"""BentoML service for driver prediction models produced by the Airflow pipeline."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import bentoml
import pandas as pd
from bentoml.io import JSON
from pydantic import BaseModel, Field, validator


DEFAULT_MODEL_TAG = os.getenv("BENTOML_MODEL_TAG", "driver_prediction:latest")
MODEL_REF = bentoml.models.get(DEFAULT_MODEL_TAG)
MODEL_RUNNER = MODEL_REF.to_runner()

svc = bentoml.Service(
    name=os.getenv("BENTO_SERVICE_NAME", "driver_prediction_service"),
    runners=[MODEL_RUNNER],
)


class PredictionRequest(BaseModel):
    """
    Incoming payload with a list of feature dictionaries.

    Each dictionary should contain the same feature keys that were used during
    training (all columns except the target column).
    """

    instances: List[Dict[str, Any]] = Field(..., description="Feature rows to score.")

    @validator("instances")
    def _ensure_instances(cls, value: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not value:
            raise ValueError("instances payload must contain at least one entry.")
        return value

    def to_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame(self.instances)
        if frame.empty:
            raise ValueError("Unable to build dataframe from payload.")
        return frame


class PredictionResponse(BaseModel):
    predictions: List[Any]
    probabilities: Optional[List[Any]] = None


@svc.api(
    input=JSON(pydantic_model=PredictionRequest),
    output=JSON(pydantic_model=PredictionResponse),
)
async def predict(payload: PredictionRequest) -> Dict[str, Any]:
    """
    Score incoming records with the registered scikit-learn model.

    The runner mirrors the pickled estimator, so `predict` and (when available)
    `predict_proba` behave exactly like in the training script.
    """

    frame = payload.to_frame()
    raw_predictions = await MODEL_RUNNER.predict.async_run(frame)
    response: Dict[str, Any] = {"predictions": raw_predictions.tolist()}

    if "predict_proba" in MODEL_REF.info.signatures:
        proba = await MODEL_RUNNER.predict_proba.async_run(frame)
        response["probabilities"] = proba.tolist()

    return response
