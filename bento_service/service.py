from __future__ import annotations

import os
from typing import Any, List

import numpy as np
import bentoml
from bentoml.io import JSON
from pydantic import BaseModel


MODEL_TAG = os.getenv("BENTOML_MODEL_TAG", "model:latest")
MODEL_REF = bentoml.sklearn.get(MODEL_TAG)
MODEL_RUNNER = MODEL_REF.to_runner()

svc = bentoml.Service(
    name=os.getenv("BENTO_SERVICE_NAME", "model"),
    runners=[MODEL_RUNNER],
    traffic={"timeout": 30},
    http={"port": 3000},
    metrics={"enabled": True, "namespace": "bento"},
    tracing={"sample_rate": 1.0},
)


class PredictIn(BaseModel):
    data: List[List[float]]


class PredictOut(BaseModel):
    pred: Any


@svc.api(
    input=JSON(pydantic_model=PredictIn),
    output=JSON(pydantic_model=PredictOut),
)
async def predict(payload: PredictIn) -> PredictOut:
    X = np.asarray(payload.data)
    y = await MODEL_RUNNER.predict.async_run(X)
    return PredictOut(pred=y.tolist())
