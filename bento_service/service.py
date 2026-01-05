"""Basic BentoML service definition for a scikit-learn model."""

from typing import List

import bentoml
from bentoml.io import JSON
from pydantic import BaseModel

# Update the MODEL_TAG with the tag shown by `bentoml models list`.
MODEL_TAG = "model:latest"

model_ref = bentoml.sklearn.get(MODEL_TAG)
model_runner = model_ref.to_runner()
svc = bentoml.Service("sklearn_service", runners=[model_runner])


class PredictionRequest(BaseModel):
    samples: List[List[float]]


@svc.api(input=JSON(pydantic_model=PredictionRequest), output=JSON())
async def predict(request: PredictionRequest):
    """Return predictions for each row in `samples`."""
    predictions = await model_runner.predict.async_run(request.samples)
    return {"predictions": predictions.tolist()}



