"""Basic BentoML service definition for a scikit-learn model."""

from typing import List

import os

import bentoml
from pydantic import BaseModel

# Update the MODEL_TAG with the tag shown by `bentoml models list`.
MODEL_TAG = os.getenv("BENTOML_MODEL_TAG", "model:latest")
SERVICE_NAME = os.getenv("BENTO_SERVICE_NAME", "sklearn_service")


class PredictionRequest(BaseModel):
    samples: List[List[float]]


class PredictionResponse(BaseModel):
    predictions: List[float]


@bentoml.service(name=SERVICE_NAME)
class SklearnService:
    def __init__(self) -> None:
        self.model = bentoml.sklearn.load_model(MODEL_TAG)

    @bentoml.api(input_spec=PredictionRequest, output_spec=PredictionResponse)
    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Return predictions for each row in `samples`."""
        predictions = self.model.predict(request.samples)
        return PredictionResponse(predictions=predictions.tolist())



