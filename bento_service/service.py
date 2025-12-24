import numpy as np
import bentoml
from bentoml.io import JSON

model_ref = bentoml.models.get("model:latest")
runner = bentoml.sklearn.get(model_ref).to_runner()
svc = bentoml.Service(
    "model",
    runners=[runner],
    traffic={
        "timeout": 30,
        "max_request_size": 1048576,
    },
    config=bentoml.ConfigDict(
        logging={"inference": {"enabled": True, "sample_rate": 1.0}},
        tracing={"sample_rate": 1.0},
        monitoring={
            "enable_prometheus": True,
            "prometheus_namespace": "bento",
            "prometheus_port": 3000,
        },
    ),
)


@svc.api(input=JSON(), output=JSON())
async def predict(payload):
    X = np.array(payload["data"])
    y = await runner.predict.async_run(X)
    return {"pred": y.tolist()}
