import numpy as np
import bentoml
from bentoml.io import JSON

model_ref = bentoml.models.get("model:latest")  # hoặc "model:<tag>"
runner = bentoml.sklearn.get(model_ref).to_runner()
svc = bentoml.Service("model", runners=[runner])

@svc.api(input=JSON(), output=JSON())
async def predict(payload):
    # payload ví dụ: {"data": [[1,2,3,4], [5,6,7,8]]}
    X = np.array(payload["data"])
    y = await runner.predict.async_run(X)
    return {"pred": y.tolist()}
