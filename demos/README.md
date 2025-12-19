## Demo Assets Overview

This folder contains ready-made datasets and parameter presets that mirror the
`driver_safe_param.json` structure so you can trigger the DAG or the standalone
CLI without hand-crafting JSON payloads. Each scenario pairs a CSV dataset with
the model family it was designed for.

### Folder Layout

- `driver_safety/`: Original driver high-risk classification sample plus relaxed
  validation settings (`driver_safety.csv`, `params_driver_safety*.json`).
- `datasets/`: Synthetic multi-hundred-row CSVs for additional demos.
- `params/`: JSON payloads that follow the same schema as `driver_safe_param.json`
  and reference the datasets above.

### Available Scenarios

| Model | Dataset | Param file | Target column | Notes |
| --- | --- | --- | --- | --- |
| `random_forest_classifier` | `demos/datasets/customer_churn_demo.csv` | `demos/params/random_forest_classifier_customer_churn.json` | `churned` | Customer churn classification with categorical contract hints baked into numeric features. |
| `logistic_regression` | `demos/datasets/lead_conversion_demo.csv` | `demos/params/logistic_regression_lead_conversion.json` | `converted` | Marketing lead conversion likelihood; good for showcasing interpretable coefficients. |
| `linear_regression` | `demos/datasets/housing_prices_demo.csv` | `demos/params/linear_regression_housing_prices.json` | `price` | Housing price regression with continuous inputs (square feet, walk score, etc.). |
| `random_forest_regressor` | `demos/datasets/energy_demand_demo.csv` | `demos/params/random_forest_regressor_energy.json` | `demand_mwh` | Regional energy demand forecasting; includes validation bounds for environmental metrics. |
| `random_forest_classifier` (driver safety) | `demos/driver_safety/driver_safety.csv` | `demos/driver_safety/params_driver_safety*.json` | `target_high_risk` | Same dataset used in docs; the relaxed variant reduces validation strictness. |

### Using a Demo Param File

1. Pick a scenario from the table.
2. Copy the corresponding JSON into your DAG run configuration (Airflow UI →
   “Trigger DAG” → “Config”) or run locally:

   ```bash
   python -m pipeline_worker.cli.train_model \
     --train-data-path demos/datasets/customer_churn_demo.csv \
     --target-column churned \
     --model-name random_forest_classifier \
     --model-version 1 \
     --hyperparameters "$(Get-Content demos/params/random_forest_classifier_customer_churn.json | jq '.hyperparameters')" \
     --training-scenario demo_customer_churn \
     --target-output-path ./artifacts/churn \
     --test-size 0.25 \
     --random-state 22
   ```

3. Adjust `target_output_path` inside the JSON if you want artifacts to land in
   a different S3 bucket/folder.

### Validation Notes

- Every param file declares `data_validation` rules so the validation task can
  enforce minimum row counts and simple value ranges before training.
- `ingestion_config.object_storage.enabled` is set to `false` because these
  demos use local files. Swap to `true` and update the bucket/object key if you
  upload the CSVs to S3.

### Extending

To add your own case:

1. Drop a CSV into `demos/datasets/`.
2. Duplicate one of the JSON files in `demos/params/` and update the fields
   (especially `data_source`, `target_column`, and `model_name`).
3. Reference the new JSON when triggering Airflow.

This keeps demos consistent and lets teammates quickly showcase different model
types without editing code. 
