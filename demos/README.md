## Demo Assets Overview

This folder contains ready-made datasets and parameter presets that mirror the
`driver_safe_param.json` structure so you can trigger the DAG or the standalone
CLI without hand-crafting JSON payloads. Each scenario pairs a dataset (CSV,
JSON, or Parquet) with the model family it was designed for.

### Folder Layout

- `driver_safety/`: Original driver high-risk classification sample plus relaxed
  validation settings (`driver_safety.csv`, `params_driver_safety*.json`).
- `datasets/`: Synthetic multi-hundred-row tabular files for additional demos.
- `params/`: JSON payloads that follow the same schema as `driver_safe_param.json`
  and reference the datasets above.

### Available Scenarios

| Model | Dataset | Param file | Target column | Notes |
| --- | --- | --- | --- | --- |
| `random_forest_classifier` | `demos/datasets/fruit_quality_demo.csv` | `demos/params/random_forest_classifier_fruit_quality.json` | `fruit_type` | Fruit grading scenario with weight/length/color features for orchard QA. |
| `logistic_regression` | `demos/datasets/coffee_promo_behavior_demo.json` | `demos/params/logistic_regression_coffee_promo.json` | `bought_coffee` | JSON records that capture discount, visit history, and rain flags for a café promo test. |
| `linear_regression` | `demos/datasets/monthly_electricity_usage_demo.parquet` | `demos/params/linear_regression_electricity_usage.json` | `electricity_kwh_month` | Household electricity forecasting with AC usage and weather inputs (stored as Parquet). |
| `random_forest_regressor` | `demos/datasets/restaurant_orders_daily_demo.csv` | `demos/params/random_forest_regressor_restaurant_orders.json` | `orders_count` | Daily order regression that mixes weekday, promo, rain, temperature, and nearby events. |
| `random_forest_classifier` (driver safety) | `demos/driver_safety/driver_safety.csv` | `demos/driver_safety/params_driver_safety*.json` | `target_high_risk` | Same dataset used in docs; the relaxed variant reduces validation strictness. |

### Using a Demo Param File

1. Pick a scenario from the table.
2. Copy the corresponding JSON into your DAG run configuration (Airflow UI →
   “Trigger DAG” → “Config”) or run locally:

   ```bash
   python -m pipeline_worker.cli.train_model \
     --train-data-path demos/datasets/fruit_quality_demo.csv \
     --target-column fruit_type \
     --model-name random_forest_classifier \
     --model-version 1 \
     --hyperparameters "$(Get-Content demos/params/random_forest_classifier_fruit_quality.json | jq '.hyperparameters')" \
     --training-scenario demo_fruit_quality \
     --target-output-path ./artifacts/fruit_quality \
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

1. Drop a dataset (CSV/JSON/Parquet) into `demos/datasets/`.
2. Duplicate one of the JSON files in `demos/params/` and update the fields
   (especially `data_source`, `target_column`, and `model_name`).
3. Reference the new JSON when triggering Airflow.

This keeps demos consistent and lets teammates quickly showcase different model
types without editing code. 
