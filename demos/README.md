# Demo Dataset Reference

Each folder under `demos/` contains a synthetic dataset plus an Airflow parameter payload (`params_*.json`) that can be used to trigger the `ml_dynamic_pipeline_with_ingestion_and_training` DAG. This document describes the schema of every dataset and explains why the accompanying model configuration is a good fit.

---

## 1. Driver Safety (Classification)

- **Files**: `driver_safety/driver_safety.csv`, `driver_safety/params_driver_safety.json`
- **Training features**
  - `weekly_miles`, `avg_speed_mph`, `incident_count`, `service_years`, `safety_score`.
  - `driver_id` is an identifier only and is **not** used as a training feature.
- **Label**
  - `target_high_risk` — binary flag (1=high risk). This is the column referenced by the DAG’s `target_column`.
- **Schema reference**
  | Column | Type | Role |
  | --- | --- | --- |
  | `driver_id` | string | Identifier (ignored during training). |
  | `weekly_miles` | float | Feature. |
  | `avg_speed_mph` | float | Feature. |
  | `incident_count` | int | Feature. |
  | `service_years` | int | Feature. |
  | `safety_score` | float | Feature. |
  | `target_high_risk` | binary | **Label** |
- **Model choice: `random_forest_classifier`**  
  Driver risk depends on nonlinear interactions (e.g., high mileage + high average speed amplifies risk, but tenure and safety score temper it). Random forests capture those feature interactions and handle mixed numeric ranges without feature scaling, making them robust for this telemetry-like dataset. The hyperparameters (200 trees, depth 12, min leaf 2) balance interpretability and performance for ~400 rows.

---

## 2. Consumer Loan Risk (Classification)

- **Files**: `loan_risk/consumer_loan_risk.csv`, `loan_risk/params_loan_risk.json`
- **Training features**
  - `annual_income`, `credit_score`, `debt_to_income_ratio`, `current_loan_amount`, `late_payments`, `has_cosigner`.
  - `application_id` is an identifier and excluded from the model matrix.
- **Label**
  - `loan_defaulted` — binary default indicator.
- **Schema reference**
  | Column | Type | Role |
  | --- | --- | --- |
  | `application_id` | string | Identifier (ignored). |
  | `annual_income` | float | Feature. |
  | `credit_score` | int | Feature. |
  | `debt_to_income_ratio` | float | Feature. |
  | `current_loan_amount` | float | Feature. |
  | `late_payments` | int | Feature. |
  | `has_cosigner` | binary | Feature. |
  | `loan_defaulted` | binary | **Label** |
- **Model choice: `logistic_regression`**  
  Credit risk is often controlled by linear decision boundaries (e.g., higher DTI and lower credit scores increase default odds). Logistic regression provides calibrated probabilities, is easy to explain to business stakeholders, and adapts well to the relatively low-dimensional, mostly numeric feature space. Regularization (`C=0.8`) and `lbfgs` solver keep the model stable over ~450 rows while delivering interpretable coefficients.

---

## 3. Regional Energy Demand (Regression)

- **Files**: `energy_demand/regional_energy_load.csv`, `energy_demand/params_energy_demand.json`
- **Training features**
  - `avg_temperature_c`, `relative_humidity_pct`, `day_of_week`, `is_holiday`, `previous_day_load_mwh`, `renewable_share_pct`.
  - `site_id` is metadata (ignored).
- **Label**
  - `total_load_mwh` — numeric energy demand target.
- **Schema reference**
  | Column | Type | Role |
  | --- | --- | --- |
  | `site_id` | string | Identifier (ignored). |
  | `avg_temperature_c` | float | Feature. |
  | `relative_humidity_pct` | float | Feature. |
  | `day_of_week` | int | Feature. |
  | `is_holiday` | binary | Feature. |
  | `previous_day_load_mwh` | float | Feature. |
  | `renewable_share_pct` | float | Feature. |
  | `total_load_mwh` | float | **Label** |
- **Model choice: `random_forest_regressor`**  
  Energy demand exhibits nonlinear seasonality (temperature, humidity, day-of-week, holidays) plus lagged dependencies. A random forest regressor captures those nonlinear effects and handles feature interactions (e.g., temperature × humidity × holiday) without manual feature engineering. Depth 14 with 300 estimators is sufficient to model ~420 records while keeping variance in check via `min_samples_leaf=2`.

---

Use these references when crafting new demo cases or when you need to understand which fields must be present for validation and model training. You can always tailor the model name or hyperparameters in each `params_*.json` if you want to experiment with alternative estimators.***
