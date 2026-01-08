# Linear Regression Demo (Insurance)

Dataset: Medical insurance cost data (common "insurance.csv" dataset).

Target (prediction): `charges` (insurance cost).

Features used:
- `age`
- `bmi`
- `children`

Notes:
- Categorical columns (`sex`, `smoker`, `region`) are dropped in the demo pipeline.
- The BentoML service expects numeric feature rows in the same order as above.
