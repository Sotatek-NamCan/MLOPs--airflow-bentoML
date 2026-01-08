# Random Forest Regressor Demo (House Prices)

Dataset: California housing dataset ("house.csv").

Target (prediction): `median_house_value`.

Features used:
- `longitude`
- `latitude`
- `housing_median_age`
- `total_rooms`
- `total_bedrooms`
- `population`
- `households`
- `median_income`

Notes:
- The pipeline drops the categorical `ocean_proximity` column.
- The BentoML service expects numeric feature rows in the same order as above.
