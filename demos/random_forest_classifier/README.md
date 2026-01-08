# Random Forest Classifier Demo (Titanic)

Dataset: Titanic survival dataset.

Target (prediction): `Survived` (0 or 1).

Features used:
- `Pclass`
- `Age`
- `SibSp`
- `Parch`
- `Fare`

Notes:
- The pipeline drops non-numeric or ID columns: `PassengerId`, `Name`, `Ticket`, `Cabin`, `Sex`, `Embarked`.
- The BentoML service expects numeric feature rows in the same order as above.
