from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

root = Path(__file__).resolve().parent.parent
df = pd.read_csv(root / "data" / "raw" / "student-mat.csv", sep=";")

y = df["G3"]
X = df.drop(columns=["G3"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

results = []

def evaluate(name, model, X_train, X_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    results.append({
        "Model": name,
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": mean_squared_error(y_test, predictions) ** 0.5,
        "R2": r2_score(y_test, predictions)
    })

# Simple baseline: always predict the training mean
evaluate(
    "Mean Baseline",
    DummyRegressor(strategy="mean"),
    X_train,
    X_test
)

# Compare early and late prediction
for name, excluded in [
    ("Early Linear Regression", ["G1", "G2"]),
    ("Late Linear Regression", [])
]:
    train = X_train.drop(columns=excluded)
    test = X_test.drop(columns=excluded)

    numeric = train.select_dtypes(include="number").columns
    categorical = train.select_dtypes(exclude="number").columns

    preprocessing = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical)
    ])

    model = Pipeline([
        ("preprocessing", preprocessing),
        ("regression", LinearRegression())
    ])

    evaluate(name, model, train, test)

metrics = pd.DataFrame(results)
print("\nMODEL COMPARISON:")
print(metrics.round(3).to_string(index=False))

output = root / "outputs" / "metrics" / "model_comparison.csv"
output.parent.mkdir(parents=True, exist_ok=True)
metrics.to_csv(output, index=False)

print("\nResults saved:", output)



