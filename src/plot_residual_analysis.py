
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]

data_file = ROOT / "data" / "raw" / "student-mat.csv"
figure_file = ROOT / "outputs" / "figures" / "residual_analysis.png"
errors_file = ROOT / "outputs" / "metrics" / "prediction_errors.csv"

# 1. Load dataset
df = pd.read_csv(data_file, sep=";")

# 2. Separate features and target
X = df.drop(columns=["G3"])
y = df["G3"]

# 3. Reproduce original train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# 4. Identify numerical and categorical features
numeric = X_train.select_dtypes(include="number").columns
categorical = X_train.select_dtypes(exclude="number").columns

# 5. Reproduce original preprocessing
preprocessing = ColumnTransformer([
    ("numeric", SimpleImputer(strategy="median"), numeric),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical)
])

# 6. Build Late Linear Regression model
model = Pipeline([
    ("preprocessing", preprocessing),
    ("regression", LinearRegression())
])

# 7. Train and predict
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 8. Calculate residuals
residuals = y_test - y_pred

# 9. Evaluate model
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print("\nLATE LINEAR REGRESSION PERFORMANCE")
print(f"MAE:  {mae:.6f}")
print(f"RMSE: {rmse:.6f}")
print(f"R2:   {r2:.6f}")

# 10. Save prediction errors
results = pd.DataFrame({
    "Actual_Score": y_test.values,
    "Predicted_Score": y_pred,
    "Residual": residuals.values,
    "Absolute_Error": abs(residuals.values)
})

errors_file.parent.mkdir(parents=True, exist_ok=True)
results.to_csv(errors_file, index=False)

# 11. Create residual visualizations
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].scatter(y_pred, residuals, alpha=0.7)
axes[0].axhline(y=0, linestyle="--")
axes[0].set_title("Residuals vs Predicted Scores")
axes[0].set_xlabel("Predicted Score")
axes[0].set_ylabel("Residual")
axes[0].grid(alpha=0.3)

axes[1].hist(residuals, bins=15, edgecolor="black")
axes[1].axvline(x=0, linestyle="--")
axes[1].set_title("Distribution of Residuals")
axes[1].set_xlabel("Residual")
axes[1].set_ylabel("Frequency")
axes[1].grid(axis="y", alpha=0.3)

fig.suptitle(
    "Student Score Prediction - Residual Analysis",
    fontsize=14,
    fontweight="bold"
)

plt.tight_layout()

figure_file.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(figure_file, dpi=300, bbox_inches="tight")
plt.show()

print("\nResidual analysis completed successfully!")
print(f"Residual plot: {figure_file}")
print(f"Prediction errors: {errors_file}")
print(f"Mean residual: {residuals.mean():.6f}")
print(f"Maximum absolute error: {abs(residuals).max():.6f}")
