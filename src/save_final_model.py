
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Project paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(
    project_root,
    "data",
    "raw",
    "student-mat.csv"
)

model_dir = os.path.join(project_root, "models")
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(
    model_dir,
    "student_score_prediction_pipeline.joblib"
)

# Load dataset
df = pd.read_csv(data_path, sep=";")

# Separate features and target
X = df.drop(columns=["G3"])
y = df["G3"]

# Use the same train-test split as the original experiment
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Identify numeric and categorical features
numeric_features = X_train.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object"]
).columns.tolist()

# Preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessing = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Create the complete machine learning pipeline
model = Pipeline(steps=[
    ("preprocessing", preprocessing),
    ("regression", LinearRegression())
])

# Train and evaluate using the original split
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("HOLDOUT TEST PERFORMANCE")
print("------------------------")
print("MAE:", mean_absolute_error(y_test, predictions))
print("RMSE:", mean_squared_error(y_test, predictions) ** 0.5)
print("R2:", r2_score(y_test, predictions))

# Retrain the final pipeline using the complete dataset
model.fit(X, y)

# Save the complete pipeline
joblib.dump(model, model_path)

print("\nFINAL MODEL SAVED")
print("-----------------")
print(model_path)

# Load the saved pipeline and verify predictions
loaded_model = joblib.load(model_path)

sample_predictions = loaded_model.predict(X.head())

print("\nMODEL LOADING TEST")
print("------------------")
print("First 5 predictions:", sample_predictions)
print("Model loaded and predictions generated successfully.")
