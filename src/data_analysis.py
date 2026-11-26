from pathlib import Path
import pandas as pd

# Load the original dataset
data_path = (
    Path(__file__).resolve().parent.parent
    / "data" / "raw" / "student-mat.csv"
)
df = pd.read_csv(data_path, sep=";")

print("1. DATASET INFORMATION")
print("Shape:", df.shape)
print("Duplicate rows:", df.duplicated().sum())

print("\n2. NUMERICAL STATISTICS")
print(df.describe().round(2).to_string())

print("\n3. CATEGORICAL STATISTICS")
print(df.describe(include="object").to_string())

print("\n4. FINAL GRADE (G3)")
print("Mean:", round(df["G3"].mean(), 2))
print("Median:", df["G3"].median())
print("Minimum:", df["G3"].min())
print("Maximum:", df["G3"].max())
print("Standard deviation:", round(df["G3"].std(), 2))