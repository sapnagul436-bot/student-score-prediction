from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Project paths
root = Path(__file__).resolve().parent.parent
data_path = root / "data" / "raw" / "student-mat.csv"
figure_dir = root / "outputs" / "figures"
metrics_dir = root / "outputs" / "metrics"

figure_dir.mkdir(parents=True, exist_ok=True)
metrics_dir.mkdir(parents=True, exist_ok=True)

# Load dataset
df = pd.read_csv(data_path, sep=";")

# Calculate correlations for numerical columns
numeric_df = df.select_dtypes(include="number")
correlations = numeric_df.corr()["G3"].drop("G3")
correlations = correlations.sort_values(ascending=False)

print("\nCORRELATION WITH FINAL GRADE (G3):")
print(correlations.round(3))

# Save correlation values
correlations.to_csv(metrics_dir / "g3_correlations.csv")

# Create correlation chart
plt.figure(figsize=(10, 6))
sns.barplot(
    x=correlations.values,
    y=correlations.index,
    color="steelblue"
)

plt.title("Numerical Feature Correlation with G3")
plt.xlabel("Pearson Correlation")
plt.ylabel("Student Feature")
plt.axvline(0, color="black", linewidth=0.8)
plt.tight_layout()

output = figure_dir / "g3_correlation.png"
plt.savefig(output, dpi=150)
plt.show()

print("\nChart saved:", output)
print("Correlation values saved successfully.")