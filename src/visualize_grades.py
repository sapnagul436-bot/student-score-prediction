from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent.parent
df = pd.read_csv(root / "data" / "raw" / "student-mat.csv", sep=";")

output = root / "docs" / "grade_distribution.png"
output.parent.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(9, 5))
plt.hist(df["G3"], bins=range(0, 22), edgecolor="black")
plt.title("Distribution of Final Grades (G3)")
plt.xlabel("Final grade (0–20)")
plt.ylabel("Number of students")
plt.xticks(range(0, 21, 2))
plt.tight_layout()
plt.savefig(output, dpi=150)
plt.show()

print("Chart saved:", output)