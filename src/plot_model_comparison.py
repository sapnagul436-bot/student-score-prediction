import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

metrics_file = ROOT / "outputs" / "metrics" / "model_comparison.csv"
output_file = ROOT / "outputs" / "figures" / "model_comparison.png"

df = pd.read_csv(metrics_file)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

metrics = [
    ("MAE", "Mean Absolute Error (MAE)", "lower"),
    ("RMSE", "Root Mean Squared Error (RMSE)", "lower"),
    ("R2", "R-squared (R²)", "higher")
]

for ax, (column, title, direction) in zip(axes, metrics):
    bars = ax.bar(df["Model"], df[column], width=0.6)

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

fig.suptitle(
    "Student Score Prediction - Model Performance Comparison",
    fontsize=15,
    fontweight="bold"
)

plt.tight_layout()

output_file.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_file, dpi=300, bbox_inches="tight")
plt.show()

print(f"\nChart saved successfully: {output_file}")
