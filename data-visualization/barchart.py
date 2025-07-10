import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = {
    "Metric":            ["Accuracy", "Precision", "Recall", "F1"],
    "EfficientNetV2-S":  [0.99995, 0.99991, 0.99994, 0.99993],
    "ResNeXt101":        [0.99991, 0.99990, 0.99990, 0.99990],
    "ViT-B/16":          [0.99491, 0.99511, 0.99493, 0.99497],
    "YOLO":  [0.99950, np.nan,    np.nan,    np.nan],
}


df = pd.DataFrame(data).melt(
    id_vars="Metric",
    var_name="Model",
    value_name="Score"
)


colors = ["#AEC6CF", "#FFB347", "#77DD77", "#FF6961"]  

metrics = data["Metric"]
models  = ["EfficientNetV2-S", "ResNeXt101", "ViT-B/16", "YOLO"]
bar_w   = 0.18
x       = np.arange(len(metrics))

fig, ax = plt.subplots(figsize=(8, 4.5))
for i, (model, color) in enumerate(zip(models, colors)):
    scores = df[df["Model"] == model].set_index("Metric").loc[metrics, "Score"]
    ax.bar(
        x + (i - (len(models)-1)/2) * bar_w,
        scores,
        width=bar_w,
        label=model,
        color=color
    )

ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11)
ax.set_ylabel("Score", fontsize=12)
ax.set_ylim(0.90, 1.00)
ax.set_yticks(np.arange(0.90, 1.01, 0.01))
ax.set_title("Performance Comparison of Models (30-minute interval)", fontsize=14)
ax.legend(frameon=True, edgecolor='black', fontsize=10)
ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)

plt.tight_layout()
plt.show()
