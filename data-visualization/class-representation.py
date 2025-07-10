import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("../image-preparation/class_summarymasked.csv")

df["genus"] = df["Class"].str.split(r'[_ ]', regex=True).str[0]
genus_occurrences = df["genus"].value_counts().sort_values(ascending=False)
num_genera = len(genus_occurrences)
cmap = plt.cm.nipy_spectral
colors = cmap(np.linspace(0, 1, num_genera))

fig, ax = plt.subplots(figsize=(8, 8))
total = genus_occurrences.sum()
wedges, texts, autotexts = ax.pie(
    genus_occurrences,
    labels=None,
    autopct=lambda pct: str(int(pct * total / 100)),
    startangle=90,
    colors=colors,
    wedgeprops={"edgecolor": "white"}
)
ax.axis("equal")

for wedge, autotext in zip(wedges, autotexts):
    r, g, b, _ = wedge.get_facecolor()
    luminance = 0.299*r + 0.587*g + 0.114*b
    autotext.set_color('white' if luminance < 0.5 else 'black')
    autotext.set_fontsize(10)

ax.legend(
    wedges,
    genus_occurrences.index,
    title="Genus",
    loc="center left",
    bbox_to_anchor=(1, 0.5),
    fontsize=8,
)


#plt.title("Number of Classes per Genus")
fig.subplots_adjust(top=0.99, bottom=0.01, left=0.01, right=0.78)

plt.show()
fig.savefig("genus_occurrences.png", dpi=300)
