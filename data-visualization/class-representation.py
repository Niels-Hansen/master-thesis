import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the data
df = pd.read_csv("../image-preparation/class_summarymasked.csv")

# 2. Extract genus and count occurrences
df["genus"] = df["Class"].str.split(r'[_ ]', regex=True).str[0]
genus_occurrences = df["genus"].value_counts().sort_values(ascending=False)

# 3. Generate distinct colors using nipy_spectral
num_genera = len(genus_occurrences)
cmap = plt.cm.nipy_spectral
colors = cmap(np.linspace(0, 1, num_genera))

# 4. Create the pie chart
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

# 5. Contrast-aware label coloring
for wedge, autotext in zip(wedges, autotexts):
    r, g, b, _ = wedge.get_facecolor()
    # luminance formula
    luminance = 0.299*r + 0.587*g + 0.114*b
    autotext.set_color('white' if luminance < 0.5 else 'black')
    autotext.set_fontsize(10)

# 6. Add a legend mapping colors to genus
ax.legend(
    wedges,
    genus_occurrences.index,
    title="Genus",
    loc="center left",
    bbox_to_anchor=(1, 0.5),
    fontsize=8,
)

# 7. Title and layout
#plt.title("Number of Classes per Genus")
fig.subplots_adjust(top=0.99, bottom=0.01, left=0.01, right=0.78)

# 8. Display and save
plt.show()
fig.savefig("genus_occurrences_readable_labels.png", dpi=300)
