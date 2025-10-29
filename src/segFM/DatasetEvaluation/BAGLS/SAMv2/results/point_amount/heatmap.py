import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

METRIC = "IoU"  # Change to "DSC", "NSD", or "IoU"
model_name = "Tiny" 
OBJECT = "Both"  # Change to "Vocal Folds", "Glottis", or "Both"
base_path = os.path.dirname(__file__)

# Path to your CSV (relative to script location)
csv_path = os.path.join(base_path, "BAGLS_point_heatmap.csv")

# Load data
df = pd.read_csv(csv_path)

# --- New step: normalize Object column ---
df["Object"] = df["Object"].replace(
    {"Left Vocal Fold": "Vocal Folds", "Right Vocal Fold": "Vocal Folds"}
)

if OBJECT != "Both":
    df = df[df["Object"] == OBJECT]

df_model = df[df["Model"] == model_name]


# Pivot into matrix form: rows = n_pos, cols = n_neg, values = IoU
heatmap_data = df_model.pivot_table(index="n_pos", columns="n_neg", values=METRIC, aggfunc="mean")

heatmap_data = heatmap_data.sort_index(ascending=False)

plt.figure(figsize=(6, 5))
ax = sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="viridis", cbar_kws={'label': METRIC})

# Set colorbar label size (METRIC)
cbar = ax.collections[0].colorbar
cbar.set_label(METRIC, fontsize=16)

# Format the annotations to remove leading zero
for text in ax.texts:
    text.set_text(text.get_text().lstrip('0'))

ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)

plt.xlabel("Number of Negative Prompts", fontsize=14)
plt.ylabel("Number of Positive Prompts", fontsize=14)
plt.tight_layout()

# Save as SVG, PNG and PDF
plt.savefig(os.path.join(base_path, f"{OBJECT.replace(' ', '_')}_heatmap.svg"), dpi=300)
plt.savefig(os.path.join(base_path, f"{OBJECT.replace(' ', '_')}_heatmap.png"), dpi=300)
plt.savefig(os.path.join(base_path, f"{OBJECT.replace(' ', '_')}_heatmap.pdf"), dpi=300)

plt.show()
