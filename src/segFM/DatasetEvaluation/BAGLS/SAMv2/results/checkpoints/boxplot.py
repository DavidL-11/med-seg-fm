import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Get current script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load CSV (checkpoints)
csv1_path = os.path.join(script_dir, "BAGLS_checkpoints.csv")
csv1 = pd.read_csv(csv1_path)

# Remove "SAM2.1" from all model names in csv1
csv1["Model"] = csv1["Model"].str.replace("SAM2.1_", "", regex=False)
csv1["Model"] = csv1["Model"].str.replace("SAM2.1 ", "", regex=False)

# Remove rows containing "BOB" in the Prompt_Finder column
csv1 = csv1[~csv1["Prompt Finder"].str.contains("BOB", na=False)]

# Seaborn style
sns.set_theme(style="whitegrid", context="talk")

# Boxplot of IoU values per model, split by Mode
plt.figure(figsize=(10, 5.5))
sns.boxplot(
    data=csv1,
    x="Model",
    y="IoU",
    hue="Mode",
    showfliers=False,
    palette="Set3"
)

plt.xlabel("")
plt.ylabel("IoU", fontsize=15, fontweight="bold")
plt.xticks(rotation=45, ha="right", fontsize=15, fontweight="bold")
plt.yticks(fontsize=15, fontweight="bold")

# Move legend outside for clarity
plt.legend(title="Mode", bbox_to_anchor=(1.05, 1), loc="upper left")

plt.tight_layout()

# Save as SVG, PNG and PDF
plt.savefig(os.path.join(script_dir, "checkpoint_comparison.svg"), dpi=300)
plt.savefig(os.path.join(script_dir, "checkpoint_comparison.png"), dpi=300)
plt.savefig(os.path.join(script_dir, "checkpoint_comparison.pdf"), dpi=300)
plt.show()
