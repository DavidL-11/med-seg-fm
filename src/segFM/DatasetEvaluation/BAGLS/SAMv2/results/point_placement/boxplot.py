import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Get current script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load CSV (checkpoints)
csv1_path = os.path.join(script_dir, "BAGLS_placement.csv")
csv1 = pd.read_csv(csv1_path)

# Seaborn style
sns.set_theme(style="whitegrid", context="talk")

# Boxplot of IoU values per model, split by Mode
plt.figure(figsize=(5, 5))
sns.boxplot(
    data=csv1,
    x="Model",
    y="IoU",
    hue="Prompt Finder",
    showfliers=False,
    palette="Set3"
)

plt.xlabel("")
plt.ylabel("IoU", fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

# Move legend outside for clarity
plt.legend(title="Mode", bbox_to_anchor=(1.05, 1), loc="upper left")

plt.tight_layout()

# Save as SVG, PNG and PDF
plt.savefig(os.path.join(script_dir, "pointplacement.svg"), dpi=300)
plt.savefig(os.path.join(script_dir, "pointplacement.png"), dpi=300)
plt.savefig(os.path.join(script_dir, "pointplacement.pdf"), dpi=300)
plt.show()
