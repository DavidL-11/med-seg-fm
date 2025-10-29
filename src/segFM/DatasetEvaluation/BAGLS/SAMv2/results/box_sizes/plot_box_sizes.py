import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.ticker as mticker

# Get current script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load CSV1
csv1_path = os.path.join(script_dir, "BAGLS_boxes.csv")
csv1 = pd.read_csv(csv1_path)

# Only keep Mode = "box"

box_df = csv1[csv1["Mode"] == "box"].copy()

# Compute average IoU per bbsize
avg_metrics = box_df.groupby("bbsize", as_index=False).agg({"IoU": "mean", "NSD": "mean"})

# Seaborn style
sns.set_theme(style="whitegrid", context="talk")

# Plot
plt.figure(figsize=(10, 6))
plt.plot(avg_metrics["bbsize"], avg_metrics["IoU"], marker="o", linewidth=2, color="blue", label="Average IoU")
plt.plot(avg_metrics["bbsize"], avg_metrics["NSD"], marker="s", linewidth=2, color="green", label="Average NSD")
plt.xlabel("Bounding box size difference (log scale)", fontsize=18)
plt.ylabel("Score", fontsize=18)
plt.grid(True, linestyle="--", linewidth=0.7)
plt.tight_layout()
plt.legend(fontsize=14)

plt.xscale('symlog', linthresh=1.0)
plt.grid(True, which='both', linestyle='--')
# Force ticks to display as plain integers
plt.gca().xaxis.set_major_formatter(mticker.ScalarFormatter())

plt.savefig(os.path.join(script_dir, f"bagls_box_size.svg"), dpi=300)
plt.savefig(os.path.join(script_dir, f"bagls_box_size.png"), dpi=300)
plt.savefig(os.path.join(script_dir, f"bagls_box_size.pdf"), dpi=300)

plt.show()
