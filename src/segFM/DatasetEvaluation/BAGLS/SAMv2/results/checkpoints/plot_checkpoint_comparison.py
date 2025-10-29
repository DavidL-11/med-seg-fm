# REQUIREMENTS: pandas, matplotlib, seaborn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Get current script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load CSVs
csv1_path = os.path.join(script_dir, "BAGLS_checkpoints.csv")
csv2_path = os.path.join(script_dir, "BAGLS_time_1k.csv")

csv1 = pd.read_csv(csv1_path)
csv2 = pd.read_csv(csv2_path)

# Remove all rows with model "SAM-Med2D"
csv1 = csv1[csv1["Model"] != "SAM-Med2D"]
csv2 = csv2[csv2["Model"] != "SAM-Med2D"]

# Merge on Model
merged = pd.merge(csv1, csv2, on="Model", how="left")

# Compute average IoU per (Model, Mode)
avg_dsc = merged.groupby(["Model", "Mode", "Time"], as_index=False)["IoU"].mean()

# Sort by time
avg_dsc = avg_dsc.sort_values(by="Time")

# Map time info for xticks
model_time_map = dict(zip(csv2["Model"], csv2["Time"]))

# Use seaborn styling
sns.set_theme(style="whitegrid", context="talk")

# Plot using seaborn with Time on x-axis
plt.figure(figsize=(12, 8))
sns.lineplot(data=avg_dsc, x="Time", y="IoU", hue="Mode", marker="o", linewidth=2)

# Custom x-tick labels with model names and time
unique_times = sorted(avg_dsc["Time"].unique())
xtick_labels = []
for time in unique_times:
    # Get the model name for this time
    model_for_time = avg_dsc[avg_dsc["Time"] == time]["Model"].iloc[0]
    # Replace model names for shorter labels
    model_display = model_for_time.replace("MedSAM2", "MS2").replace("SAM2.1", "S2")
    xtick_labels.append(f"{model_display} ({time}s)")

plt.xticks(unique_times, xtick_labels, rotation=0, fontsize=12, fontweight="bold")


# Disable x labels
plt.xlabel("")
# Rotate x-tick labels for better readability
plt.xticks(rotation=90, ha='right')
plt.ylabel("Average IoU", fontsize=14, fontweight="bold")
plt.legend(title="Prompt Type", fontsize=12, title_fontsize=13)
plt.tight_layout()

plt.savefig(os.path.join(script_dir, "checkpoint_comparison_time.svg"), dpi=300)
plt.savefig(os.path.join(script_dir, "checkpoint_comparison_time.png"), dpi=300)
plt.savefig(os.path.join(script_dir, "checkpoint_comparison_time.pdf"), dpi=300)

plt.show()
