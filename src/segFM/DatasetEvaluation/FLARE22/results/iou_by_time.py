import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Get current script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load CSV
csv_path = os.path.join(script_dir, "FLARE22.csv")
df = pd.read_csv(csv_path)

# Exclude models containing "BOB"
df_filtered = df[~df["Mode"].str.contains("BOB")].copy()

# Exclude MedSAM2_CTLesion
df_filtered = df_filtered[df_filtered["Model"] != "MedSAM2_CTLesion.pt"].copy()

# Replace "MedSAM2_" with "MS2_", VISTA3D with V3D, and SAM2.1_ with S2_
df_filtered["Model"] = df_filtered["Model"].str.replace("MedSAM2_", "MS2_", regex=False)

# Print unique models and modes for verification
print("Unique Models:", df_filtered["Model"].unique())
print("Unique Modes:", df_filtered["Mode"].unique())

# Group by model and mode, and calculate means
grouped = df_filtered.groupby(["Model", "Mode"]).agg({
    "Time": "mean",
    "IoU": "mean", 
    "NSD": "mean"
}).reset_index()

# Create combined model-mode identifier
grouped["Model_Mode"] = grouped["Model"] + " - " + grouped["Mode"]

# Sort by time for proper line plotting
grouped = grouped.sort_values("Time")

# Set up seaborn style and font size
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({'font.size': 16})

# Create single plot with two lines
plt.figure(figsize=(12, 4.6))

# Create short labels for x-ticks
grouped["Short_Label"] = [f"M{i+1}" for i in range(len(grouped))]

# Create mapping for legend
model_mapping = {short: full for short, full in zip(grouped["Short_Label"], grouped["Model_Mode"])}

# Plot both IoU and NSD on the same plot
plt.plot(grouped["Time"], grouped["IoU"], marker="o", linewidth=2, label="IoU")
plt.plot(grouped["Time"], grouped["NSD"], marker="s", linewidth=2, label="NSD")

# Set x-ticks with short labels and time
xtick_labels = [f"{short} ({time:.1f}s)" for short, time in zip(grouped["Short_Label"], grouped["Time"])]
plt.xticks(grouped["Time"], xtick_labels, rotation=50)

plt.xlabel("Time (s)")
plt.ylabel("Mean Score")

# Set more granular y-ticks with rounded values
y_min = min(grouped["IoU"].min(), grouped["NSD"].min())
y_max = max(grouped["IoU"].max(), grouped["NSD"].max())
ytick_values = [round(y_min + i * (y_max - y_min) / 5, 2) for i in range(6)]
plt.yticks(ytick_values)

# Add legend for IoU/NSD
plt.legend(loc='lower right')

# Add text box with model mapping on the right side
mapping_text = "\n".join([f"{short}: {full}" for short, full in model_mapping.items()])
plt.text(1.02, 1.0, mapping_text, transform=plt.gca().transAxes, fontsize=18,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=1.0, edgecolor='black'))

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(os.path.join(script_dir, "iou_by_time.png"), dpi=300)
plt.savefig(os.path.join(script_dir, "iou_by_time.svg"), dpi=300)
plt.savefig(os.path.join(script_dir, "iou_by_time.pdf"), dpi=300)


plt.show()
