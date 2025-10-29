import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- Parameters ---
script_dir = os.path.dirname(os.path.abspath(__file__))
METRIC = "IoU"
PLOT_TYPE = sns.barplot
csv_file = os.path.join(script_dir, "msd.csv")

# --- Load CSV ---
df = pd.read_csv(csv_file)

# --- Exclude lines where Mode includes "BOB" or "YOLO" or whatever I dont want to plot ---
df = df[~df["Mode"].str.contains("YOLO")]

# Remove all Objects "edema", "non-enhancing tumor", "enhancing tumor"
df = df[~df["Object"].isin(["edema", "non-enhancing tumor", "enhancing tumor"])]

# --- Multiply DSC by NSD and write to DSC column ---
# df[METRIC] = df["DSC"] * df["NSD"]

# --- Calculate average execution time for each Model-Mode combination ---
avg_times = df.groupby(["Model", "Mode"])["Time"].mean().reset_index()
avg_times["Time_str"] = avg_times["Time"].round(2).astype(str) + "s"

# --- Combine Model and Mode into one column with average time ---
df["Model_Mode"] = df["Model"].astype(str) + " (" + df["Mode"].astype(str) + ")"

# Create legend labels with average times
legend_labels = []
for _, row in avg_times.iterrows():
    model_mode = f"{row['Model']} ({row['Mode']})"
    legend_labels.append(f"{model_mode} - {row['Time_str']}")

# Create mapping from Model_Mode to legend label
model_mode_to_label = {}
for _, row in avg_times.iterrows():
    model_mode = f"{row['Model']} ({row['Mode']})"
    model_mode_to_label[model_mode] = f"{model_mode} - {row['Time_str']}"

# --- Create horizontal boxplot ---
plt.figure(figsize=(16, 8))

# Set custom color palette
custom_palette = [
                  "#B69917", # SAM2
                  "#BD1A25", "#D94141", # MedSAM2 latest
                  "#40418E", # SAM-Med3D
                  "#087613", "#44B04F", "#76F182", # VISTA3D
                ]

# Create horizontal boxplot
PLOT_TYPE(
    data=df,
    x="Object",
    y=METRIC,
    hue="Model_Mode",
    palette=custom_palette,
)

# --- Formatting ---
plt.xlabel("")
plt.ylabel(METRIC, fontsize=14)

# Get the legend and update labels with average times
legend = plt.legend(loc="lower left")
handles = legend.legend_handles
labels = [model_mode_to_label[label.get_text()] for label in legend.get_texts()]
plt.legend(handles, labels, loc="lower left", bbox_to_anchor=(0, -0.7), fontsize=18)

plt.tight_layout()


# Add vertical lines between all y-ticks to differentiate objects
for i in range(len(df["Object"].unique()) - 1):
    plt.axvline(x=i + 0.5, color='gray', linestyle='--', linewidth=1)


# Set y-ticks font size
plt.tick_params(axis='x', labelsize=18)

# Adjust bottom margin to prevent x-label cutoff
plt.subplots_adjust(bottom=0.4)

# Save as SVG, PNG and PDF
plt.savefig(os.path.join(script_dir, f"{str(PLOT_TYPE.__name__)}_MSD.svg"), dpi=300)
plt.savefig(os.path.join(script_dir, f"{str(PLOT_TYPE.__name__)}_MSD.png"), dpi=300)
plt.savefig(os.path.join(script_dir, f"{str(PLOT_TYPE.__name__)}_MSD.pdf"), dpi=300)

plt.show()
