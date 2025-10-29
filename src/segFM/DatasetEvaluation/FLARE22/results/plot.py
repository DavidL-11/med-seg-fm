import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

METRIC = "IoU"
PLOT_TYPE = sns.barplot

# --- Parameters ---
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_file = os.path.join(script_dir, "FLARE22.csv")

# --- Load CSV ---
df = pd.read_csv(csv_file)

# --- Format model names - remove .pt, replace "Lesion" with "" ---
df["Model"] = df["Model"].str.replace(".pt", "", regex=False)
df["Model"] = df["Model"].str.replace("Lesion", "", regex=False)
df["Model"] = df["Model"].str.replace("sam2.1_hiera_tiny", "SAM2.1 Tiny", regex=False)

# --- Combine Model and Mode into one column ---
df["Model_Mode"] = df["Model"].astype(str) + " (" + df["Mode"].astype(str) + ")"

# --- Print mean/median metrics per model-mode over all objects ---
print(f"\n=== {METRIC} Statistics by Model-Mode (across all objects) ===")
print("-" * 95)

# # If "BOB" in mode, remove all objects with a mean DSC of 0.0
# if any(df["Model_Mode"].str.contains("BOB")):
#     bob_means = df[df["Model_Mode"].str.contains("BOB")].groupby("Object")[METRIC].mean()
#     objects_to_remove = bob_means[bob_means == 0.0].index.tolist()
#     if objects_to_remove:
#         print(f"Removing objects with mean {METRIC} of 0.0 for BOB: {objects_to_remove}")
#         df = df[~df["Object"].isin(objects_to_remove)]

# Calculate mean and median for each model-mode combination, including time
stats_df = df.groupby("Model_Mode").agg({
    METRIC: ['mean', 'median', 'std', 'count'],
    'Time': ['mean']
}).round(4)

# Flatten column names
stats_df.columns = [f'{col[0]}_{col[1]}' if col[1] != '' else col[0] for col in stats_df.columns]
stats_df = stats_df.sort_values(f'{METRIC}_mean', ascending=False)

print(f"{'Model-Mode':<40} {'Mean':<8} {'Median':<8} {'Std':<8} {'Count':<6} {'Avg Time':<10}")
print("-" * 95)
for model_mode, row in stats_df.iterrows():
    print(f"{model_mode:<40} {row[f'{METRIC}_mean']:<8.4f} {row[f'{METRIC}_median']:<8.4f} {row[f'{METRIC}_std']:<8.4f} {int(row[f'{METRIC}_count']):<6} {row['Time_mean']:<10.2f}")

print(f"\nOverall {METRIC} statistics:")
print(f"Overall Mean: {df[METRIC].mean():.4f}")
print(f"Overall Median: {df[METRIC].median():.4f}")
print(f"Overall Std: {df[METRIC].std():.4f}")
print(f"Total samples: {len(df)}")
print(f"Overall Average Time: {df['Time'].mean():.2f}")
print("-" * 95)

# --- Create horizontal boxplot ---
plt.figure(figsize=(16, 7))

# Create horizontal boxplot
PLOT_TYPE(
    data=df,
    x="Object",
    y=METRIC,
    hue="Model_Mode",
    palette=["#0dbc96", "#1f77b4", "#56a4dc", "#076c07", "#b50f0f", "#e25050"],
)

# --- Formatting ---
plt.xlabel("")
plt.ylabel(METRIC, fontsize=16)
plt.legend(title="Model (Mode)", loc="lower left", fontsize=12, title_fontsize=14)
plt.tight_layout()


# Add vertical lines between all y-ticks to differentiate objects
for i in range(len(df["Object"].unique()) - 1):
    plt.axvline(x=i + 0.5, color='gray', linestyle='--', linewidth=1)


# Set y-ticks font size
plt.tick_params(axis='x', labelsize=16, rotation=50)

# Adjust bottom margin to prevent x-label cutoff
plt.subplots_adjust(bottom=0.3)

# Save as SVG, PNG and PDF
plt.savefig(os.path.join(script_dir, f"{PLOT_TYPE.__name__}_FLARE22_{METRIC}.svg"), dpi=300)
plt.savefig(os.path.join(script_dir, f"{PLOT_TYPE.__name__}_FLARE22_{METRIC}.png"), dpi=300)
plt.savefig(os.path.join(script_dir, f"{PLOT_TYPE.__name__}_FLARE22_{METRIC}.pdf"), dpi=300)

plt.show()
