import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


supported_classes = ["lung", "polyp", "liver", "kidney_right", "spleen", "aorta", "inferior_vena_cava",
"gallbladder", "esophagus", "stomach", "kidney_left", "prostate_and_uterus", "skin_lesion", "glioma", 
"optic_disc", "optic_cup", "heart_myocardium", "heart_ventricle_left", "heart_ventricle_right",
"heart_atrium_left"]


script_dir = os.path.dirname(__file__)
METRIC = "IoU"  # Metric to plot
FILE = "IMed361_bob"  # CSV file to load

# Translation dictionary for model names
model_name_map = {
    "MedSAM2_latest": "MedSAM2-latest",
}

# Load CSV
file_path = os.path.join(script_dir, f"{FILE}.csv")
df = pd.read_csv(file_path)

# Filter to supported classes only
df = df[df["Object"].isin(supported_classes)]

print(df)
# Save DF in new CSV
df.to_csv(os.path.join(script_dir, f"{FILE}_filtered.csv"), index=False)

# Format model names
df["FormattedModel"] = df["Model"].map(model_name_map).fillna(df["Model"])


# Create Label column
def create_label(row):
    if row["Mode"] == "point":
        return f"{row['Mode']} {int(row['n_pos'])}p{int(row['n_neg'])}n"
    elif row["Mode"] == "box":
        return f"{row['Mode']} {row['bbsize']}px"
    else:
        return f"{row['Mode']}"


df["Label"] = df.apply(create_label, axis=1)


# Create a group key for coloring
def create_group_key(row):
    if row["Mode"] == "point":
        return f"{row['Mode']} {int(row['n_pos'])}p{int(row['n_neg'])}n"
    elif row["Mode"] == "box":
        return f"{row['Mode']} {row['bbsize']}px"
    else:
        return row["Mode"]


df["GroupColorKey"] = df.apply(create_group_key, axis=1)



# Assign color palette to GroupColorKey
unique_groups = df["GroupColorKey"].unique()
palette = sns.color_palette("Set2", n_colors=len(unique_groups))
color_dict = dict(zip(unique_groups, palette))

# Plotting
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 2))

# Horizontal boxplot grouped by color key
sns.boxplot(
    data=df,
    y="Label",
    x=METRIC,
    hue="GroupColorKey",
    palette=color_dict,
    showfliers=False,
)

# Print the Average and Median for each group
grouped = df.groupby("GroupColorKey")[METRIC]
for name, group in grouped:
    avg = group.mean()
    median = group.median()
    print(f"{name}: Average {METRIC} = {avg:.3f}, Median {METRIC} = {median:.3f}")

plt.xlim(0, 1)

# Set y label font size
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)

# Final layout
plt.xlabel(METRIC)
plt.ylabel("")
plt.tight_layout()

plt.savefig(
    os.path.join(script_dir, f"{FILE}_{METRIC}_distribution.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.savefig(
    os.path.join(script_dir, f"{FILE}_{METRIC}_distribution.pdf"),
    dpi=300,
    bbox_inches="tight",
)
plt.savefig(
    os.path.join(script_dir, f"{FILE}_{METRIC}_distribution.svg"),
    dpi=300,
    bbox_inches="tight",
)

plt.show()
