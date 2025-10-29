import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

script_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(script_dir, "MedSegBench.csv")
output_path = os.path.join(script_dir, "baseline_table_s.tex")

METRIC = 'IoU'  # Metric to plot
FILE = 'MedSegBench'  # CSV file to load

# Translation dictionary for model names
model_name_map = {
    "MedSAM2_latest": "MedSAM2-latest",
}

df = pd.read_csv(input_path)

# Format model names
df['FormattedModel'] = df['Model'].map(model_name_map).fillna(df['Model'])

supported_classes = [
    "Glottis",
    "Left Vocal Cord",
    "Right Vocal Cord",
    "Lung",
    #"Kidney",
    #"Polyp",
    #"Tool",
    #"Organ",
    "Neoplastic Polyp"
    "Liver",
    "Right Kidney",
    "Spleen",
    "Mitochondria",
    "Aorta",
    "Inferior Vena Cava",
    "Pharynx",
    "Fetal Head",
    "Gallbladder",
    "Esophagus",
    "Stomach",
    "Tooth",
    "Left Kidney",
    "Prostate/Uterus",
    "Prostate",
    "Skin Lesion",
    "Glioma",
    "Optic Disc",
    "Optic Cup",
    "Nucleus",
    "Heart Myocardium",
    "Heart Left Ventricle",
    "Heart Right Ventricle",
    "Heart Atrium Left"
]

# Filter to keep only supported classes
initial_count = len(df)
df = df[df['Object'].isin(supported_classes)]
filtered_count = len(df)
print(f"Filtered from {initial_count} to {filtered_count} rows")
print(f"Kept objects: {', '.join(sorted(df['Object'].unique()))}")
if filtered_count == 0:
    print("No rows remaining after filtering. Check object names in CSV.")
    exit()

# Create Label column
def create_label(row):
    if row['Mode'] == 'point':
        return f"{row['FormattedModel']} {row['Mode']} {int(row['n_pos'])}p{int(row['n_neg'])}n"
    elif row['Mode'] == 'box':
        return f"{row['FormattedModel']} {row['Mode']} {row['bbsize']}px"
    else:
        return f"{row['FormattedModel']} {row['Mode']}"

df['Label'] = df.apply(create_label, axis=1)

# Create a group key for coloring
def create_group_key(row):
    if row['Mode'] == 'point':
        return f"{row['Mode']} {int(row['n_pos'])}p{int(row['n_neg'])}n"
    elif row['Mode'] == 'box':
        return f"{row['Mode']} {row['bbsize']}px"
    else:
        return row['Mode']

df['GroupColorKey'] = df.apply(create_group_key, axis=1)

# Create a sorting key for label order
def label_sort_key(row):
    if row['Mode'] == 'point':
        return (0, row['Mode'], row['n_neg'], row['n_pos'], row['FormattedModel'])
    elif row['Mode'] == 'box':
        return (1, row['Mode'], row['bbsize'], row['FormattedModel'])
    else:
        return (2, row['Mode'], 0, 0, row['FormattedModel'])

df['_sort_key'] = df.apply(label_sort_key, axis=1)
label_order = df[['Label', '_sort_key']].drop_duplicates().sort_values('_sort_key')['Label'].tolist()

# Assign color palette to GroupColorKey
unique_groups = df['GroupColorKey'].unique()
palette = sns.color_palette("Set2", n_colors=len(unique_groups))
color_dict = dict(zip(unique_groups, palette))

# Plotting
sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 3))

# Horizontal boxplot grouped by color key, large font size for hue
sns.boxplot(
    data=df,
    y='Label',
    x=METRIC,
    hue='GroupColorKey',
    width=0.7,
    palette=color_dict,
    order=label_order,
    showfliers=False,
    linewidth=1.5,
)

# Final layout
plt.xlabel(METRIC, fontsize=18)
plt.ylabel('', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(title='Prompt Type', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16)
plt.tight_layout()


plt.savefig(os.path.join(script_dir, f"{FILE}_{METRIC}_distribution.png"), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(script_dir, f"{FILE}_{METRIC}_distribution.pdf"), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(script_dir, f"{FILE}_{METRIC}_distribution.svg"), dpi=300, bbox_inches='tight')

plt.show()
