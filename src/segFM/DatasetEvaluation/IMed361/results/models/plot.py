import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

script_dir = os.path.dirname(__file__)
METRIC = 'IoU'  # Metric to plot
FILE = 'IMed361'  # CSV file to load

# Translation dictionary for model names
model_name_map = {
    "MedSAM2_latest": "MedSAM2-latest",
    "MedSAM2_CTLesion": "MedSAM2-CT",
    "MedSAM2_Chest": "MedSAM2-Chest",
    "sam2.1_hiera_tiny": "SAM2.1-tiny",
    "sam2.1_hiera_small": "SAM2.1-small",
    "sam2.1_hiera_base_plus": "SAM2.1-base+",
    "sam2.1_hiera_large": "SAM2.1-large",
}

# Load CSV
file_path = os.path.join(script_dir, f"{FILE}.csv")
df = pd.read_csv(file_path)

# Format model names
df['FormattedModel'] = df['Model'].map(model_name_map).fillna(df['Model'])

# Create Label column (simplified to show only model name)
def create_label(row):
    return row['FormattedModel']

df['Label'] = df.apply(create_label, axis=1)

# Create a group key for coloring
def create_group_key(row):
    if row['Mode'] == 'point':
        return f"{row['Mode']} {int(row['n_pos'])}p{int(row['n_neg'])}n"
    elif row['Mode'] == 'box':
        return f"{row['Mode']} {row['bbsize']}{"" if "%" in row['bbsize'] else "px"}"
    else:
        return row['Mode']

df['GroupColorKey'] = df.apply(create_group_key, axis=1)

# Create a unique identifier for each model-prompt combination for proper ordering
def create_unique_label(row):
    if row['Mode'] == 'point':
        return f"{row['FormattedModel']}_{row['Mode']}_{int(row['n_pos'])}p{int(row['n_neg'])}n"
    elif row['Mode'] == 'box':
        return f"{row['FormattedModel']}_{row['Mode']}_{row['bbsize']}{"" if "%" in row['bbsize'] else "px"}"
    else:
        return f"{row['FormattedModel']}_{row['Mode']}"

df['UniqueLabel'] = df.apply(create_unique_label, axis=1)

# Create a sorting key for proper ordering by prompt type first, then model
def sort_key(row):
    if row['Mode'] == 'point':
        return (0, row['Mode'], row['n_neg'], row['n_pos'], row['FormattedModel'])
    elif row['Mode'] == 'box':
        return (1, row['Mode'], row['bbsize'], row['FormattedModel'])
    else:
        return (2, row['Mode'], 0, 0, row['FormattedModel'])

df['_sort_key'] = df.apply(sort_key, axis=1)

# Create the proper order using unique labels, then map back to display labels
unique_label_order_df = df[['UniqueLabel', 'Label', 'GroupColorKey', '_sort_key']].drop_duplicates()
unique_label_order_df = unique_label_order_df.sort_values('_sort_key')
unique_label_order = unique_label_order_df['UniqueLabel'].tolist()

# Create mapping from unique labels to display labels
label_mapping = dict(zip(unique_label_order_df['UniqueLabel'], unique_label_order_df['Label']))

# Assign color palette to GroupColorKey
unique_groups = df['GroupColorKey'].unique()
palette = sns.color_palette("Set2", n_colors=len(unique_groups))
color_dict = dict(zip(unique_groups, palette))

# Plotting
sns.set_theme(style="whitegrid")
plt.figure(figsize=(8, 5))


# Horizontal boxplot grouped by color key
ax = sns.boxplot(
    data=df,
    y='UniqueLabel',
    x=METRIC,
    width=0.75,
    hue='GroupColorKey',
    palette=color_dict,
    order=unique_label_order,
    showfliers=False,
)

# Set custom y-tick labels to show only model names
plt.yticks(range(len(unique_label_order)), [label_mapping[label] for label in unique_label_order])

# Print the mean, median and std
for unique_label in unique_label_order:
    subset = df[df['UniqueLabel'] == unique_label]
    display_label = label_mapping[unique_label]
    mean_val = subset[METRIC].mean()
    median_val = subset[METRIC].median()
    std_val = subset[METRIC].std()
    print(f"{display_label}: Mean={mean_val:.3f}, Median={median_val:.3f}, Std={std_val:.3f}")

# Final layout
plt.xlabel(METRIC)
plt.ylabel('')
plt.title(f'{FILE}  - {METRIC} Distribution by Model and Prompt Config')
#plt.legend(title='Prompt Type', bbox_to_anchor=(1.05, 1), loc='upper left')
# Turn off legend for cleaner look
plt.legend([],[], frameon=False)

# Set FontSize of y-axis labels
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.tight_layout()

plt.savefig(os.path.join(script_dir, f"{FILE}_{METRIC}_distribution.png"), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(script_dir, f"{FILE}_{METRIC}_distribution.pdf"), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(script_dir, f"{FILE}_{METRIC}_distribution.svg"), dpi=300, bbox_inches='tight')

plt.show()
