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
    "MedSAM2_CTLesion": "MedSAM2-CT",
    "MedSAM2_Chest": "MedSAM2-Chest",
    "sam2.1_hiera_tiny": "SAM2.1-tiny",
    "sam2.1_hiera_small": "SAM2.1-small",
    "sam2.1_hiera_base_plus": "SAM2.1-base+",
    "sam2.1_hiera_large": "SAM2.1-large",
}

df = pd.read_csv(input_path)

# Format model names
df['FormattedModel'] = df['Model'].map(model_name_map).fillna(df['Model'])

# Create Label column
def create_label(row):
    if row['Mode'] == 'point':
        return f"{row['FormattedModel']} {row['Mode']} {int(row['n_pos'])}p{int(row['n_neg'])}n"
    elif row['Mode'] == 'box':
        return f"{row['FormattedModel']} {row['Mode']} {row['bbsize']}{"" if "%"  in row['bbsize'] else "px"}"
    else:
        return f"{row['FormattedModel']} {row['Mode']}"

df['Label'] = df.apply(create_label, axis=1)

# Create a group key for coloring
def create_group_key(row):
    if row['Mode'] == 'point':
        return f"{row['Mode']} {int(row['n_pos'])}p{int(row['n_neg'])}n"
    elif row['Mode'] == 'box':
        return f"{row['Mode']} {row['bbsize']}{"" if "%"  in row['bbsize'] else "px"}"
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
plt.figure(figsize=(6, 5))

# Horizontal boxplot grouped by color key, x-tick size 16pt
sns.boxplot(
    data=df,
    y='Label',
    x=METRIC,
    hue='GroupColorKey',
    width=0.75,
    palette=color_dict,
    order=label_order,
    showfliers=False,
    
)

# Final layout
plt.xlabel(METRIC, fontsize=16)
plt.ylabel('')
plt.title(f'{FILE}  - {METRIC} Distribution by Model and Prompt Config')
#plt.legend(title='Prompt Type', bbox_to_anchor=(1.05, 1), loc='upper left')

# Turn off legend for cleaner look
plt.legend([],[], frameon=False)

# Turn off y-axis labels
plt.yticks([])
plt.xticks(fontsize=14)

plt.tight_layout()


plt.savefig(os.path.join(script_dir, f"{FILE}_{METRIC}_distribution.png"), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(script_dir, f"{FILE}_{METRIC}_distribution.pdf"), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(script_dir, f"{FILE}_{METRIC}_distribution.svg"), dpi=300, bbox_inches='tight')

plt.show()
