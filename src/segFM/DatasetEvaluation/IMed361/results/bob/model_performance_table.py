import pandas as pd
import numpy as np
import os

supported_classes = ["lung", "polyp", "liver", "kidney_right", "spleen", "aorta", "inferior_vena_cava",
"gallbladder", "esophagus", "stomach", "kidney_left", "prostate_and_uterus", "skin_lesion", "glioma", 
"optic_disc", "optic_cup", "heart_myocardium", "heart_ventricle_left", "heart_ventricle_right",
"heart_atrium_left"]

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(script_dir, "IMed361_bob.csv")
output_path = os.path.join(script_dir, "model_performance.tex")

# Read CSV
df = pd.read_csv(input_path)

# Filter to supported classes only
df = df[df["Object"].isin(supported_classes)]

# Model name mapping
model_name_map = {
    "MedSAM2_latest": "MedSAM2-latest",
}

dataset_name_map = {
    "Learn2Reg2022 L2R task1 AbdomenCTCT": "L2R2022 AbdomenCTCT",
    "Learn2Reg2022 L2R task1 AbdomenMRCT seg CT": "L2R2022 T1 AbdomenMRCT-Ct",
    "Learn2Reg2022 L2R task1 AbdomenMRCT seg MR": "L2R2022 T1 AbdomenMRCT-MR",
    "Prostate MRI Segmentation Dataset": "Prostate MRI",
    "hyper-kvasir-segmented-images": "Hyper-Kvasir Segmented",
    "Continuous Registration task1": "Continuous Registration 1",
}

prompt_finder_map = {
    "center": "mid",
    "random": "rand. ",
}

# Metrics and grouping
metrics = ['DSC', 'NSD']
group_cols = ['Model', 'Dataset', 'Modality', 'Mode', 'n_pos', 'n_neg', 'bbsize', 'Prompt Finder']
agg_df = df.groupby(group_cols)[metrics].mean().reset_index()

# Label creation
def make_prompt_label(row):
    if row['Mode'] == 'point':
        return f"{row['Prompt Finder']}point {row['n_pos']}p/{row['n_neg']}n"
    elif row['Mode'] == 'box':
        return f"box {int(row['bbsize'])}px"
    return row['Mode']

# Apply label styles
# Shorten the prompt finder names
agg_df['Prompt Finder'] = agg_df['Prompt Finder'].map(prompt_finder_map).fillna(agg_df['Prompt Finder'])
# The large multicolumn should specify the name of the model, shortened for readability
agg_df['ModelShort'] = agg_df['Model'].map(model_name_map).fillna(agg_df['Model'])
# Some dataset names are too long, so we shorten them
agg_df['Dataset'] = agg_df['Dataset'].map(dataset_name_map).fillna(agg_df['Dataset'])
# One column should specify the prompt type, e.g. "center point 1p/0n" or "box 10px"
agg_df['RowLabel'] = agg_df.apply(make_prompt_label, axis=1)

# Combine Dataset and Modality for labeling
agg_df['Dataset (modality)'] = agg_df.apply(lambda row: f"\\makecell[l]{{{row['Dataset']}\\\\({row['Modality'].replace('_', ' ')})}}", axis=1)
agg_df['Index'] = list(zip(agg_df['Dataset (modality)'], agg_df['RowLabel']))

# Reorder and reshape pivot table to have Models as top-level
pivot_df = agg_df.pivot(index='Index', columns=['ModelShort'], values=metrics)
pivot_df = pivot_df.swaplevel(axis=1).sort_index(axis=1, level=0)

# Flatten for processing
flat_df = pivot_df.copy()
flat_df.columns = [f"{model} {metric}" for model, metric in flat_df.columns]
flat_df = flat_df.reset_index()
flat_df.columns.name = None
highlight_df = flat_df.set_index(['Index'])
highlight_df['Dataset (modality)'] = [ds for ds, _ in highlight_df.index]

# Identify maximum values per Dataset and Metric (across all models)
highlight_df_clean = highlight_df.drop(columns='Dataset (modality)')
max_highlight = pd.DataFrame(False, index=highlight_df_clean.index, columns=highlight_df_clean.columns)

for metric in metrics:
    metric_cols = [col for col in highlight_df_clean.columns if col.endswith(metric)]
    dataset_group = highlight_df['Dataset (modality)']
    for dataset in dataset_group.unique():
        idx = dataset_group == dataset
        metric_subset = highlight_df_clean.loc[idx, metric_cols]
        max_val = metric_subset.max().max()
        max_locs = metric_subset == max_val
        max_highlight.loc[idx, metric_cols] |= max_locs

# Format values, bold maxima
formatted_df = highlight_df_clean.copy()
for col in formatted_df.columns:
    formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.3f}")
    formatted_df[col] = np.where(max_highlight[col], formatted_df[col].apply(lambda x: f"\\textbf{{{x}}}"), formatted_df[col])

# Restore index for LaTeX output
formatted_df['Dataset (modality)'] = [ds for ds, _ in highlight_df.index]
formatted_df['Prompt'] = [lbl for _, lbl in highlight_df.index]
formatted_df = formatted_df.set_index(['Dataset (modality)', 'Prompt'])

# Reconstruct MultiIndex columns
column_tuples = [col.split() for col in formatted_df.columns if col not in ['Dataset (modality)', 'Prompt']]
column_tuples = [(model, metric) for model, metric in column_tuples]
formatted_df = formatted_df[[f"{model} {metric}" for model, metric in column_tuples]]
formatted_df.columns = pd.MultiIndex.from_tuples(column_tuples)

# Compute Mean, Median, Std across all prompt rows (excluding summary rows)
summary_stats = formatted_df.copy()

# Compute stats
mean_row = summary_stats.map(lambda x: float(x.replace("\\textbf{", "").replace("}", ""))).mean().to_frame().T
median_row = summary_stats.map(lambda x: float(x.replace("\\textbf{", "").replace("}", ""))).median().to_frame().T
std_row = summary_stats.map(lambda x: float(x.replace("\\textbf{", "").replace("}", ""))).std().to_frame().T

# Format values
mean_row = mean_row.map(lambda x: f"{x:.3f}")
median_row = median_row.map(lambda x: f"{x:.3f}")
std_row = std_row.map(lambda x: f"{x:.3f}")

# Set multi-index labels
mean_row.index = pd.MultiIndex.from_tuples([("Summary", "Mean")])
median_row.index = pd.MultiIndex.from_tuples([("Summary", "Median")])
std_row.index = pd.MultiIndex.from_tuples([("Summary", "Std")])

# Append summary rows to the table
formatted_df = pd.concat([formatted_df, mean_row, median_row, std_row])

# Export to LaTeX
latex_str = formatted_df.to_latex(
    escape=False,
    multirow=True,
    multicolumn=True,
    multicolumn_format='c',
    longtable=True,
    caption="IMed-361M Model Performance Summary",
    label="tab:model_performance"
)
# Place Dataset labels in the middle [m] instead of top [t]
latex_str = latex_str.replace(r'\multirow[t]{', r'\multirow[m]{')

# Replace unicode characters like μ with LaTeX commands
latex_str = latex_str.replace('μ', r'$\mu$')

# Save LaTeX
with open(output_path, "w") as f:
    f.write(latex_str)

# Print LaTeX path
print(f"LaTeX table saved to: {output_path}")