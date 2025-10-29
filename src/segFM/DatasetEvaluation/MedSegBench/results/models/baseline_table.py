import pandas as pd
import numpy as np
import os

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(script_dir, "MedSegBench.csv")
output_path = os.path.join(script_dir, "baseline_table.tex")

metrics = ["IoU"]

# Read CSV
df = pd.read_csv(input_path)

# Filter out rows where 'Mode' is not 'box' and 'bbsize' is not 0
df = df[(df["Mode"] == "box") & (df["bbsize"] == str(0))]

# Model name mapping
model_name_map = {
    "MedSAM2_latest": "MedSAM2-latest",
    "MedSAM2_CTLesion": "MedSAM2-CT",
    "MedSAM2_Chest": "MedSAM2-Chest",
    "sam2.1_hiera_tiny": "SAM2.1-tiny",
    "sam2.1_hiera_small": "SAM2.1-small",
    "sam2.1_hiera_base_plus": "SAM2.1-base+",
    "sam2.1_hiera_large": "SAM2.1-large",
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

dn121_baseline = {
    "abdomenus": 0.632,
    "bbbc010": 0.854,
    "bkai-igh": 0.615,
    "brifiseg": 0.738,
    "busi": 0.615,
    "cellnuclei": 0.838,
    "chasedb1": 0.618,
    "chuac": 0.4,
    "covid19radio": 0.983,
    "covidquex": 0.647,
    "cystoidfluid": 0.761,
    "dca1": 0.623,
    "deepbacs": 0.867,
    "drive": 0.643,
    "dynamicnuclear": 0.897,
    "fhpsaop": 0.928,
    "idrib": 0.054,
    "isic2016": 0.825,
    "isic2018": 0.785,
    "kvasir": 0.718,
    "m2caiseg": 0.2,
    "monusac": 0.54,
    "mosmedplus": 0.686,
    "nuclei": 0.166,
    "nuset": 0.91,
    "pandental": 0.932,
    "polypgen": 0.545,
    "promise12": 0.832,
    "robotool": 0.798,
    "tnbcnuclei": 0.654,
    "ultrasoundnerve": 0.676,
    "usforkidney": 0.96,
    "uwskincancer": 0.779,
    "wbc": 0.936,
    "yeaz": 0.914,
}


# Metrics and grouping
group_cols = ["Model", "Dataset", "Mode", "n_pos", "n_neg", "bbsize", "Prompt Finder"]
agg_df = df.groupby(group_cols)[metrics].mean().reset_index()


# Add the DN121 baseline IoU values to the DataFrame, Model is "DN121 Baseline", NSD is empty
dn121_df = pd.DataFrame(
    {
        "Model": "DN121",
        "Dataset": list(dn121_baseline.keys()),
        "Mode": "box",
        "IoU": list(dn121_baseline.values())
    }
)
# Append the DN121 baseline to the highlight DataFrame
agg_df = pd.concat([agg_df, dn121_df], ignore_index=False)

# The large multicolumn should specify the name of the model, shortened for readability
agg_df["ModelShort"] = agg_df["Model"].map(model_name_map).fillna(agg_df["Model"])
# Some dataset names are too long, so we shorten them
agg_df["Dataset"] = agg_df["Dataset"].map(dataset_name_map).fillna(agg_df["Dataset"])

# Combine Dataset and Modality for labeling
agg_df["Dataset"] = agg_df.apply(
    lambda row: f"\\makecell[l]{{{row['Dataset']}}}", axis=1
)
agg_df["Index"] = list(agg_df["Dataset"])

# Reorder and reshape pivot table to have Models as top-level
pivot_df = agg_df.pivot(index="Index", columns=["ModelShort"], values=metrics)
pivot_df = pivot_df.swaplevel(axis=1).sort_index(axis=1, level=0)

# Flatten for processing
flat_df = pivot_df.copy()
flat_df.columns = [f"{model} {metric}" for model, metric in flat_df.columns]
flat_df = flat_df.reset_index()
flat_df.columns.name = None
highlight_df = flat_df.set_index(["Index"])
highlight_df["Dataset"] = highlight_df.index

# Identify maximum values per Dataset and Metric (across all models)
highlight_df_clean = highlight_df.drop(columns="Dataset")
max_highlight = pd.DataFrame(
    False, index=highlight_df_clean.index, columns=highlight_df_clean.columns
)

for metric in metrics:
    metric_cols = [col for col in highlight_df_clean.columns if col.endswith(metric)]
    dataset_group = highlight_df["Dataset"]
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
    formatted_df[col] = np.where(
        max_highlight[col],
        formatted_df[col].apply(lambda x: f"\\textbf{{{x}}}"),
        formatted_df[col],
    )

# Restore index for LaTeX output
formatted_df["Dataset"] = highlight_df.index
formatted_df = formatted_df.set_index(["Dataset"])

# Compute summary statistics
column_means = highlight_df_clean.mean(skipna=True)
column_stds = highlight_df_clean.std(skipna=True)
column_medians = highlight_df_clean.median(skipna=True)

# Format each row
mean_row = column_means.apply(lambda x: f"{x:.3f}")
mean_row.name = r"\textit{Mean}"

std_row = column_stds.apply(lambda x: f"{x:.3f}")
std_row.name = r"\textit{Std}"

median_row = column_medians.apply(lambda x: f"{x:.3f}")
median_row.name = r"\textit{Median}"

# Append to formatted DataFrame
summary_df = pd.DataFrame([mean_row, std_row, median_row])
formatted_df = pd.concat([formatted_df, summary_df])

# Reconstruct MultiIndex columns
column_tuples = [
    col.split() for col in formatted_df.columns if col not in ["Dataset"]
]
print(column_tuples)
column_tuples = [(model, metric) for model, metric in column_tuples]
formatted_df = formatted_df[[f"{model} {metric}" for model, metric in column_tuples]]
formatted_df.columns = pd.MultiIndex.from_tuples(column_tuples)

# Export to LaTeX
latex_str = formatted_df.to_latex(
    escape=False,
    multirow=True,
    multicolumn=True,
    multicolumn_format="c",
    longtable=True,
    caption="MedSegBench DenseNet121 Baseline vs Box Prompts",
    label="tab:model_performance",
)

# Fix vertical alignment for multirow
latex_str = latex_str.replace(r"\multirow[t]{", r"\multirow[m]{")

# Replace unicode characters with LaTeX equivalents
latex_str = latex_str.replace("μ", r"$\mu$")

# Insert \midrule before the mean row
latex_lines = latex_str.splitlines()
for i, line in enumerate(latex_lines):
    if line.strip().startswith(r"\textit{Mean}"):
        latex_lines.insert(i, r"\midrule")
        break
latex_str = "\n".join(latex_lines)

# Save LaTeX
with open(output_path, "w") as f:
    f.write(latex_str)


# Print LaTeX path
print(f"LaTeX table saved to: {output_path}")
