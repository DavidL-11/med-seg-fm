import pandas as pd
import os

df = pd.read_csv(
    os.path.join(os.path.dirname(__file__), "ToothFairy3_models.csv")
)

# Replace "Incisor", "Premolar", "Molar", "Canine" with "Tooth"
df['Object'] = df['Object'].replace(
    to_replace=["Incisor", "Premolar", "Molar", "Canine"],
    value="Tooth",
    regex=True
)

# Keyword order: most specific first
keywords = ["Pulp", "Canal", "Sinus", "Jawbone", "Bridge", "Crown", "Implant", "Pharynx", "Tooth"]

# Process by model
model_results = {}

for model_name, group_df in df.groupby('Model'):
    remaining_df = group_df.copy()
    mean_rows = []

    for keyword in keywords:
        matches = remaining_df[remaining_df['Object'].str.contains(keyword, case=False, na=False)]
        if not matches.empty:
            means = matches[['IoU', 'NSD']].mean()
            means['Object'] = keyword
            mean_rows.append(means)
            remaining_df = remaining_df.drop(matches.index)

    result_df = pd.DataFrame(mean_rows)[['Object', 'IoU', 'NSD']]
    result_df = result_df.rename(columns={'IoU': 'mIoU', 'NSD': 'mNSD'})
    result_df.set_index('Object', inplace=True)
    model_results[model_name] = result_df

# Combine into multi-column DataFrame
final_df = pd.concat(model_results.values(), axis=1, keys=model_results.keys())

print(final_df)

# Print the total mean for each model
for model_name, result_df in model_results.items():
    total_mean_iou = result_df['mIoU'].mean()
    total_mean_nsd = result_df['mNSD'].mean()
    print(f"{model_name}: Total Mean mIoU = {total_mean_iou:.3f}, Total Mean mNSD = {total_mean_nsd:.3f}")

# Export to LaTeX with multicolumn format
latex_table = final_df.to_latex(
    multicolumn=True,
    multicolumn_format='c',
    multirow=True,
    index=True,
    float_format="%.3f"
)

latex_table = latex_table.replace("_", "\_")  # Escape underscores for LaTeX

# Save to .tex file
with open(os.path.join(os.path.dirname(__file__), "ToothFairy3.tex"), "w") as f:
    f.write(latex_table)