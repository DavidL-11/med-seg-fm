import pandas as pd
import os

# Load CSV from the same folder as the script
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, 'BAGLS_full.csv')
df = pd.read_csv(csv_path)

# Optional: Shorten model names
model_name_map = {
    'sam2.1_hiera_tiny': 'SAM2.1 Tiny',
    # Add more mappings if needed
}
df['Model'] = df['Model'].replace(model_name_map)

# Format prompt description
def format_prompt(row):
    if str(row['Prompt Finder']).lower() == 'yolo':
        return 'YOLO'
    elif row['Mode'] == 'point':
        return f"{row['Mode']} {int(row['n_pos'])}p{int(row['n_neg'])}n"
    elif row['Mode'] == 'box':
        return f"{row['Mode']} {int(row['bbsize'])}px"
    else:
        return row['Mode']

df['Prompt'] = df.apply(format_prompt, axis=1)

# Group by Model, Prompt, Object
grouped = df.groupby(['Model', 'Prompt', 'Object'])[['DSC', 'NSD', 'IoU']].mean()

# Truncate (not round) to 3 decimal places
grouped = grouped.applymap(lambda x: f"{x:.3f}").reset_index()

# Rename columns
grouped.rename(columns={
    'DSC': 'mDSC',
    'NSD': 'mNSD',
    'IoU': 'mIoU'
}, inplace=True)

# Export to LaTeX
latex_str = grouped.to_latex(index=False, escape=False, caption="Aggregated Results", label="tab:results")

# Save LaTeX table (optional)
with open(os.path.join(current_dir, 'yolo_table.tex'), 'w') as f:
    f.write(latex_str)

# Print LaTeX
print(latex_str)
