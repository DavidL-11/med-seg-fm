import os
import pandas as pd

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_file = os.path.join(script_dir, 'neopolyp.csv')

# Read the CSV file
df = pd.read_csv(csv_file)

# Group by the specified columns and calculate statistics
group_columns = ['Model', 'Mode', 'bbsize']

# Calculate mean and median for IoU and NSD
grouped = df.groupby(group_columns).agg({
    'IoU': ['mean', 'median'],
    'NSD': ['mean', 'median']
}).round(3)

# Flatten column names
grouped.columns = ['_'.join(col).strip() for col in grouped.columns]

# Reset index to make grouped columns regular columns
result_df = grouped.reset_index()

# Rename columns for better display and remove LaTeX-unsafe characters
result_df = result_df.rename(columns={
    'IoU_mean': 'IoU Mean',
    'IoU_median': 'IoU Median', 
    'NSD_mean': 'NSD Mean',
    'NSD_median': 'NSD Median'
})

# Create LaTeX table
latex_str = result_df.to_latex(
    index=False,
    escape=False,
    float_format='{:.3f}'.format,
    column_format='l|c|c|cc|cc',
    multicolumn_format='c',
    caption='Summary statistics for neopolyp dataset by model configuration',
    label='tab:neopolyp_summary'
)

# Add multicolumn headers manually by modifying the LaTeX string
lines = latex_str.split('\n')

# Find the header line (contains Model, Mode, etc.)
header_idx = None
for i, line in enumerate(lines):
    if 'Model &' in line:
        header_idx = i
        break

if header_idx is not None:
    # Create the multicolumn header
    multicolumn_header = r'Model & Mode & bbsize & \multicolumn{2}{c|}{IoU} & \multicolumn{2}{c}{NSD} \\'
    subheader = r' & & & Mean & Median & Mean & Median \\'
    
    # Replace the original header with multicolumn header
    lines[header_idx] = multicolumn_header
    lines.insert(header_idx + 1, subheader)
    lines.insert(header_idx + 2, r'\hline')

# Reconstruct the LaTeX string
latex_str = '\n'.join(lines)

latex_str = latex_str.replace('_', r'\_')

# Save to file
output_file = os.path.join(script_dir, 'neopolyp_summary_table.tex')
with open(output_file, 'w') as f:
    f.write(latex_str)

print(f"\nLaTeX table saved to: {output_file}")

