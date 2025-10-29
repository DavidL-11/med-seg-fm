import pandas as pd
import os
import numpy as np
    
# Get the directory of the current script
script_dir = os.path.dirname(__file__)

# Use the first CSV file found (or you can specify a particular name)
csv_file = os.path.join(script_dir, "ToothFairy3_bob.csv")

# Read the CSV file
df = pd.read_csv(csv_file)

# Group by Model, Mode, and Object to calculate statistics
grouped = df.groupby(['Model', 'Mode', 'Object']).agg({
    'IoU': 'mean',
    'NSD': 'mean'
}).round(3)

# Reset index to make grouping columns regular columns
grouped = grouped.reset_index()

# Pivot to get Pharynx and Tooth as separate columns
pivot_df = grouped.pivot_table(
    index=['Model', 'Mode'], 
    columns='Object', 
    values=['IoU', 'NSD'], 
    fill_value=0
)

# Flatten column names
pivot_df.columns = ['_'.join(col).strip() for col in pivot_df.columns.values]

# Reset index to make Model and Mode regular columns
result_df = pivot_df.reset_index()

# Create the final dataframe with desired structure
final_df = pd.DataFrame()
final_df['Model'] = result_df['Model']
final_df['Mode'] = result_df['Mode']

# Add Pharynx columns (mIoU and mNSD)
final_df['Pharynx_mIoU'] = result_df.get('IoU_Pharynx', 0)
final_df['Pharynx_mNSD'] = result_df.get('NSD_Pharynx', 0)

# Add Tooth columns (mIoU and mNSD)
final_df['Tooth_mIoU'] = result_df.get('IoU_Tooth', 0)
final_df['Tooth_mNSD'] = result_df.get('NSD_Tooth', 0)

# Create multicolumn header structure
# pandas.to_latex() doesn't directly support multicolumn headers,
# so we'll create the table and then modify the header

# Generate basic LaTeX table
latex_table = final_df.to_latex(
    index=False,
    float_format=lambda x: f'{x:.3f}' if pd.notnull(x) else '',
    escape=False,
    column_format='ll|cc|cc'
)

# Replace the header to add multicolumn structure
lines = latex_table.split('\n')

# Find the header lines (between \begin{tabular} and first data row)
header_start = None
header_end = None

for i, line in enumerate(lines):
    if '\\begin{tabular}' in line:
        header_start = i + 1
    elif '\\midrule' in line and header_start is not None:
        header_end = i
        break

if header_start is not None and header_end is not None:
    # Create new header with multicolumn
    new_header = [
        lines[header_start],  # Keep the \toprule line
        'Model & Mode & \\multicolumn{2}{c|}{Pharynx} & \\multicolumn{2}{c}{Tooth} \\\\',
        '\\cmidrule(lr){3-4} \\cmidrule(lr){5-6}',
        ' & & mIoU & mNSD & mIoU & mNSD \\\\'
    ]
    
    # Replace the original header
    lines = lines[:header_start] + new_header + lines[header_end:]

# Join the lines back together
modified_latex = '\n'.join(lines)

modified_latex = modified_latex.replace("_", "\_")  # Escape underscores for LaTeX
modified_latex = modified_latex.replace(".pt", "")  # Remove .pt from model names

# Save the LaTeX table to a file
output_file = os.path.join(script_dir, "ToothFairy3_bob_table.tex")
with open(output_file, 'w') as f:
    f.write(modified_latex)

print(f"LaTeX table saved to: {output_file}")

# Also print the table to console
print("\nGenerated LaTeX table:")
print(modified_latex)
    