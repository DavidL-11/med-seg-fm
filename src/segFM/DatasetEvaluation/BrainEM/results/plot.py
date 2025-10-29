import pandas as pd
import numpy as np
import os

def create_latex_table():
    # Get the CSV file path in the same directory
    current_dir = os.path.dirname(__file__)
    csv_path = os.path.join(current_dir, 'brainem.csv')
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Group by Model, Mode, and Prompt Finder and calculate statistics
    grouped = df.groupby(['Model', 'Mode']).agg({
        'IoU': ['mean', 'median'],
        'NSD': ['mean', 'median']
    }).round(3)
    
    # Reset index to make grouped columns regular columns
    result_df = grouped.reset_index()
    
    # Create the LaTeX table with multicolumn headers using pandas built-in functionality
    complete_table = result_df.to_latex(
        index=False,
        escape=True,  # Changed to True to escape underscores
        column_format='l|l|cc|cc',
        caption='Brain EM Segmentation Results',
        label='tab:brainem_results',
        multicolumn=True,
        multicolumn_format='c|',
        float_format='%.3f'  # Format floats to 3 decimal places
    )
    
    # Print the table
    print("LaTeX Table:")
    print("=" * 50)
    print(complete_table)
    
    # Also save to a file
    output_file = os.path.join(current_dir, 'brainem_results_table.tex')
    with open(output_file, 'w') as f:
        f.write(complete_table)
    
    print(f"\nTable saved to: {output_file}")
    
    # Display summary statistics
    print("\nSummary Statistics:")
    print("=" * 50)
    print(grouped.to_string())
    
    return grouped, complete_table

if __name__ == "__main__":
    df, latex_table = create_latex_table()