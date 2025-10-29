import pandas as pd
import os
import numpy as np

def create_latex_table():
    """
    Create a LaTeX table from AMOS22_windowing.csv with rows for Model-Mode combinations.
    Columns represent IoU and NSD metrics with mean and median subcolumns.
    """
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'AMOS22_windowing.csv')
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Create Model-Mode combinations
    df['Model_Mode'] = df['Model'] + ' - ' + df['Mode']
    
    # Calculate mean and median IoU and NSD for each Model-Mode combination across all objects
    grouped = df.groupby('Model_Mode').agg({
        'IoU': ['mean', 'median'],
        'NSD': ['mean', 'median']
    }).round(4)
    
    # Flatten the column names
    grouped.columns = [f'{col[0]}_{col[1]}' for col in grouped.columns]
    
    # Create a MultiIndex for columns
    column_tuples = [
        ('IoU', 'Mean'),
        ('IoU', 'Median'),
        ('NSD', 'Mean'),
        ('NSD', 'Median')
    ]
    
    # Create the final DataFrame with MultiIndex columns
    result_df = pd.DataFrame(index=grouped.index)
    result_df[('IoU', 'Mean')] = grouped['IoU_mean']
    result_df[('IoU', 'Median')] = grouped['IoU_median']
    result_df[('NSD', 'Mean')] = grouped['NSD_mean']
    result_df[('NSD', 'Median')] = grouped['NSD_median']
    
    # Set the MultiIndex for columns
    result_df.columns = pd.MultiIndex.from_tuples(
        column_tuples,
        names=['Metric', 'Statistic']
    )
    
    # Generate LaTeX table
    latex_string = result_df.to_latex(
        multicolumn=True,
        multicolumn_format='c',
        escape=False,
        float_format=lambda x: f'{x:.4f}' if pd.notna(x) else '-',
        caption='AMOS22 Windowing Results: Mean and Median IoU and NSD by Model-Mode',
        label='tab:amos22_windowing_results_by_model'
    )
    
    # Save the LaTeX table to file
    output_path = os.path.join(script_dir, 'amos22_windowing_table2.tex')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(latex_string)
    
    print(f"LaTeX table saved to: {output_path}")
    print(f"Number of Model-Mode combinations: {len(result_df)}")
    print("\nModel-Mode combinations:")
    for i, model_mode in enumerate(result_df.index, 1):
        print(f"{i}. {model_mode}")
    
    # Display summary statistics
    print("\nOverall summary statistics:")
    print(f"Overall mean IoU: {df['IoU'].mean():.4f}")
    print(f"Overall median IoU: {df['IoU'].median():.4f}")
    print(f"Overall mean NSD: {df['NSD'].mean():.4f}")
    print(f"Overall median NSD: {df['NSD'].median():.4f}")
    
    # Display comparison between modes
    print("\nComparison between Model-Mode combinations:")
    for metric in ['IoU', 'NSD']:
        print(f"\n{metric} Metrics:")
        for stat in ['Mean', 'Median']:
            print(f"  {stat}:")
            for model_mode in result_df.index:
                value = result_df.loc[model_mode, (metric, stat)]
                print(f"    {model_mode}: {value:.4f}")
    
    return result_df, latex_string

if __name__ == "__main__":
    df_result, latex_table = create_latex_table()
    print("\nTable preview:")
    print(df_result)