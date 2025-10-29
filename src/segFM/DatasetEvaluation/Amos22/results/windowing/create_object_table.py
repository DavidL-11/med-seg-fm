import pandas as pd
import os
import numpy as np

def create_latex_table():
    """
    Create a LaTeX table from AMOS22_windowing.csv with multicolumns for each Model-Mode combination.
    Rows represent unique objects, columns represent Model-Mode combinations with mIoU and mNSD subcolumns.
    """
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'AMOS22_windowing.csv')
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Create Model-Mode combinations
    df['Model_Mode'] = df['Model'] + ' - ' + df['Mode']
    
    # Calculate mean IoU and NSD for each Object and Model-Mode combination
    grouped = df.groupby(['Object', 'Model_Mode']).agg({
        'IoU': 'mean',
        'NSD': 'mean'
    }).round(4)
    
    # Pivot the data to create the desired table structure
    pivot_IoU = grouped['IoU'].unstack(level=1)
    pivot_nsd = grouped['NSD'].unstack(level=1)
    
    # Get unique Model-Mode combinations
    model_modes = df['Model_Mode'].unique()
    
    # Create a MultiIndex for columns
    column_tuples = []
    for model_mode in model_modes:
        column_tuples.extend([
            (model_mode, 'mIoU'),
            (model_mode, 'mNSD')
        ])
    
    # Create the final DataFrame with MultiIndex columns
    result_df = pd.DataFrame(index=pivot_IoU.index)
    
    for model_mode in model_modes:
        if model_mode in pivot_IoU.columns:
            result_df[(model_mode, 'mIoU')] = pivot_IoU[model_mode]
        else:
            result_df[(model_mode, 'mIoU')] = np.nan
            
        if model_mode in pivot_nsd.columns:
            result_df[(model_mode, 'mNSD')] = pivot_nsd[model_mode]
        else:
            result_df[(model_mode, 'mNSD')] = np.nan
    
    # Set the MultiIndex for columns
    result_df.columns = pd.MultiIndex.from_tuples(
        [(col[0], col[1]) for col in result_df.columns],
        names=['Model-Mode', 'Metric']
    )
    
    # Generate LaTeX table
    latex_string = result_df.to_latex(
        multicolumn=True,
        multicolumn_format='c',
        escape=False,
        float_format=lambda x: f'{x:.4f}' if pd.notna(x) else '-',
        caption='AMOS22 Windowing Results: Mean IoU and NSD by Object and Model-Mode',
        label='tab:amos22_windowing_results'
    )
    
    # Use only the table content without document headers
    complete_latex = latex_string
    
    # Save the LaTeX table to file
    output_path = os.path.join(script_dir, 'amos22_windowing_table.tex')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(complete_latex)
    
    print(f"LaTeX table saved to: {output_path}")
    print(f"Number of objects: {len(result_df)}")
    print(f"Number of Model-Mode combinations: {len(model_modes)}")
    print("\nModel-Mode combinations:")
    for i, model_mode in enumerate(model_modes, 1):
        print(f"{i}. {model_mode}")
    
    # Display summary statistics
    print("\nSummary statistics:")
    print(f"Mean IoU across all combinations: {df['IoU'].mean():.4f}")
    print(f"Mean NSD across all combinations: {df['NSD'].mean():.4f}")
    
    return result_df, latex_string

if __name__ == "__main__":
    df_result, latex_table = create_latex_table()
    print("\nTable preview (first 5 rows):")
    print(df_result.head())