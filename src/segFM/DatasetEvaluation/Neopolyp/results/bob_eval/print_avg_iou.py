import os
import pandas as pd
    
# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_file = os.path.join(script_dir, 'neopolyp.csv')

# Read the CSV file
print(f"Reading CSV file: {csv_file}")
df = pd.read_csv(csv_file)

print(f"Loaded {len(df)} rows from CSV")
print(f"Unique objects: {sorted(df['Object'].unique())}")
print(f"Unique models: {sorted(df['Model'].unique())}")
print(f"Unique modes: {sorted(df['Mode'].unique())}")
print("\n" + "="*80)

# Get unique objects and sort them
unique_objects = sorted(df['Object'].unique())

# Process each object
for obj in unique_objects:
    print(f"---> {obj}")
    
    # Filter data for current object
    obj_data = df[df['Object'] == obj]
    
    # Get unique model-mode combinations for this object
    model_mode_combinations = obj_data.groupby(['Model', 'Mode'])['IoU'].agg([
        ('mean', 'mean'),
        ('std', 'std'),
        ('median', 'median')
    ]).reset_index()
    
    # Sort by model name then mode name
    model_mode_combinations = model_mode_combinations.sort_values(['Model', 'Mode'])
    
    # Print results for each combination
    for _, row in model_mode_combinations.iterrows():
        model = row['Model']
        mode = row['Mode']
        mean_iou = row['mean']
        std = row['std']
        median = row['median']
        
        print(f"- {model} - {mode}: mIoU = {mean_iou:.3f}, STD = {std:.3f}, Median = {median:.3f}")
    
    print()  # Empty line between objects

print("="*80)