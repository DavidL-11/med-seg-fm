import os
import pandas as pd
import numpy as np

def main():
    # Get the CSV file path relative to this script
    current_dir = os.path.dirname(__file__)
    csv_path = os.path.join(current_dir, 'msd.csv')
    
    # Read the CSV file
    try:
        df = pd.read_csv(csv_path)
        print(f"Successfully loaded {len(df)} rows from msd.csv")
        print(f"Columns: {list(df.columns)}")
    except FileNotFoundError:
        print(f"Error: msd.csv not found in {current_dir}")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    # Show unique objects before grouping
    print(f"\nOriginal unique objects: {sorted(df['Object'].unique())}")
    
    # Group brain tumor objects into "BrainTumor"
    brain_tumor_objects = ['edema', 'non-enhancing tumor', 'enhancing tumor']
    df_processed = df.copy()
    
    # Replace brain tumor objects with "BrainTumor"
    for obj in brain_tumor_objects:
        df_processed.loc[df_processed['Object'] == obj, 'Object'] = 'BrainTumor'
    
    print(f"Objects after grouping: {sorted(df_processed['Object'].unique())}")
    
    # Group by Object and Model-Mode combination
    grouped = df_processed.groupby(['Object', 'Model', 'Mode'])
    
    print("\n" + "="*80)
    print("RESULTS: Mean/Std/Median metrics for each Object by Model-Mode combination")
    print("="*80)
    
    for (obj, model, mode), group in grouped:
        print(f"\nObject: {obj} | Model: {model} | Mode: {mode}")
        print(f"Number of samples: {len(group)}")
        
        # Calculate IoU statistics if column exists
        if 'IoU' in group.columns:
            iou_mean = group['IoU'].mean()
            iou_std = group['IoU'].std()
            iou_median = group['IoU'].median()
            print(f"IoU  - Mean: {iou_mean:.4f}, Std: {iou_std:.4f}, Median: {iou_median:.4f}")
        
        # Calculate NSD statistics if column exists
        if 'NSD' in group.columns:
            nsd_mean = group['NSD'].mean()
            nsd_std = group['NSD'].std()
            nsd_median = group['NSD'].median()
            print(f"NSD  - Mean: {nsd_mean:.4f}, Std: {nsd_std:.4f}, Median: {nsd_median:.4f}")
        
        # Calculate DSC statistics if column exists
        if 'DSC' in group.columns:
            dsc_mean = group['DSC'].mean()
            dsc_std = group['DSC'].std()
            dsc_median = group['DSC'].median()
            print(f"DSC  - Mean: {dsc_mean:.4f}, Std: {dsc_std:.4f}, Median: {dsc_median:.4f}")
        
        print("-" * 60)
    
    # Summary statistics
    print(f"\nSUMMARY:")
    print(f"Total unique objects after grouping: {len(df_processed['Object'].unique())}")
    print(f"Brain tumor objects grouped: {', '.join(brain_tumor_objects)}")
    print(f"Total Object-Model-Mode combinations: {len(grouped)}")
    
    # Show breakdown by object
    object_counts = df_processed.groupby('Object').size()
    print(f"\nSample counts by object:")
    for obj, count in object_counts.items():
        print(f"  {obj}: {count} samples")

if __name__ == "__main__":
    main()