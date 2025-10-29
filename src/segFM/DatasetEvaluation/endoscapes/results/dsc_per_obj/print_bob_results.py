import os
import pandas as pd
import numpy as np

FILTER_BOB = True

def main():
    # Get the CSV file path relative to this script
    current_dir = os.path.dirname(__file__)
    csv_path = os.path.join(current_dir, 'endoscapes.csv')
    
    # Read the CSV file
    try:
        df = pd.read_csv(csv_path)
        print(f"Successfully loaded {len(df)} rows from {csv_path}")
    except FileNotFoundError:
        print(f"Error: {csv_path} not found")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
 
    # Group by Model-Mode combination and calculate statistics
    grouped = df.groupby(['Model', 'Mode'])
    
    print("\n" + "="*80)
    print("RESULTS: Mean/Std/Median IoU and NSD for each Model-Mode combination")
    print("="*80)
    
    for (model, mode), group in grouped:
        print(f"\nModel: {model}, Mode: {mode}")
        print(f"Number of samples: {len(group)}")
        
        # Calculate IoU statistics
        iou_mean = group['IoU'].mean()
        iou_std = group['IoU'].std()
        iou_median = group['IoU'].median()
        
        # Calculate NSD statistics
        nsd_mean = group['NSD'].mean()
        nsd_std = group['NSD'].std()
        nsd_median = group['NSD'].median()
        
        print(f"IoU  - Mean: {iou_mean:.3f}, Std: {iou_std:.3f}, Median: {iou_median:.3f}")
        print(f"NSD  - Mean: {nsd_mean:.3f}, Std: {nsd_std:.3f}, Median: {nsd_median:.3f}")
        print("-" * 60)
    
    # Summary statistics
    print(f"\nSUMMARY:")
    print(f"Total Model-Mode combinations: {len(grouped)}")
    print(f"Objects analyzed: {sorted(df['Object'].unique())}")

if __name__ == "__main__":
    main()