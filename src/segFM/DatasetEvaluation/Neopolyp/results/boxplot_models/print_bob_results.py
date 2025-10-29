import os
import pandas as pd
import numpy as np

FILTER_BOB = True

def main():
    # Get the CSV file path relative to this script
    current_dir = os.path.dirname(__file__)
    csv_path = os.path.join(current_dir, 'neopolyp.csv')
    
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
    
    # Define the objects of interest
    target_objects = [
        'Liver', 'Right Kidney', 'Left Kidney', 'Spleen', 'Aorta', 
        'Inferior Vena Cava', 'Gallbladder', 'Espophagus', 'Stomach'
        , "lung", "polyp", "liver", "kidney_right", "spleen", "aorta", "inferior_vena_cava",
        "gallbladder", "esophagus", "stomach", "kidney_left", "prostate_and_uterus", "skin_lesion", "glioma", 
        "optic_disc", "optic_cup", "heart_myocardium", "heart_ventricle_left", "heart_ventricle_right",
        "heart_atrium_left",
        'Glottis', 'Bounding-box Oracle for Biomedicine 3', 'Left Vocal Cord', 'Right Vocal Cord',
        'Lung', 'Polyp', 'Tool', 'Organ', 'Liver', 'Right Kidney', 'Spleen', 'Mitochondria',
        'Aorta', 'Inferior Vena Cava', 'Pharynx', 'Fetal Head', 'Gallbladder', 'Esophagus',
        'Stomach', 'Tooth', 'Left Kidney', 'Prostate/Uterus', 'Skin Lesion', 'Glioma',
        'Optic Disc', 'Optic Cup', 'Nucleus', 'Heart Myocardium', 'Heart Left Ventricle',
        'Heart Right Ventricle', 'Heart Atrium Left',
    ]
    
    
    # Filter rows to keep only target objects
    initial_count = len(df)
    if FILTER_BOB:
        df_filtered = df[df['Object'].isin(target_objects)]
    else:
        df_filtered = df

    filtered_count = len(df_filtered)
    
    print(f"\nFiltered from {initial_count} to {filtered_count} rows")
    print(f"Kept objects: {', '.join(target_objects)}")
    
    if filtered_count == 0:
        print("No rows remaining after filtering. Check object names in CSV.")
        return
    
    # Group by Model-Mode combination and calculate statistics
    grouped = df_filtered.groupby(['Model', 'Mode'])
    
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
    print(f"Objects analyzed: {sorted(df_filtered['Object'].unique())}")

if __name__ == "__main__":
    main()