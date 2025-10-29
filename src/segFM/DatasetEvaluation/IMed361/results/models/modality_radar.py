import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from math import pi

def main():
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'IMed361.csv')
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Filter data: Mode=="box", bbsize=="0" (keep all models)
    filtered_df = df[
        (df['Mode'] == 'box') & 
        (df['bbsize'].astype(str) == '0')
    ].copy()
    
    print(f"Original dataset size: {len(df)}")
    print(f"Filtered dataset size: {len(filtered_df)}")
    
    # Get unique models
    unique_models = filtered_df['Model'].unique()
    print(f"Unique models: {list(unique_models)}")
    
    # Function to assign modality groups
    def assign_modality_group(modality):
        modality_lower = str(modality).lower()
        
        if 'ct' in modality_lower:
            return 'CT'
        elif 'mr' in modality_lower:
            return 'MR'
        elif modality_lower in ['fundus_photography']:
            return 'Fundus Photography'
        elif modality_lower in ['endoscopy']:
            return 'Endoscopy'
        elif modality_lower in ['us']:
            return 'Ultrasound'
        elif modality_lower in ['x_ray']:
            return 'X-Ray'
        elif modality_lower in ['dermoscopy']:
            return 'Dermoscopy'
        else:
            return 'Other'
    
    # Assign modality groups
    filtered_df['modality_group'] = filtered_df['Modality'].apply(assign_modality_group)
    
    # Print unique modalities and their groups for verification
    print("\nModality groupings:")
    modality_groups = filtered_df.groupby(['Modality', 'modality_group']).size().reset_index(name='count')
    for _, row in modality_groups.iterrows():
        print(f"  {row['Modality']} -> {row['modality_group']} (n={row['count']})")
    
    # Calculate mean IoU per modality group for each model
    all_model_stats = {}
    for model in unique_models:
        model_df = filtered_df[filtered_df['Model'] == model]
        group_stats = model_df.groupby('modality_group')['IoU'].agg(['mean', 'std', 'count']).reset_index()
        group_stats.columns = ['modality_group', 'mean_iou', 'std_iou', 'count']
        group_stats = group_stats.sort_values('mean_iou', ascending=False)
        all_model_stats[model] = group_stats
        
        print(f"\nMean IoU by modality group for {model}:")
        for _, row in group_stats.iterrows():
            print(f"  {row['modality_group']}: {row['mean_iou']:.4f} ± {row['std_iou']:.4f} (n={row['count']})")
    
    # Check if we have data to plot
    if len(all_model_stats) == 0:
        print("No data available for plotting!")
        return
    
    # Create radar plot
    create_radar_plot(all_model_stats, script_dir)

def create_radar_plot(all_model_stats, output_dir):
    # Set up the figure and polar plot
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
    
    # Get all unique modality groups across all models
    all_modalities = set()
    for model_stats in all_model_stats.values():
        all_modalities.update(model_stats['modality_group'].tolist())
    all_modalities = sorted(list(all_modalities))
    
    # Number of modality groups
    N = len(all_modalities)
    
    # Compute angle for each modality group
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    # Define colors for different models
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    # Plot each model
    for i, (model, group_stats) in enumerate(all_model_stats.items()):
        # Create a mapping of modality to IoU for this model
        modality_to_iou = dict(zip(group_stats['modality_group'], group_stats['mean_iou']))
        
        # Values to plot (mean IoU) - use 0 if modality not present for this model
        values = [modality_to_iou.get(modality, 0) for modality in all_modalities]
        values += values[:1]  # Complete the circle
        
        # Plot the radar chart for this model
        color = colors[i % len(colors)]
        ax.plot(angles, values, 'o-', linewidth=2, label=model, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)
    
    # Add labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(all_modalities, fontsize=18)
    
    # Set y-axis limits and labels
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=16)
    ax.grid(True)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save in multiple formats
    formats = ['png', 'pdf', 'svg']
    for fmt in formats:
        output_path = os.path.join(output_dir, f'modality_radar_plot_all_models.{fmt}')
        plt.savefig(output_path, format=fmt, dpi=300, bbox_inches='tight')
        print(f"Radar plot saved as: {output_path}")
    
    plt.show()

if __name__ == "__main__":
    main()