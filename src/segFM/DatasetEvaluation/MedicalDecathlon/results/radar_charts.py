import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from math import pi

FONTSIZE = 24

def create_radar_charts(metric='DSC'):
    """Create radar charts showing performance of different modes per Medical model.
    
    Args:
        metric (str): Either 'DSC' or 'NSD' to specify which metric to plot
    """
    
    # Get the directory of current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'msd.csv')
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Filter for models containing "Med" or "VISTA"
    med_models = df[df['Model'].str.contains('Med|VISTA', na=False)].copy()

    # For rows where "Med" is in model, remove objects in ["edema", "non-enhancing tumor", "enhancing tumor"]
    brain_tumor_objects = ['edema', 'non-enhancing tumor', 'enhancing tumor']
    med_models = med_models[~((med_models['Model'].str.contains('Med', na=False)) & (med_models['Object'].isin(brain_tumor_objects)))]

    # For rows where VISTA is in model, remove objects in ["Prostate", "Glioma", "Hippocampus"]
    vista_exclude_objects = ['Prostate', 'Glioma', 'Hippocampus']
    med_models = med_models[~((med_models['Model'].str.contains('VISTA', na=False)) & (med_models['Object'].isin(vista_exclude_objects)))]
    

    if med_models.empty:
        print("No models containing 'Med' or 'VISTA' found in the dataset.")
        return
    
    # Validate metric parameter
    if metric not in ['DSC', 'NSD']:
        print(f"Invalid metric '{metric}'. Must be 'DSC' or 'NSD'.")
        return
    
    # Create shortened model names
    med_models['Model_Short'] = med_models['Model'].str.replace('.pt', '')
    
    # Group by Model, Mode, and Object to calculate mean scores
    grouped = med_models.groupby(['Model_Short', 'Mode', 'Object'])[['DSC', 'NSD']].median().reset_index()
    
    # Get unique models and objects
    models = grouped['Model_Short'].unique()
    objects = grouped['Object'].unique()
    modes = grouped['Mode'].unique()
    
    print(f"Found models: {models}")
    print(f"Found modes: {modes}")
    print(f"Found objects: {objects}")
    print(f"Creating radar charts for metric: {metric}")
    
    # Create radar charts for all models side by side
    create_models_radar_chart(grouped, models, objects, modes, metric, script_dir)

def create_models_radar_chart(data, models, objects, modes, metric, output_dir):
    """Create radar charts for all models side by side, each showing different modes."""
    
    # Set up the figure with subplots for each model
    num_models = len(models)
    fig, axes = plt.subplots(1, num_models, figsize=(10 * num_models, 10), subplot_kw=dict(projection='polar'))
    
    # If only one model, make axes a list for consistency
    if num_models == 1:
        axes = [axes]
    
    # Number of variables (objects)
    N = len(objects)
    
    # Compute angle for each axis
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    # Colors for different modes (using high contrast colors for print)
    colors = ["#3B087AA1", "#A5545D98", '#A23B72', '#C73E1D', '#592E83', '#0F4C75']
    mode_colors = {mode: colors[i % len(colors)] for i, mode in enumerate(modes)}
    
    # Create a chart for each model
    for model_idx, model in enumerate(models):
        ax = axes[model_idx]
        
        # Filter data for this model
        model_data = data[data['Model_Short'] == model]
        
        ax.set_title(f'{model} - Median {metric}', size=26, fontweight='bold', pad=30)
        
        # Add object labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(objects, fontsize=FONTSIZE, fontweight='bold')
        
        # Set y-axis limits
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=FONTSIZE, fontweight='bold')
        ax.grid(True, alpha=0.5, linewidth=1.5)
        ax.set_facecolor('#f8f9fa')
        
        # Plot data for each mode
        for mode in modes:
            mode_model_data = model_data[model_data['Mode'] == mode]
            
            # Get values for each object
            values = []
            for obj in objects:
                obj_data = mode_model_data[mode_model_data['Object'] == obj]
                if not obj_data.empty:
                    values.append(obj_data[metric].iloc[0])
                else:
                    values.append(0)  # or np.nan for missing data
            
            # Complete the circle
            values += values[:1]
            
            # Plot
            ax.plot(angles, values, 'o-', linewidth=5, label=mode, 
                   color=mode_colors[mode], markersize=12, markeredgewidth=2, markeredgecolor='white')
            ax.fill(angles, values, alpha=0.25, color=mode_colors[mode])
        
        # Add legend only to the last subplot to avoid clutter
        if model_idx == num_models - 1:
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=FONTSIZE, 
                     frameon=True, fancybox=True, shadow=True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the chart
    output_file = os.path.join(output_dir, f'all_models_{metric.lower()}_radar_chart.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Radar chart saved: {output_file}")
    
    # Save as PDF
    output_file_pdf = os.path.join(output_dir, f'all_models_{metric.lower()}_radar_chart.pdf')
    plt.savefig(output_file_pdf, bbox_inches='tight', facecolor='white')
    print(f"Radar chart saved: {output_file_pdf}")
    
    # Close the plot to free memory
    plt.close()
        

if __name__ == "__main__":
    # Set matplotlib backend for better compatibility
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    plt.style.use('default')
    
    # Create radar charts for both metrics
    print("Creating DSC radar charts...")
    create_radar_charts(metric='DSC')
    
    print("\nCreating NSD radar charts...")
    create_radar_charts(metric='NSD')
