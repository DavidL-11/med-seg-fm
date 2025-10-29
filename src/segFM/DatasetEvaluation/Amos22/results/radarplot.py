import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import pi
import os

def create_radar_plots():
    """
    Create radar plots for each model in the AMOS22.csv file.
    Each radar plot shows one line for each mode with mean DSC values for different objects.
    """
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'AMOS22.csv')
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Get unique models and modes
    models = df['Model'].unique()
    modes = df['Mode'].unique()
    
    print(f"Found {len(models)} models: {models}")
    print(f"Found {len(modes)} modes: {modes}")
    
    # For each model, create a radar plot
    for model in models:
        create_model_radar_plot(df, model, modes, script_dir)

def filter_objects_by_bob_performance(model_data):
    """
    Filter out objects where the mean DSC for all BOB modes is 0.
    """
    # Get all unique objects
    all_objects = model_data['Object'].unique()
    
    # Filter objects based on BOB performance
    filtered_objects = []
    for obj in all_objects:
        # Get data for this object and BOB modes
        obj_data = model_data[model_data['Object'] == obj]
        bob_data = obj_data[obj_data['Mode'].str.contains('BOB', case=False, na=False)]
        
        if len(bob_data) == 0:
            # If no BOB data for this object, include it
            filtered_objects.append(obj)
        else:
            # Calculate mean DSC for BOB modes
            mean_bob_dsc = bob_data['DSC'].mean()
            if mean_bob_dsc > 0:
                filtered_objects.append(obj)
            # If mean_bob_dsc == 0, we exclude this object
    
    return filtered_objects

def create_model_radar_plot(df, model, modes, output_dir):
    """
    Create a radar plot for a specific model showing all modes.
    """
    # Filter data for the current model
    model_data = df[df['Model'] == model]
    
    # Filter out objects where mean DSC for BOB modes is 0
    objects = filter_objects_by_bob_performance(model_data)
    N = len(objects)
    
    # Calculate angles for each axis
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Colors for different modes
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']
    
    # For each mode, calculate mean DSC values and plot
    for i, mode in enumerate(modes):
        mode_data = model_data[model_data['Mode'] == mode]
        
        if len(mode_data) == 0:
            continue
            
        # Calculate mean DSC for each filtered object
        mean_dsc_values = []
        for obj in objects:
            obj_data = mode_data[mode_data['Object'] == obj]
            if len(obj_data) > 0:
                mean_dsc = obj_data['DSC'].mean()
            else:
                mean_dsc = 0
            mean_dsc_values.append(mean_dsc)
        
        # Close the polygon
        values = mean_dsc_values + [mean_dsc_values[0]]
        
        # Plot the line
        color = colors[i % len(colors)]
        ax.plot(angles, values, 'o-', linewidth=2, label=mode, color=color)
        ax.fill(angles, values, alpha=0.25, color=color)
    
    # Customize the plot
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(objects, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
    ax.grid(True)
    
    # Add title and legend
    model_name = model.replace('.pt', '').replace('_', ' ')
    plt.title(f'Mean DSC Performance by Mode\n{model_name}', size=16, fontweight='bold', pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    # Save the plot
    output_filename = f"{model.replace('.pt', '')}_radar_plot.png"
    output_path = os.path.join(output_dir, output_filename)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Radar plot saved as: {output_path}")
    
    # Print summary statistics
    print(f"\nSummary for {model}:")
    for mode in modes:
        mode_data = model_data[model_data['Mode'] == mode]
        if len(mode_data) > 0:
            overall_mean_dsc = mode_data['DSC'].mean()
            print(f"  {mode}: Overall Mean DSC = {overall_mean_dsc:.4f}")

def print_data_summary():
    """
    Print a summary of the data structure.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'AMOS22.csv')
    
    df = pd.read_csv(csv_path)
    
    print("=== Data Summary ===")
    print(f"Total rows: {len(df)}")
    print(f"Models: {df['Model'].unique()}")
    print(f"Modes: {df['Mode'].unique()}")
    print(f"Objects: {sorted(df['Object'].unique())}")
    print(f"Images: {sorted(df['Image'].unique())}")
    
    # Show sample of mean DSC by mode and object
    print("\n=== Mean DSC by Mode and Object ===")
    for mode in df['Mode'].unique():
        print(f"\n{mode}:")
        mode_data = df[df['Mode'] == mode]
        mean_by_object = mode_data.groupby('Object')['DSC'].mean().sort_values(ascending=False)
        for obj, mean_dsc in mean_by_object.items():
            print(f"  {obj}: {mean_dsc:.4f}")

if __name__ == "__main__":
    print("Creating radar plots for AMOS22 dataset evaluation...")
    print_data_summary()
    print("\n" + "="*50)
    create_radar_plots()
    print("Done!")
