import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# --- Parameters ---
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_file = os.path.join(script_dir, "vfss_heatmap.csv")
METRIC = "NSD"  # Change to "DSC", "NSD", or "IoU"

# --- Load CSV ---
df = pd.read_csv(csv_file)

# --- Get unique models ---
unique_models = df['Model'].unique()

# Set style and font sizes for print-ready formatting
plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.titlesize': 20
})

# --- Create heatmap for each model ---
for model in unique_models:
    # Filter data for current model
    model_data = df[df['Model'] == model]
    
    # Calculate average DSC for each n_pos, n_neg combination
    heatmap_data = model_data.pivot_table(index="n_pos", columns="n_neg", values=METRIC, aggfunc="mean")
    
    # Sort both axes for better readability
    heatmap_data = heatmap_data.sort_index(ascending=False)

    # Create figure
    plt.figure(figsize=(8.3, 6))
    
    # Create heatmap
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt='.2f',
        cmap='viridis',
        square=True,
        linewidths=0.5,
        annot_kws={'fontsize': 14}
    )
    
    # Set labels and title
    plt.xlabel('Number of Negative Prompts (n_neg)', fontsize=18)
    plt.ylabel('Number of Positive Prompts (n_pos)', fontsize=18)
    # plt.title(f'Average {METRIC} Heatmap - {model}', fontsize=18, pad=20)

    # Adjust layout
    plt.tight_layout()
    
    # Create safe filename
    safe_model_name = model.replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '')
    
    # Save in multiple formats with high DPI
    base_filename = f"heatmap_{safe_model_name}"
    plt.savefig(os.path.join(script_dir, f"{base_filename}.pdf"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(script_dir, f"{base_filename}.svg"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(script_dir, f"{base_filename}.png"), dpi=300, bbox_inches='tight')
    
    # Show plot
    plt.show()

print(f"Created heatmaps for {len(unique_models)} model(s)")
