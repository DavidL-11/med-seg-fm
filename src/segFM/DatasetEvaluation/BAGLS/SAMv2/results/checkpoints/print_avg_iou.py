import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
    
# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_file = os.path.join(script_dir, 'BAGLS_checkpoints.csv')

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

# Create plots for metrics across all objects (excluding "Unknown")
print("Creating plots for metrics across all objects (excluding 'Unknown')...")

# Filter out "Unknown" objects
df_filtered = df[df['Object'] != 'Unknown'].copy()

if len(df_filtered) == 0:
    print("No data found after filtering out 'Unknown' objects.")
else:
    print(f"Plotting data for {len(df_filtered)} rows across {len(df_filtered['Object'].unique())} objects")
    print(f"Objects included: {sorted(df_filtered['Object'].unique())}")
    
    # Calculate overall statistics for each model-mode combination
    overall_stats = df_filtered.groupby(['Model', 'Mode'])['IoU'].agg([
        ('mean', 'mean'),
        ('std', 'std'),
        ('median', 'median'),
        ('count', 'count')
    ]).reset_index()
    
    # Create model-mode labels for plotting
    overall_stats['Model_Mode'] = overall_stats['Model'] + ' - ' + overall_stats['Mode']
    
    # Sort by mean IoU for better visualization
    overall_stats = overall_stats.sort_values('mean', ascending=True)
    
    # Print overall statistics
    print("\nOverall statistics across all objects (excluding 'Unknown'):")
    for _, row in overall_stats.iterrows():
        print(f"- {row['Model_Mode']}: mIoU = {row['mean']:.3f}, STD = {row['std']:.3f}, "
              f"Median = {row['median']:.3f}, Count = {row['count']}")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('IoU Metrics Across All Objects (Excluding "Unknown")', fontsize=16, fontweight='bold')
    
    # 1. Bar plot of mean IoU with error bars
    ax1 = axes[0, 0]
    bars = ax1.bar(range(len(overall_stats)), overall_stats['mean'], 
                   yerr=overall_stats['std'], capsize=5, alpha=0.7)
    ax1.set_xlabel('Model - Mode')
    ax1.set_ylabel('Mean IoU')
    ax1.set_title('Mean IoU by Model-Mode Combination')
    ax1.set_xticks(range(len(overall_stats)))
    ax1.set_xticklabels(overall_stats['Model_Mode'], rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, mean_val) in enumerate(zip(bars, overall_stats['mean'])):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{mean_val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 2. Box plot of IoU distributions
    ax2 = axes[0, 1]
    df_filtered['Model_Mode'] = df_filtered['Model'] + ' - ' + df_filtered['Mode']
    
    # Create box plot data in the same order as the bar plot
    box_data = []
    box_labels = []
    for _, row in overall_stats.iterrows():
        model_mode = row['Model_Mode']
        data = df_filtered[df_filtered['Model_Mode'] == model_mode]['IoU']
        box_data.append(data)
        box_labels.append(model_mode)
    
    bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True)
    ax2.set_xlabel('Model - Mode')
    ax2.set_ylabel('IoU Distribution')
    ax2.set_title('IoU Distribution by Model-Mode Combination')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Color the boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(bp['boxes'])))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # 3. Violin plot for distribution comparison
    ax3 = axes[1, 0]
    
    # Prepare data for violin plot
    violin_data = []
    violin_positions = []
    violin_labels = []
    
    for i, (_, row) in enumerate(overall_stats.iterrows()):
        model_mode = row['Model_Mode']
        data = df_filtered[df_filtered['Model_Mode'] == model_mode]['IoU']
        if len(data) > 0:
            violin_data.append(data)
            violin_positions.append(i)
            violin_labels.append(model_mode)
    
    if violin_data:
        vp = ax3.violinplot(violin_data, positions=violin_positions, showmeans=True, showmedians=True)
        ax3.set_xlabel('Model - Mode')
        ax3.set_ylabel('IoU')
        ax3.set_title('IoU Distribution Shapes by Model-Mode')
        ax3.set_xticks(violin_positions)
        ax3.set_xticklabels(violin_labels, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
    
    # 4. Sample count per model-mode combination
    ax4 = axes[1, 1]
    bars4 = ax4.bar(range(len(overall_stats)), overall_stats['count'], alpha=0.7)
    ax4.set_xlabel('Model - Mode')
    ax4.set_ylabel('Number of Samples')
    ax4.set_title('Sample Count by Model-Mode Combination')
    ax4.set_xticks(range(len(overall_stats)))
    ax4.set_xticklabels(overall_stats['Model_Mode'], rotation=45, ha='right')
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars4, overall_stats['count'])):
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{int(count)}', ha='center', va='bottom', fontsize=9)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save the plot
    output_file = os.path.join(script_dir, 'overall_metrics_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as: {output_file}")
    
    # Also save as PDF and SVG
    plt.savefig(os.path.join(script_dir, 'overall_metrics_comparison.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(script_dir, 'overall_metrics_comparison.svg'), bbox_inches='tight')
    
    # Show the plot
    plt.show()

print("="*80)