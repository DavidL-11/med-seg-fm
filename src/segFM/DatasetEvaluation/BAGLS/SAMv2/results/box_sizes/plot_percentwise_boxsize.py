import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def extract_percentage_value(bbsize_str):
    """Extract numeric value from percentage string (e.g., '+10%' -> 10, '-15%' -> -15)"""
    if bbsize_str == '+0%' or bbsize_str == '-0%':
        return 0
    return int(bbsize_str[1:-1]) * (1 if bbsize_str[0] == '+' else -1)

def main():
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'BAGLS_boxsize_percent.csv')
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Extract numeric percentage values
    df['bbsize_numeric'] = df['bbsize'].apply(extract_percentage_value)
    
    # Group by box size and calculate average IoU and NSD
    grouped = df.groupby('bbsize_numeric').agg({
        'IoU': 'mean',
        'NSD': 'mean'
    }).reset_index()
    
    # Sort by box size percentage
    grouped = grouped.sort_values('bbsize_numeric')
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Plot both metrics
    ax.plot(grouped['bbsize_numeric'], grouped['IoU'], 
            marker='o', linewidth=2, markersize=6, label='Average IoU', color='#1f77b4')
    ax.plot(grouped['bbsize_numeric'], grouped['NSD'], 
            marker='s', linewidth=2, markersize=6, label='Average NSD', color='#ff7f0e')
    
    # Customize the plot
    ax.set_xlabel('Box Size Difference (%)', fontsize=14)
    ax.set_ylabel('Performance Score', fontsize=14)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Set x-axis ticks to show percentage labels with better spacing
    x_ticks = grouped['bbsize_numeric'].values
    
    # Create more readable tick spacing - show every 5th or 10th value depending on range
    x_range = x_ticks.max() - x_ticks.min()
    if x_range > 100:
        # For large ranges, show every 20%
        major_ticks = [x for x in x_ticks if x % 20 == 0]
        minor_ticks = [x for x in x_ticks if x % 10 == 0 and x % 20 != 0]
    elif x_range > 50:
        # For medium ranges, show every 10%
        major_ticks = [x for x in x_ticks if x % 10 == 0]
        minor_ticks = [x for x in x_ticks if x % 5 == 0 and x % 10 != 0]
    else:
        # For small ranges, show every 5%
        major_ticks = [x for x in x_ticks if x % 5 == 0]
        minor_ticks = [x for x in x_ticks if x not in major_ticks]
    
    # Set major ticks with labels
    ax.set_xticks(major_ticks)
    ax.set_xticklabels([f'{x:+d}%' if x != 0 else '0%' for x in major_ticks], 
                       rotation=0, ha='center', fontsize=10)
    
    # Set minor ticks without labels
    ax.set_xticks(minor_ticks, minor=True)
    ax.tick_params(axis='x', which='minor', length=3)
    ax.tick_params(axis='x', which='major', length=6)
    
    # Add legend
    ax.legend(loc='best', fontsize=11)
    
    # Set axis limits for better visualization
    ax.set_ylim(0, 1)
    ax.set_xlim(-50, 100)

    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.7)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot in multiple formats
    output_dir = script_dir
    plot_basename = 'bagls_boxsize_performance'
    
    plt.savefig(os.path.join(output_dir, f'{plot_basename}.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f'{plot_basename}.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f'{plot_basename}.svg'), bbox_inches='tight')
    
    # Show summary statistics
    print("Summary Statistics:")
    print("==================")
    print(f"Number of box size variations: {len(grouped)}")
    print(f"Box size range: {grouped['bbsize_numeric'].min()}% to {grouped['bbsize_numeric'].max()}%")
    print(f"Best IoU: {grouped['IoU'].max():.4f} at {grouped.loc[grouped['IoU'].idxmax(), 'bbsize_numeric']:+d}%")
    print(f"Best NSD: {grouped['NSD'].max():.4f} at {grouped.loc[grouped['NSD'].idxmax(), 'bbsize_numeric']:+d}%")
    print(f"IoU at 0%: {grouped.loc[grouped['bbsize_numeric'] == 0, 'IoU'].iloc[0]:.4f}")
    print(f"NSD at 0%: {grouped.loc[grouped['bbsize_numeric'] == 0, 'NSD'].iloc[0]:.4f}")
    
    # Print detailed results table
    print("\nDetailed Results:")
    print("=================")
    print(grouped.to_string(index=False, float_format='%.4f'))
    
    plt.show()

if __name__ == "__main__":
    main()