import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

METRIC = 'DSC'  # Change to 'NSD' to plot NSD instead of DSC

# Load the CSV file from the same directory
csv_path = os.path.join(os.path.dirname(__file__), 'AMOS22_windowing.csv')
df = pd.read_csv(csv_path)

# Remove all rows where Mode contains "BOB"
# df = df[~df['Mode'].str.contains("BOB")]

# Create Model-Mode combination for grouping
df['Model_Mode'] = df['Model'] + ' (' + df['Mode'] + ')'

# Set up the plot
plt.figure(figsize=(16, 8))

# Print the mean and median DSC across all objects, for each Model-Mode combination
mean_dsc = df.groupby('Model_Mode')[METRIC].mean().reset_index()
median_dsc = df.groupby('Model_Mode')[METRIC].median().reset_index()
print(f"Mean {METRIC} by Model-Mode combination:")
print(mean_dsc)
print(f"\nMedian {METRIC} by Model-Mode combination:")
print(median_dsc)

# Create a seaborn barplot with built-in error bars
ax = sns.barplot(
    data=df, 
    x='Object', 
    y=METRIC, 
    hue='Model_Mode',
    palette='Set2',
)

# Customize the plot
plt.xlabel('Object', fontsize=12, fontweight='bold')
plt.ylabel(f'Mean {METRIC} Score', fontsize=12, fontweight='bold')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Adjust legend
plt.legend(title='Model-Mode', bbox_to_anchor=(1.05, 1), loc='upper left')

# Set y-axis limits to show full range
plt.ylim(0, 1.0)

# Add grid for better readability
plt.grid(True, alpha=0.3, axis='y')

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot
output_path = os.path.join(os.path.dirname(__file__), 'dsc_barplot.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')

# Show the plot
plt.show()

# Print summary statistics
print("\nSummary Statistics:")
print(f"Total number of objects: {len(df['Object'].unique())}")
print(f"Total number of Model-Mode combinations: {len(df['Model_Mode'].unique())}")
print(f"Objects evaluated: {', '.join(sorted(df['Object'].unique()))}")
print(f"Model-Mode combinations: {', '.join(sorted(df['Model_Mode'].unique()))}")