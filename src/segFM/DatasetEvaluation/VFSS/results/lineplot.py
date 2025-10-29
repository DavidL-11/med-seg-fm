import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the CSV file from the same folder as this script
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'vfss_promptstep.csv')

# Read the CSV file
df = pd.read_csv(csv_path)

# Calculate mean IoU and NSD for each model and prompt_step combination
grouped_data = df.groupby(['Model', 'prompt_step']).agg({
    'IoU': 'mean',
    'NSD': 'mean'
}).reset_index()

# Reshape the data for plotting
# Create separate rows for IoU and NSD metrics
dsc_data = grouped_data.copy()
dsc_data['Metric'] = 'IoU'
dsc_data['Value'] = dsc_data['IoU']

nsd_data = grouped_data.copy()
nsd_data['Metric'] = 'NSD'
nsd_data['Value'] = nsd_data['NSD']

# Combine the data
plot_data = pd.concat([dsc_data[['Model', 'prompt_step', 'Metric', 'Value']], 
                      nsd_data[['Model', 'prompt_step', 'Metric', 'Value']]], 
                     ignore_index=True)

# Create the lineplot
plt.figure(figsize=(8, 4))
sns.lineplot(data=plot_data, x='prompt_step', y='Value', 
             hue='Model', style='Metric', markers=True, markersize=8)

# Customize the plot
plt.xlabel('Prompt Step (log scale)', fontsize=16)
plt.ylabel('Score', fontsize=16)
plt.xscale('log')
# Set custom x-axis ticks to show actual values instead of 10^N
prompt_steps = sorted(df['prompt_step'].unique())
plt.xticks(prompt_steps, [str(x) for x in prompt_steps], fontsize=14)
plt.yticks(fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the plot
output_path = os.path.join(script_dir, 'vfss_performance_plot')
plt.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight')
plt.savefig(f"{output_path}.pdf", dpi=300, bbox_inches='tight')
plt.savefig(f"{output_path}.svg", dpi=300, bbox_inches='tight')
plt.show()