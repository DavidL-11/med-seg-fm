import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Set font sizes
plt.rcParams.update({'font.size': 14})
TITLE_SIZE = 16
LABEL_SIZE = 16

# Read the CSV file
current_dir = os.path.dirname(__file__)
csv_path = os.path.join(current_dir, 'endoscapes.csv')
df = pd.read_csv(csv_path)

# Create a unique configuration identifier
df['ModelMode'] = df['Model'] + ' (' + df['Mode'] + ')'

# Get unique objects and model modes combinations
objects = df['Object'].unique()
# Update the order in the dataframe to match our rotated objects
df['Object'] = pd.Categorical(df['Object'], categories=objects, ordered=True)
model_modes_combinations = df['ModelMode'].unique()

# Calculate mean IoU and NSD for each configuration and object
mean_iou = pd.pivot_table(df, values='IoU', index='ModelMode', columns='Object', aggfunc='mean')
mean_nsd = pd.pivot_table(df, values='NSD', index='ModelMode', columns='Object', aggfunc='mean')

print(mean_iou)

N = len(objects)
theta = np.linspace(0, 2*np.pi, N, endpoint=False)
theta = np.roll(theta, 2)  # Rotate the angles to match the rotated objects
theta = np.concatenate((theta, [theta[0]]))  # Complete the circle

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), subplot_kw=dict(projection='polar'))

ax1.set_theta_zero_location('N')
ax2.set_theta_zero_location('N')


# Store lines for legend
lines = []
labels = []

# Plot IoU
for config in model_modes_combinations:
    values = mean_iou.loc[config].values

    values = np.concatenate((values, [values[0]]))  # Complete the circle
    line = ax1.plot(theta, values, 'o-', linewidth=2, label=config)[0]
    ax1.fill(theta, values, alpha=0.25)
    if config not in labels:
        lines.append(line)
        labels.append(config)

ax1.set_title('Intersection over Union (IoU)', fontsize=TITLE_SIZE, pad=40)
ax1.set_xticks(theta[:-1])
ax1.set_xticklabels(objects, fontsize=LABEL_SIZE)
ax1.tick_params(axis='y', labelsize=LABEL_SIZE)
ax1.set_ylim(0, 1)

# Plot NSD
for config in model_modes_combinations:
    values = mean_nsd.loc[config].values
    values = np.concatenate((values, [values[0]]))  # Complete the circle
    ax2.plot(theta, values, 'o-', linewidth=2, label=config)
    ax2.fill(theta, values, alpha=0.25)

ax2.set_title('Normalized Surface Distance (NSD)', fontsize=TITLE_SIZE, pad=40)
ax2.set_xticks(theta[:-1])
ax2.set_xticklabels(objects, fontsize=LABEL_SIZE)
ax2.tick_params(axis='y', labelsize=LABEL_SIZE)
ax2.set_ylim(0, 1)

# Add single legend with better visibility
legend = fig.legend(lines, labels, 
                   loc='center left', 
                   bbox_to_anchor=(1.05, 0.5),
                   fontsize=18,
                   frameon=True,
                   facecolor='white',
                   edgecolor='black')

# Adjust layout with more space for legend
plt.tight_layout()
# Save in the same directory as the script
plt.savefig(os.path.join(current_dir, 'endoscapes_radar.png'), bbox_inches='tight', dpi=300)
plt.savefig(os.path.join(current_dir, 'endoscapes_radar.pdf'), bbox_inches='tight', dpi=300)
plt.savefig(os.path.join(current_dir, 'endoscapes_radar.svg'), bbox_inches='tight', dpi=300)
plt.show()
plt.close()
