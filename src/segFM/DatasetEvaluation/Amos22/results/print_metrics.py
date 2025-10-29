import pandas as pd
import os

# Load the CSV file from the same directory
csv_path = os.path.join(os.path.dirname(__file__), 'AMOS22.csv')
df = pd.read_csv(csv_path)

print("Original DataFrame:")
print(df.head())
print(f"Total rows: {len(df)}")
print("\n" + "="*50 + "\n")

# Group images into MRI and CT groups
df['Image_Group'] = df['Image'].apply(lambda x: 'MRI Images' if 55 <= x <= 74 else 'CT Images')

# Print the count in each group
image_group_counts = df['Image_Group'].value_counts()
print("Image Group Counts:")
for group, count in image_group_counts.items():
    print(f"{group}: {count}")
print("\n" + "="*50 + "\n")

# Create Model-Mode combination
df['Model_Mode'] = df['Model'] + ' (' + df['Mode'] + ')'

# Get unique Model-Mode combinations
unique_model_modes = df['Model_Mode'].unique()
print("Unique Model-Mode combinations:")
for combo in sorted(unique_model_modes):
    print(f"  {combo}")
print(f"\nTotal unique combinations: {len(unique_model_modes)}")
print("\n" + "="*50 + "\n")

# Print the full dataframe with new columns
print("DataFrame with groupings:")
print(df.to_string())
print("\n" + "="*50 + "\n")

# Calculate and print mean and median values for DSC and NSD
print("Overall Statistics:")
print(f"DSC - Mean: {df['DSC'].mean():.4f}, Median: {df['DSC'].median():.4f}")
print(f"NSD - Mean: {df['NSD'].mean():.4f}, Median: {df['NSD'].median():.4f}")
print("\n" + "="*30 + "\n")

# Statistics by Image Group
print("Statistics by Image Group:")
for group in ['MRI Images', 'CT Images']:
    group_data = df[df['Image_Group'] == group]
    if len(group_data) > 0:
        print(f"\n{group}:")
        print(f"  DSC - Mean: {group_data['DSC'].mean():.4f}, Median: {group_data['DSC'].median():.4f}")
        print(f"  NSD - Mean: {group_data['NSD'].mean():.4f}, Median: {group_data['NSD'].median():.4f}")

print("\n" + "="*30 + "\n")

# Statistics by Model-Mode combination
print("Statistics by Model-Mode combination:")
for combo in sorted(unique_model_modes):
    combo_data = df[df['Model_Mode'] == combo]
    print(f"\n{combo}:")
    print(f"  Count: {len(combo_data)}")
    print(f"  DSC - Mean: {combo_data['DSC'].mean():.4f}, Median: {combo_data['DSC'].median():.4f}")
    print(f"  NSD - Mean: {combo_data['NSD'].mean():.4f}, Median: {combo_data['NSD'].median():.4f}")
