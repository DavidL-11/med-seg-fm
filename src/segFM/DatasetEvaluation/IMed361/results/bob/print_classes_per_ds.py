import os
import pandas as pd

# Get the current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, 'IMed361_bob.csv')

# Read the CSV file
df = pd.read_csv(csv_path)

# Group by Dataset and get unique objects
dataset_objects = df.groupby('Dataset')['Object'].unique()

# Print results
print("\nUnique objects per dataset:")
print("-" * 50)
for dataset, objects in dataset_objects.items():
    print(f"\nDataset: {dataset}")
    print("Objects:", ", ".join(sorted(objects)))
