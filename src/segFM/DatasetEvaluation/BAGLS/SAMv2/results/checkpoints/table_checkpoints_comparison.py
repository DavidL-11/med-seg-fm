import pandas as pd
import os

# Get current script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load CSVs
csv1_path = os.path.join(script_dir, "BAGLS_checkpoints.csv")
csv2_path = os.path.join(script_dir, "BAGLS_time_1k.csv")

csv1 = pd.read_csv(csv1_path)
csv2 = pd.read_csv(csv2_path)

# Merge on Model to include Time
merged = pd.merge(csv1, csv2, on="Model", how="left")

# Only keep rows where Object=='Glottis'
merged = merged[merged["Object"] == "Glottis"]

# Compute average IoU per (Model, Mode)
avg_iou = merged.groupby(["Model", "Mode", "Time"], as_index=False)["IoU"].mean()

# Sort by Time ascending
avg_iou = avg_iou.sort_values(by="Time")

# Pivot to have Models as rows and Modes as columns
pivot_df = avg_iou.pivot(index="Model", columns="Mode", values="IoU")

# Add Time column for reference
pivot_df = pivot_df.copy()
pivot_df["Time"] = avg_iou.groupby("Model")["Time"].first().values

# Generate LaTeX table
latex_table = pivot_df.to_latex(float_format="{:.4f}".format, caption="Average IoU per Model and Mode", label="tab:avg_dsc")

# Save LaTeX table to file
output_path = os.path.join(script_dir, "avg_iou_table.tex")
with open(output_path, "w") as f:
    f.write(latex_table)

print("LaTeX table saved to", output_path)