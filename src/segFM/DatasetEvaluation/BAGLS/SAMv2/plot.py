import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2, os

csv_file = "BAGLS_full.csv"
csv_path = os.path.join(os.path.dirname(__file__), csv_file)

metrics = pd.read_csv(csv_path)

# Get all the values which have Model=SAM2
metrics = metrics.groupby("Model")

yolo_metrics = metrics.get_group("SAM2t YOLO")
box_metrics = metrics.get_group("SAM2t Box")
point_metrics = metrics.get_group("SAM2t Point")

# Create a table with the metrics, round the values to 3 decimal places
yolo_metrics = yolo_metrics.round(3)
box_metrics = box_metrics.round(3)
point_metrics = point_metrics.round(3)

objects = ["Glottis", "Left Vocal Cord", "Right Vocal Cord"]



# Get the mean DSC for each object in the YOLO metrics
table_data = {
    "Model": ["SAM2t YOLO Glottis", "SAM2t YOLO Left Vocal Cord", "SAM2t YOLO Right Vocal Cord",
                "SAM2t Box Glottis", "SAM2t Box Left Vocal Cord", "SAM2t Box Right Vocal Cord",
                "SAM2t Point Glottis", "SAM2t Point Left Vocal Cord", "SAM2t Point Right Vocal Cord"],
    "mDSC": [
        round(yolo_metrics[yolo_metrics["Object"] == "Glottis"]["DSC"].mean(), 3),
        round(yolo_metrics[yolo_metrics["Object"] == "Left Vocal Cord"]["DSC"].mean(), 3),
        round(yolo_metrics[yolo_metrics["Object"] == "Right Vocal Cord"]["DSC"].mean(), 3),
        round(box_metrics[box_metrics["Object"] == "Glottis"]["DSC"].mean(), 3),
        round(box_metrics[box_metrics["Object"] == "Left Vocal Cord"]["DSC"].mean(), 3),
        round(box_metrics[box_metrics["Object"] == "Right Vocal Cord"]["DSC"].mean(), 3),
        round(point_metrics[point_metrics["Object"] == "Glottis"]["DSC"].mean(), 3),
        round(point_metrics[point_metrics["Object"] == "Left Vocal Cord"]["DSC"].mean(), 3),
        round(point_metrics[point_metrics["Object"] == "Right Vocal Cord"]["DSC"].mean(), 3),
    ],
    "mNSD": [
        round(yolo_metrics[yolo_metrics["Object"] == "Glottis"]["NSD"].mean(), 3),
        round(yolo_metrics[yolo_metrics["Object"] == "Left Vocal Cord"]["NSD"].mean(), 3),
        round(yolo_metrics[yolo_metrics["Object"] == "Right Vocal Cord"]["NSD"].mean(), 3),
        round(box_metrics[box_metrics["Object"] == "Glottis"]["NSD"].mean(), 3),
        round(box_metrics[box_metrics["Object"] == "Left Vocal Cord"]["NSD"].mean(), 3),
        round(box_metrics[box_metrics["Object"] == "Right Vocal Cord"]["NSD"].mean(), 3),
        round(point_metrics[point_metrics["Object"] == "Glottis"]["NSD"].mean(), 3),
        round(point_metrics[point_metrics["Object"] == "Left Vocal Cord"]["NSD"].mean(), 3),
        round(point_metrics[point_metrics["Object"] == "Right Vocal Cord"]["NSD"].mean(), 3),
    ],
    "mIoU": [
        round(yolo_metrics[yolo_metrics["Object"] == "Glottis"]["IoU"].mean(), 3),
        round(yolo_metrics[yolo_metrics["Object"] == "Left Vocal Cord"]["IoU"].mean(), 3),
        round(yolo_metrics[yolo_metrics["Object"] == "Right Vocal Cord"]["IoU"].mean(), 3),
        round(box_metrics[box_metrics["Object"] == "Glottis"]["IoU"].mean(), 3),
        round(box_metrics[box_metrics["Object"] == "Left Vocal Cord"]["IoU"].mean(), 3),
        round(box_metrics[box_metrics["Object"] == "Right Vocal Cord"]["IoU"].mean(), 3),
        round(point_metrics[point_metrics["Object"] == "Glottis"]["IoU"].mean(), 3),
        round(point_metrics[point_metrics["Object"] == "Left Vocal Cord"]["IoU"].mean(), 3),
        round(point_metrics[point_metrics["Object"] == "Right Vocal Cord"]["IoU"].mean(), 3),
    ]
}
table_df = pd.DataFrame(table_data)
table_df.style.highlight_max(axis=0, color='lightgreen', props='font-weight: bold;')  # Highlight max values
# Plot the metrics
fig, ax = plt.subplots(figsize=(8, 4))
table = ax.table(cellText=table_df.values, colLabels=table_df.columns, cellLoc='center', loc='center')

# Color all cells with the same Model (e.g. SAM2t YOLO) in the same color
for i, model in enumerate(table_df['Model']):
    if "YOLO" in model:
        color = 'lightblue'
    elif "Box" in model:
        color = 'lightgreen'
    elif "Point" in model:
        color = 'lightcoral'
    else:
        color = 'white'
    
    for j in range(len(table_df.columns)):
        table[(i+1, j)].set_facecolor(color)
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.2)
ax.axis('off')
plt.title("BAGLS Dataset Evaluation Metrics")
plt.tight_layout()
plt.show()