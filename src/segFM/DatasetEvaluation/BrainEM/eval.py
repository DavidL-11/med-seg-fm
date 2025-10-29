import numpy as np

from BOB import BOB

from segFM.DataLoaders.brain_em import BrainMicroscopy
from segFM.predictors.sam2_2d import SAM2ImageSegmenter
from segFM.logger import logger
from segFM import checkpoints, utils

N_POS = 1  # Number of positive prompts
N_NEG = 0  # Number of negative prompts
MODE = "box"  # "point", "box", "yolo"
BBSIZE = 0  # 0 for no bounding box, >0 for bounding
PROMPT_FINDER = "BOB YOLOv12n"  # "random", "box", "BOB D-FINE-N"

bob = BOB(model="YOLOv12n")

# Initialize the Endoscapes dataset
dataset = BrainMicroscopy(plot_prompts=False, mode=MODE,
                          n_pos=N_POS, n_neg=N_NEG, bbsize=BBSIZE,
                          prompt_finder=PROMPT_FINDER,
                          bob=bob)

N_IMAGES = 100  # Number of images to evaluate

# Create a SAM predictor object
sam_predictor = SAM2ImageSegmenter(
    checkpoint=checkpoints.MedSAM2_latest, #checkpoints.SAM2_tiny,
    config=checkpoints.MedSAM_cfg, #checkpoints.SAM2_tiny_cfg,
    fine_tuned_weights=None,
    postprocessing=True,
)

# Evaluate the model's performance on a large number of images
df = sam_predictor.evaluate_model(dataset=dataset, n_images=N_IMAGES, plot_results=False)

print(df)

utils.save_dataframe_to_csv(__file__, df, "results/brainem.csv")
