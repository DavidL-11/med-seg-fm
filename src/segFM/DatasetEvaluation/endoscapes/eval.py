import numpy as np

from segFM.DataLoaders.endoscapes import EndoscapesDataset
from segFM.predictors.sam2_2d import SAM2ImageSegmenter
from segFM.logger import logger
from segFM import checkpoints, utils

N_POS = 1  # Number of positive prompts
N_NEG = 0  # Number of negative prompts
MODE = "box"  # "point", "box", "random"
BBSIZE = 0  # 0 for no bounding box, >0 for bounding
PROMPT_FINDER = "random"  # "center", "random", "BOB"

# Initialize the Endoscapes dataset
dataset = EndoscapesDataset(
    bbsize=BBSIZE,
    mode=MODE,
    prompt_finder=PROMPT_FINDER,
    n_pos=N_POS,
    n_neg=N_NEG,
)

N_IMAGES = dataset.n_images  # Number of images to evaluate

# Create a SAM predictor object
sam_predictor = SAM2ImageSegmenter(
    checkpoint=checkpoints.MedSAM2_latest, #SAM2_tiny, MedSAM2_latest...
    config=checkpoints.MedSAM_cfg,
    fine_tuned_weights=None,
    postprocessing=True,
)
# Evaluate the model's performance on a large number of images
df = sam_predictor.evaluate_model(dataset=dataset, n_images=10, plot_results=True)

print(df)

#utils.save_dataframe_to_csv(__file__, df, "results/endoscapes.csv")
