import numpy as np
import torch

from segFM.DataLoaders.medsegbench import MedSegBench
from segFM.predictors.sam2_2d import SAM2ImageSegmenter
#from segFM.predictors.sam_med2d import SAMMed2DImageSegmenter
from segFM.logger import logger
from segFM import checkpoints, utils

from BOB import BOB

bob = BOB(model="YOLOv12n")  # Initialize BOB prompt generator

CHECKPOINT = checkpoints.SAM2_tiny  # Path to the SAM2 checkpoint
CHECKPOINT_CFG = checkpoints.SAM2_tiny_cfg  # Path to the SAM2 configuration

N_DATASETS = 33  # Number of datasets to evaluate

DATASETS = range(N_DATASETS)  # Dataset IDs to evaluate, from 0 to N_DATASETS-1
N_IMAGES_PER_DATASET = 200  # Number of images per dataset to evaluate

# Set parameters for each dataset
N_POS = [1, 4, 0, 0]  # Number of positive point prompts
N_NEG = [0, 2, 0, 0]  # Number of negative point prompts
MODE = ["point", "point", "BOB YOLOv12n", "box"]
PROMPT_FINDER = ["random", "random", "random", "random"]
BBSIZE = [0, 0, 0, 0]  # Bounding box size for box mode

#sam_predictor = SAMMed2DImageSegmenter() # 1950 MB
sam_predictor = SAM2ImageSegmenter(
    checkpoint=CHECKPOINT,
    config=CHECKPOINT_CFG,
    fine_tuned_weights=None,
    postprocessing=True,
)

for dataset_id in DATASETS:
    logger.info(f"Evaluating dataset {dataset_id}/{N_DATASETS}")

    if dataset_id == 16:  # Skip dataset 16 as it is not available
        continue

    for setting in [2]:
        dataset = MedSegBench(
            dataset_id=dataset_id,
            split="test",
            transform=None,  # No transformation for evaluation
            plot_prompts=False,  # Do not plot prompts during evaluation
            mode=MODE[setting],  # Use the specified mode (point or box)
            prompt_finder=PROMPT_FINDER[setting],  # Method to find prompts
            use_multiprompts=True,
            n_pos=N_POS[setting],  # Number of positive prompts
            n_neg=N_NEG[setting],  # Number of negative prompts
            bbsize=BBSIZE[setting],  # Bounding box size
            prompt_gen=bob,
        )

        # Evaluate the model's performance on a large number of images
        df = sam_predictor.evaluate_model(
            dataset=dataset,
            n_images=min(N_IMAGES_PER_DATASET, dataset.n_images),
            plot_results=False,
            save_dataset_name=True,
        )

        # Insert "Multiprompt" column based on the dataset's prompt type
        df.insert(2, "Multiprompt", dataset.use_multiprompts)

        utils.save_dataframe_to_csv(__file__, df, "results/bob/MedSegBench.csv")
