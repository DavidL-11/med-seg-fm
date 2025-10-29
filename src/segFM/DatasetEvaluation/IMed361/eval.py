import numpy as np
from segFM.DataLoaders.imed361 import IMed361
from segFM.predictors.sam2_2d import SAM2ImageSegmenter
#from segFM.predictors.sam_med2d import SAMMed2DImageSegmenter
from segFM.logger import logger
from segFM import checkpoints, utils
from BOB import BOB

N_DATASETS = 57  # Number of datasets to evaluate
DATASETS = range(N_DATASETS) # Dataset IDs to evaluate (0 to N_DATASETS-1)
N_IMAGES_PER_DATASET = 20  # Number of images per dataset to evaluate

# Set parameters for each dataset
N_POS = [1, 4, 0, 0]  # Number of positive point prompts
N_NEG = [0, 2, 0, 0]  # Number of negative point prompts
MODE = ["point", "point", "box", "box"]
MODE = ["random", "random", "BOB YOLOv12n", "random"]
PROMPT_FINDER = ["random", "random", "BOB YOLOv12n", "random"]
BBSIZE = [0, 0, 0, "10%"]  # Bounding box size for box mode

bob = BOB(model="YOLOv12n")

# sam_predictor = SAMMed2DImageSegmenter()
sam_predictor = SAM2ImageSegmenter( # 652 MB MedSAM2, 926 MB SAM2.1 Tiny, 1966 MB SAM-Med2D
    checkpoint=checkpoints.SAM2_tiny,
    config=checkpoints.SAM2_tiny_cfg,
    fine_tuned_weights=None,
    postprocessing=True,
)


for dataset_id in DATASETS:
    logger.info(f"Evaluating dataset {dataset_id}/{N_DATASETS}")
    for setting in [2]:
        dataset = IMed361(
            plot_prompts=False,
            dataset_id=dataset_id,
            mode=MODE[setting],
            prompt_finder=PROMPT_FINDER[setting],
            n_pos=N_POS[setting],
            n_neg=N_NEG[setting],
            bbsize=BBSIZE[setting],
            prompt_gen=bob
        )

        # Evaluate the model's performance on a large number of images
        df = sam_predictor.evaluate_model(
            dataset=dataset,
            n_images=min(N_IMAGES_PER_DATASET, dataset.n_images),
            plot_results=False,
            save_dataset_name=True,
        )

        # Insert the dataset ID into the DataFrame
        df.insert(2, "Modality", dataset.modality)

        print(df)

        utils.save_dataframe_to_csv(__file__, df, "results/bob/IMed361_bob.csv")
