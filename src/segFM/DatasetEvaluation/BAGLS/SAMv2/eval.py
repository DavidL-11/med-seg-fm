import numpy as np
import time
import os

from segFM.predictors.sam2_2d import SAM2ImageSegmenter
from segFM.DataLoaders.bagls import BAGLS_Images, BAGLSImagesFull
from segFM import checkpoints, utils
from segFM.logger import logger
from BOB.prompt_generator import BOB
#from segFM.predictors.sam_med2d import SAMMed2DImageSegmenter

BBSIZE = 0  # 0 for no bounding box, >0 for bounding box size
PROMPT_MODEL = "D-FINE-N"  # "D-FINE-N", "YOLOv12n", "YOLOv12s"
PROMPT_FINDER = f"center"  # "yolo", "random", "center", "darkest", "rim"
MODE = f"box"  # "point", "box", "random", "BOB {PROMPT_MODEL}""

N_POS = 1  # Number of positive prompts
N_NEG = 0  # Number of negative prompts
N_IMAGES = 200  # Number of images to evaluate
OUTPUT_FILE = "results/box_sizes/BAGLS_boxsize_percent.csv"

def eval_single_param(mode, prompt_finder, n_pos, n_neg, bbsize):
    prompt_gen = BOB(model=PROMPT_MODEL)
    dataset = BAGLSImagesFull(
        mode=mode,
        prompt_finder=prompt_finder,
        n_pos=n_pos,
        n_neg=n_neg,
        bbsize=bbsize,
        prompt_gen=prompt_gen,
    )

    # Create a SAM predictor object
    sam_predictor = SAM2ImageSegmenter(
        checkpoint=checkpoints.MedSAM2_latest,
        config=checkpoints.MedSAM_cfg,
        fine_tuned_weights=None,#"src/segFM/DatasetEvaluation/BAGLS/SAMv2/finetuned/bagls_tuned_sam2t_1000.torch",
        postprocessing=True,
    )

    # Create a MedSAM2 predictor object
    # sam_predictor = SAM2ImageSegmenter(
    #     checkpoint=checkpoints.MedSAM2_latest,
    #     config=checkpoints.MedSAM_cfg,
    #     fine_tuned_weights=None,
    #     postprocessing=True,
    # )


    # sam_predictor = SAMMed2DImageSegmenter()

    df = sam_predictor.evaluate_model(dataset=dataset, n_images=N_IMAGES, plot_results=False)
    return df


def try_combinations():
    for sign in ["+", "-"]:
        for box_size in ["0%", "1%", "2%", "3%", "4%", "5%", "6%", "7%", "8%", "9%", "10%", "15%", "20%", "25%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "100%"]:
                df = eval_single_param(
                    mode="box",
                    n_pos=N_POS,
                    n_neg=N_NEG,
                    prompt_finder=PROMPT_FINDER,
                    bbsize=sign + box_size,
                )

                # Directory of this file + FLARE22_Multiobject.csv
                utils.save_dataframe_to_csv(
                    __file__,
                    df,
                    OUTPUT_FILE,
                )

if __name__ == "__main__":
    try_combinations()
    # df = eval_single_param(
    #     mode=MODE,
    #     prompt_finder=PROMPT_FINDER,
    #     n_pos=N_POS,
    #     n_neg=N_NEG,
    #     bbsize=BBSIZE
    # )

    # utils.save_dataframe_to_csv(__file__, df, OUTPUT_FILE)
