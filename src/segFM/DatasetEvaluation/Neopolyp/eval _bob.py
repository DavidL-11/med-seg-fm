import numpy as np

from segFM.DataLoaders.refuge2 import Refuge2Dataset
from segFM.predictors.sam2_2d import SAM2ImageSegmenter
from segFM.DataLoaders.neopolyp import NeopolypDataset
from segFM import checkpoints, utils
from segFM.logger import logger

from BOB.prompt_generator import BOB

PROMPT_FINDER = "random"  # random, center
PROMPT_GEN_MODEL = "YOLOv12n"
MODE = f"BOB {PROMPT_GEN_MODEL}"  # point, box, BOB {PROMPT_GEN_MODEL}
N_POS = 1 #>= 1
N_NEG = 0 #>= 0
BBSIZE = 0  # Bounding box size, 0 for no additional padding

CHECKPOINT = checkpoints.SAM2_tiny
CONFIG = checkpoints.SAM2_tiny_cfg

dataset = NeopolypDataset(
    bbsize=BBSIZE,
    mode=MODE,
    prompt_finder=PROMPT_FINDER,
    n_pos=N_POS,
    n_neg=N_NEG,
    plot_prompts=False,
    prompt_gen=BOB(model=PROMPT_GEN_MODEL),
    split="test",
)

segmenter = SAM2ImageSegmenter(
    checkpoint=CHECKPOINT,
    config=CONFIG,
    fine_tuned_weights=None
)

#from segFM.predictors.sam_med2d import SAMMed2DImageSegmenter
#segmenter = SAMMed2DImageSegmenter(dataset=dataset)

df = segmenter.evaluate_bob(dataset=dataset, n_images=200, model_name=f"BOB {PROMPT_GEN_MODEL}")

utils.save_dataframe_to_csv(__file__, df, "results/bob_eval/neopolyp_bob.csv")


