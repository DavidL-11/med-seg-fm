from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2
import numpy as np
import torch
import cv2
import time
import pandas as pd

from segFM.prompts import Prompt
from segFM.logger import logger
from segFM import utils, checkpoints
from segFM.predictors.BaseSegmenter import BaseSegmenter

CHECKPOINT = checkpoints.SAM2_tiny
CONFIG = checkpoints.SAM2_tiny_cfg


class SAM2ImageSegmenter(BaseSegmenter):
    def __init__(
        self, checkpoint, config, fine_tuned_weights=None, postprocessing=True
    ):
        self.checkpoint = checkpoint
        self.model_config = config
        self.fine_tuned_weights = fine_tuned_weights

        # Setup device depending on hardware (CPU, CUDA, MPS)
        self.device = utils.setup_device()

        # load the model
        self.sam2 = build_sam2(
            self.model_config,
            self.checkpoint,
            device=self.device,
            apply_postprocessing=postprocessing,
        )
        self.predictor = SAM2ImagePredictor(self.sam2)

        # Load the fine-tuned model weights if specified
        if fine_tuned_weights is not None:
            logger.info(
                f"Loading fine-tuned model weights from {fine_tuned_weights}..."
            )
            self.predictor.model.load_state_dict(
                torch.load(fine_tuned_weights, map_location=self.device)
            )

    def segment_image_multiprompt(self, image, prompts: list[Prompt]):
        """
        Predicts the segmentation mask for a given image and a list of prompts.
        """
        segmasks = []
        scores = []
        prompts_new = []

        # Perform inference
        with torch.no_grad():
            self.predictor.set_image(image)

            for prompt in prompts:
                if prompt.is_empty():
                    continue
                if prompt.point is not None and prompt.point.ndim == 3:
                    object_amount = len(prompt.point)
                elif prompt.box is not None and prompt.box.ndim == 2:
                    object_amount = len(prompt.box)
                else:
                    object_amount = 1

                logger.debug(f"Predicting {object_amount} objects...")

                mask, score, logits = self.predictor.predict(
                    point_coords=prompt.point,
                    point_labels=prompt.label,
                    box=prompt.box,
                    multimask_output=False,
                )

                for i in range(object_amount):
                    # Get the segmentation mask from the predicted masks
                    segmask = utils.create_binary_segmentation(mask[i])
                    segmasks.append(segmask)
                    scores.append(score)
                    prompts_new.append(prompt)

        return segmasks, scores, prompts_new

    def get_model_name(self):
        """
        Generates the model name based on the checkpoint and fine-tuning status.

        Returns:
            str: The name of the model and if it is fine-tuned.
        """
        # The checkpoint name without fine-tuning
        # e.g. "sam2.1_hiera_small.pt" -> "S", large -> "L" etc.
        if "hiera" in self.checkpoint:
            # Remove the last .pt from the checkpoint name and capitalize the first letter
            checkpoint_type = self.checkpoint.split("_")[-1].replace(".pt", "").capitalize()
            if "Plus" in checkpoint_type:
                checkpoint_type = "Base+"
            checkpoint_short = "SAM2.1 " + checkpoint_type
        elif "MedSAM" in self.checkpoint:
            checkpoint_short = self.checkpoint.split("/")[-1].replace(".pt", "")
        else:
            checkpoint_short = self.checkpoint.split("_")[-1].split(".")[0].capitalize()

        if self.fine_tuned_weights is not None:
            # Show the number of images used for fine-tuning, e.g. "bagls_tuned_sam2s_100.torch" -> 100
            n_FT = self.fine_tuned_weights.split("_")[-1].split(".")[0]

            # Return the checkpoint name and the number of images used for fine-tuning
            # e.g. "sam2.1_hiera_small.pt" -> "S_FT_100"
            return f"{checkpoint_short}_FT{n_FT}"
        else:
            return checkpoint_short