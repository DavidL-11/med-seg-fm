from transformers import Sam3Processor, Sam3Model, Sam3TrackerProcessor, Sam3TrackerModel

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


class SAM3ImageSegmenter(BaseSegmenter):
    def __init__(
        self,
        mode: str = "visual",
    ):
        self.mode = mode

        # Setup device depending on hardware (CPU, CUDA, MPS)
        self.device = utils.setup_device()

        # load the model
        if mode in ["text", "box"]:
            self.sam3 = Sam3Model.from_pretrained("facebook/sam3").to(self.device)
            self.processor = Sam3Processor.from_pretrained("facebook/sam3")
        elif mode == "point":
            self.sam3 = Sam3TrackerModel.from_pretrained("facebook/sam3").to(self.device)
            self.processor = Sam3TrackerProcessor.from_pretrained("facebook/sam3")


    def segment_image_multiprompt(self, image, prompts: list[Prompt]):
        """
        Predicts the segmentation mask for a given image and a list of prompts.
        """
        segmasks = []
        scores = []
        prompts_new = []

        for prompt in prompts:
            if self.mode == "text":
                inputs = self.processor(images=image, text=prompt.class_label, return_tensors="pt").to(self.device)

                print("Adding text prompt:", prompt.class_label)

                with torch.no_grad():
                    outputs = self.sam3(**inputs)

                results = self.processor.post_process_instance_segmentation(
                    outputs,
                    threshold=0.5,
                    mask_threshold=0.5,
                    # Make target sizes H x W, 2 dimensional, no rgb
                    target_sizes=[(size[0], size[1]) for size in inputs.get("original_sizes").tolist()]
                )[0]

                # Get the masks, bounding boxes, and scores
                mask, box, score = results["masks"], results["boxes"], results["scores"]
            elif self.mode == "box":
                mask, box, score = self.predict_visual(image, prompt)
            elif self.mode == "point":
                mask, box, score = self.predict_point(image, prompt)


            for i in range(len(mask)):
                # Get the segmentation mask from the predicted masks
                #print(f"Mask shape: {mask[i].cpu().numpy().shape}")
                segmask = utils.create_binary_segmentation(mask[i].cpu().numpy())
                segmasks.append(segmask)
                scores.append(score)
                prompts_new.append(prompt)

        return segmasks, scores, prompts_new
    
    def predict_visual(self, image, prompt: Prompt):
        """
        Predicts the segmentation mask for a given visual prompt.
        """
        if prompt.is_empty():
            return None, None, None

        inputs = self.processor(
            images=image,
            input_boxes=[[prompt.box]],
            input_boxes_labels=[[1]],
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.sam3(**inputs)

        # Post-process results
        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=0.5,
            mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist()
        )[0]

        mask, box, score = results["masks"], results["boxes"], results["scores"]
        return mask, box, score
    
    def predict_point(self, image, prompt: Prompt):
        """
        Predicts the segmentation mask for a given visual prompt.
        """
        if prompt.is_empty():
            return None, None, None

        inputs = self.processor(
            images=image,
            input_points=[[prompt.point]],
            input_labels=[[prompt.label]],
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.sam3(**inputs)

        #print(inputs.get("original_sizes"))

        # Post-process results
        # Output: (batch_size, num_channels, height, width)
        masks = self.processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]

        # Convert to binary masks with (h, w) shape
        # Only the first channel includes the relevant mask
        masks = masks[:, 0, :, :]

        return masks, None, None

    def get_model_name(self):
        """
        Generates the model name based on the checkpoint and fine-tuning status.

        Returns:
            str: The name of the model and if it is fine-tuned.
        """
        # The checkpoint name without fine-tuning
        # e.g. "sam2.1_hiera_small.pt" -> "S", large -> "L" etc.
        return "SAM3"