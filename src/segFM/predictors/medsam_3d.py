import pandas as pd
import numpy as np
import torch
from collections import defaultdict
import time
from sam2.build_sam import build_sam2_video_predictor_npz

from segFM import checkpoints, utils
from segFM.logger import logger

torch.set_float32_matmul_precision("high")

class MedSAM2Predictor3D:
    """
    A class to handle the MedSAM2 model for 3D segmentation tasks.
    It initializes the model and provides methods to segment images.
    """

    def __init__(self, checkpoint=checkpoints.MedSAM2CT):
        self.checkpoint = checkpoint
        self.model_cfg = checkpoints.MedSAM_cfg

        self.predictor = build_sam2_video_predictor_npz(checkpoints.MedSAM_cfg, checkpoint)

    def predict(self, image, prompts):
        """
        Segment the image using the MedSAM2 model.
        Args:
            image (np.ndarray): The input image in (z, y, x, rgb) format.
            prompts (list): List of prompts to use for segmentation.
        Returns:
            tuple: A tuple containing the segmented result arrays and the object names, identified by object IDs.
        """

        # Image is in (z, y, x, rgb)
        depth, height, width = image.shape[0], image.shape[1], image.shape[2]

        image = utils.resize_and_normalize(image)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            inference_state = self.predictor.init_state(image, height, width)

            # Cluster prompts by object ID -> Returns 2D list of prompts
            # Each sublist contains prompts for a specific object
            # Example: [[prompt1, prompt2], [prompt3], ...]
            prompts_by_obj = defaultdict(list)
            for prompt in prompts:
                prompts_by_obj[prompt.obj_id].append(prompt)

            # Create n_obj result arrays, one for each object
            result_arrays = {obj_id: np.zeros((depth, height, width), dtype=np.uint8) for obj_id in prompts_by_obj.keys()}
            result_names = {obj_id: f"{prs[0].class_label} {obj_id}" for obj_id, prs in prompts_by_obj.items()}

            for obj_id, prompts in prompts_by_obj.items():
                # Reset the result array for each prompt
                color = prompts[0].color if prompts[0].color else 1  # Default color if not specified
                label = prompts[0].class_label

                self.predictor.reset_state(inference_state)

                for prompt in prompts:
                    logger.info(
                        f"Adding prompt for {label} at frame {prompt.z} with color {prompt.color}"
                    )
                    # Add the prompts to the predictor
                    _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=prompt.z,
                        obj_id=prompt.obj_id,
                        box=prompt.box,
                    )

                # Propagate forward in z
                for (
                    out_frame_idx,
                    out_obj_ids,
                    out_mask_logits,
                ) in self.predictor.propagate_in_video(inference_state):
                    # Set the result array to 1 where the mask logits are greater than 0
                    result_arrays[obj_id][out_frame_idx, (out_mask_logits[0] > 0.0).cpu().numpy()[0]] = obj_id

                # Reset the predictor state
                self.predictor.reset_state(inference_state)

                # Add the prompts again for backward propagation
                for prompt in prompts:
                    _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=prompt.z,
                        obj_id=prompt.obj_id,
                        box=prompt.box,
                    )

                # Propagate backward in z
                for (
                    out_frame_idx,
                    out_obj_ids,
                    out_mask_logits,
                ) in self.predictor.propagate_in_video(inference_state, reverse=True):
                    # Set the result array to 1 where the mask logits are greater than 0
                    result_arrays[obj_id][out_frame_idx, (out_mask_logits[0] > 0.0).cpu().numpy()[0]] = obj_id

                self.predictor.reset_state(inference_state)

        logger.info("Segmentation completed.")

        return result_arrays, result_names

    def analyze_image_segmentation(self, data) -> pd.DataFrame:
        """
        Segment the image using the MedSAM2 model and calculate metrics.

        Args:
            data (dict): A dictionary containing the image, ground truth, and prompts.
        Returns:
            pd.DataFrame: A DataFrame containing the segmentation metrics for each object.
        """
        # Load the image
        image, gt, prompts = data["img"], data["gt"], data["prompts"]

        # Image is in (z, y, x, rgb)
        depth, height, width = image.shape

        # Create an empty array to store the segmentation results
        metrics = pd.DataFrame(columns=["Image", "Object", "DSC", "IoU", "NSD"])

        image = utils.resize_and_normalize(image)

        start_time = time.time()

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            inference_state = self.predictor.init_state(image, height, width)

            # Cluster prompts by object ID -> Returns 2D list of prompts
            # Each sublist contains prompts for a specific object
            # Example: [[prompt1, prompt2], [prompt3], ...]
            prompts_by_obj = defaultdict(list)
            for prompt in prompts:
                prompts_by_obj[prompt.obj_id].append(prompt)

            for obj_id, prompts in prompts_by_obj.items():
                # Reset the result array for each prompt
                result = np.zeros((depth, height, width), dtype=np.uint8)
                color = prompts[0].color #All prompts have the same color
                label = prompts[0].class_label

                self.predictor.reset_state(inference_state)

                for prompt in prompts:
                    logger.info(
                        f"Adding prompt for {label} at frame {prompt.z} with color {prompt.color}"
                    )
                    # Add the prompts to the predictor
                    _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=prompt.z,
                        obj_id=prompt.obj_id,
                        box=prompt.box,
                    )

                # Propagate forward in z
                for (
                    out_frame_idx,
                    out_obj_ids,
                    out_mask_logits,
                ) in self.predictor.propagate_in_video(inference_state):
                    # Set the result array to 1 where the mask logits are greater than 0
                    result[out_frame_idx, (out_mask_logits[0] > 0.0).cpu().numpy()[0]] = 1

                # Reset the predictor state
                self.predictor.reset_state(inference_state)

                # Add the prompts again for backward propagation
                for prompt in prompts:
                    _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=prompt.z,
                        obj_id=prompt.obj_id,
                        box=prompt.box,
                    )

                # Propagate backward in z
                for (
                    out_frame_idx,
                    out_obj_ids,
                    out_mask_logits,
                ) in self.predictor.propagate_in_video(inference_state, reverse=True):
                    # Set the result array to 1 where the mask logits are greater than 0
                    result[out_frame_idx, (out_mask_logits[0] > 0.0).cpu().numpy()[0]] = 1
                    
                self.predictor.reset_state(inference_state)

                # Calculate the metrics for the current object
                gt_mask = (gt == color).astype(np.uint8)


                dsc, iou, nsd = utils.compute_metrics(result, gt_mask)
                metrics = pd.concat([
                    metrics,
                    pd.DataFrame([{
                        "Image": data["id"],
                        "Object": label,
                        "DSC": dsc,
                        "IoU": iou,
                        "NSD": nsd,
                    }])
                ], ignore_index=True)

        end_time = time.time()

        # Add "Time" column to metrics
        metrics["Time"] = round(end_time - start_time, 2)

        return metrics
    
    def analyze_image_segmentation_multiobject(self, data) -> pd.DataFrame:
        """
        Segment the image using the MedSAM2 model and calculate the metrics for the whole image after all objects have been segmented.
        
        Args:
            data (dict): A dictionary containing the image, ground truth, and prompts.
        Returns:
            pd.DataFrame: A DataFrame containing the segmentation metrics for the whole image.
        """

        # Load the image
        image, gt, prompts = data["img"], data["gt"], data["prompts"]

        # Image is in (z, y, x, rgb)
        depth, height, width = image.shape

        # Create an empty array to store the segmentation results
        metrics = pd.DataFrame(columns=["Image", "Object", "DSC", "IoU", "NSD"])

        image = utils.resize_and_normalize(image)

        start_time = time.time()

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            inference_state = self.predictor.init_state(image, height, width)

            # Cluster prompts by object ID -> Returns 2D list of prompts
            # Each sublist contains prompts for a specific object
            # Example: [[prompt1, prompt2], [prompt3], ...]
            prompts_by_obj = defaultdict(list)
            for prompt in prompts:
                prompts_by_obj[prompt.obj_id].append(prompt)

            # Initialize the result array BEFORE the loop
            result = np.zeros((depth, height, width), dtype=np.uint8)
            
            for obj_id, prompts in prompts_by_obj.items():
                # Reset the result array for each prompt
                color = prompts[0].color #All prompts have the same color
                label = prompts[0].class_label

                self.predictor.reset_state(inference_state)

                for prompt in prompts:
                    logger.info(
                        f"Adding prompt for {label} at frame {prompt.z} with color {prompt.color}"
                    )
                    # Add the prompts to the predictor
                    _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=prompt.z,
                        obj_id=prompt.obj_id,
                        box=prompt.box,
                    )

                # Propagate forward in z
                for (
                    out_frame_idx,
                    out_obj_ids,
                    out_mask_logits,
                ) in self.predictor.propagate_in_video(inference_state):
                    # Set the result array to 1 where the mask logits are greater than 0
                    result[out_frame_idx, (out_mask_logits[0] > 0.0).cpu().numpy()[0]] = prompt.color

                # Reset the predictor state
                self.predictor.reset_state(inference_state)

                # Add the prompts again for backward propagation
                for prompt in prompts:
                    _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=prompt.z,
                        obj_id=prompt.obj_id,
                        box=prompt.box,
                    )

                # Propagate backward in z
                for (
                    out_frame_idx,
                    out_obj_ids,
                    out_mask_logits,
                ) in self.predictor.propagate_in_video(inference_state, reverse=True):
                    # Set the result array to the prompt color where the mask logits are greater than 0
                    result[out_frame_idx, (out_mask_logits[0] > 0.0).cpu().numpy()[0]] = prompt.color

                self.predictor.reset_state(inference_state)

            # At this point, the result array contains the segmentation for all objects
            # Now, calculate metrics for each color present in the ground truth
            for color in np.unique(gt)[1:]:  # Skip background (0)
                # Find the label corresponding to the color
                label = None
                for prompts in prompts_by_obj.values():
                    if prompts[0].color == color:
                        label = prompts[0].class_label
                        break
                gt_mask = (gt == color).astype(np.uint8)
                pred_mask = (result == color).astype(np.uint8)
                
                dsc, iou, nsd = utils.compute_metrics(pred_mask, gt_mask)

                metrics = pd.concat([
                    metrics,
                    pd.DataFrame([{
                        "Image": data["id"],
                        "Object": label,
                        "DSC": dsc,
                        "IoU": iou,
                        "NSD": nsd,
                    }])
                ], ignore_index=True)  

        end_time = time.time()

        # Add "Time" column to metrics
        metrics["Time"] = round(end_time - start_time, 2)

        return metrics

    
    def evaluate_model(self, dataset, n_images=100, concat_colors=False) -> pd.DataFrame:
        """
        Evaluate the model on a dataset.
        Args:
            n_images (int): The number of images to evaluate.
            dataset (Dataset): The dataset to evaluate on.
            concat_colors (bool): Whether to concatenate results for all colors into a single mask. ONLY use this if you know what you are doing and in special cases (e.g. for ToothFairy3 + BOB)
        Returns:
            pd.DataFrame: A DataFrame containing the segmentation metrics for each image and object.
        """
        images = dataset.get_n_nonduplicate_images(n_images)

        # Createthe final result dataframe
        columns = ["Image", "Object", "DSC", "IoU", "NSD", "Time"]
        df = pd.DataFrame(columns=columns)

        # Iterate over the images and segment them
        for i, data in enumerate(images):
            if i % 10 == 0:
                logger.info(f"Processing image {i + 1}/{n_images}...")

            # Segment the image
            if concat_colors:
                metrics = self.analyze_image_segmentation_multiobject(data)
            else:
                metrics = self.analyze_image_segmentation(data)

            print(metrics)

            # Append the row to the dataframe
            df = pd.concat([df, metrics], ignore_index=True)

        # Add the model name to the dataframe
        df.insert(0, "Model", str(self.checkpoint).split("/")[-1])
        df.insert(1, "Mode", dataset.mode)

        logger.info(
            f" DSC - Average: {np.mean(df['DSC']):.4f}, Median: {np.median(df['DSC']):.4f}"
        )
        logger.info(
            f" IOU - Average: {np.mean(df['IoU']):.4f}, Median: {np.median(df['IoU']):.4f}"
        )
        logger.info(
            f" NSD - Average: {np.mean(df['NSD']):.4f}, Median: {np.median(df['NSD']):.4f}"
        )

        return df

