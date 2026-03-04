import pandas as pd
import numpy as np
import torch
import os
from collections import defaultdict
import time
from huggingface_hub import snapshot_download  # Install huggingface_hub if not already installed
from nnInteractive.inference.inference_session import nnInteractiveInferenceSession

from segFM.logger import logger
from segFM import utils

torch.set_float32_matmul_precision("high")

class nnInteractivePredictor3D:
    """
    A class to handle the nnInteractive model for 3D segmentation tasks.
    It initializes the model and provides methods to segment images.
    """

    def __init__(self, model_name="nnInteractive_v1.0"):
        REPO_ID = "nnInteractive/nnInteractive"
        DOWNLOAD_DIR = "."  # Specify the download directory
        self.model_name = model_name
        
        download_path = snapshot_download(
            repo_id=REPO_ID,
            allow_patterns=[f"{self.model_name}/*"],
            local_dir=DOWNLOAD_DIR
        )
        
        self.session = nnInteractiveInferenceSession(
            device=torch.device("cuda:0"),  # Set inference device
            use_torch_compile=False,  # Experimental: Not tested yet
            verbose=False,
            torch_n_threads=os.cpu_count(),  # Use available CPU cores
            do_autozoom=True,  # Enables AutoZoom for better patching
            use_pinned_memory=True,  # Optimizes GPU memory transfers
        )
        
        model_path = os.path.join(DOWNLOAD_DIR, self.model_name)
        self.session.initialize_from_trained_model_folder(model_path)
        

    def predict(self, image, prompts):
        """
        Segment the image using the MedSAM2 model.
        Args:
            image (np.ndarray): The input image in (z, y, x) format.
            prompts (list): List of prompts to use for segmentation.
        Returns:
            tuple: A tuple containing the segmented result arrays and the object names, identified by object IDs.
        """
        if image.ndim != 3:
            raise ValueError(f"nnInteractive3D only supports 3D grayscale images! Found image with shape {image.shape}")

        depth, height, width = image.shape[0], image.shape[1], image.shape[2]
        image = np.transpose(image, (2, 1, 0))[None, ...]
        
        # Set the image and prepare the target buffer (-> result array)
        self.session.set_image(image)
        target_tensor = torch.zeros(image.shape[1:], dtype=torch.uint8)  # Must be 3D (x, y, z)
        self.session.set_target_buffer(target_tensor)
        
        # Cluster prompts by object ID -> Returns 2D list of prompts
        # Each sublist contains prompts for a specific object
        # Example: [[prompt1, prompt2], [prompt3], ...]
        prompts_by_obj = defaultdict(list)
        for prompt in prompts:
            prompts_by_obj[prompt.obj_id].append(prompt)

        # Create n_obj result arrays, one for each object
        result_arrays = {obj_id: np.zeros((depth, height, width), dtype=np.uint8) for obj_id in prompts_by_obj.keys()}
        result_names = {obj_id: f"{prs[0].class_label}" for obj_id, prs in prompts_by_obj.items()}

        for obj_id, prompts in prompts_by_obj.items():
            # Reset the result array for each prompt
            label = prompts[0].class_label

            for prompt in prompts:
                color = obj_id if prompt.color is None else prompt.color
                box = prompt.box # (4, ) in xyxy format
                z = prompt.z # Scalar value indicating the slice index
                # ATTENTION: nnInteractive has different bbox format than SAM
                # BBOX_COORDINATES must be specified as [[x1, x2], [y1, y2], [z1, z2]] (half-open intervals).
                
                box = np.array([
                    [box[0], box[2]],  # x1, x2
                    [box[1], box[3]],  # y1, y2
                    [z, z + 1]         # z1, z2 (half-open interval)
                ])
                
                # Add the prompts to the predictor
                self.session.add_bbox_interaction(
                    bbox_coords=box,
                    include_interaction=True,
                )
                

            results = target_tensor.clone().numpy().transpose(2, 1, 0)
            result_arrays[obj_id][results > 0] = color
            self.session.reset_interactions()

        logger.info("Segmentation completed.")

        return result_arrays, result_names
    
    
    def analyze_image_segmentation(self, data) -> pd.DataFrame:
        """
        Segment the image using the nnInteractive model and calculate metrics.

        Args:
            data (dict): A dictionary containing the image, ground truth, and prompts.
        Returns:
            pd.DataFrame: A DataFrame containing the segmentation metrics for each object.
        """
        # Load the image
        image, gt, prompts = data["img"], data["gt"], data["prompts"]

        if image.ndim != 3:
            raise ValueError(f"nnInteractive3D only supports 3D grayscale images! Found image with shape {image.shape}")

        depth, height, width = image.shape[0], image.shape[1], image.shape[2]
        image = np.transpose(image, (2, 1, 0))[None, ...]
        
        # Set the image and prepare the target buffer (-> result array)
        self.session.set_image(image)
        target_tensor = torch.zeros(image.shape[1:], dtype=torch.uint8)  # Must be 3D (x, y, z)
        self.session.set_target_buffer(target_tensor)

        # Create an empty array to store the segmentation results
        metrics = pd.DataFrame(columns=["Image", "Object", "DSC", "IoU", "NSD"])
        
        # Visualize prompts in napari
        # import napari
        # viewer = napari.Viewer()
        # viewer.add_image(image[0].transpose(2, 1, 0), name="Image", colormap="gray")
        # viewer.add_labels(gt, name="Ground Truth")

        start_time = time.time()

        # Cluster prompts by object ID -> Returns 2D list of prompts
        # Each sublist contains prompts for a specific object
        # Example: [[prompt1, prompt2], [prompt3], ...]
        prompts_by_obj = defaultdict(list)
        for prompt in prompts:
            prompts_by_obj[prompt.obj_id].append(prompt)

        for obj_id, prompts in prompts_by_obj.items():
            label = prompts[0].class_label
            
            print(f"Adding {len(prompts)} prompts for object '{label}' (ID: {obj_id} - Color: {prompts[0].color})")
            
            for prompt in prompts:
                color = obj_id if prompt.color is None else prompt.color
                box = prompt.box # (4, ) in xyxy format
                z = prompt.z # Scalar value indicating the slice index
                # ATTENTION: nnInteractive has different bbox format than SAM
                # BBOX_COORDINATES must be specified as [[x1, x2], [y1, y2], [z1, z2]] (half-open intervals).
                
                box = np.array([
                    [box[0], box[2]],  # x1, x2
                    [box[1], box[3]],  # y1, y2
                    [z, z + 1]         # z1, z2 (half-open interval)
                ])
                
                # Add the prompts to the predictor
                self.session.add_bbox_interaction(
                    bbox_coords=box,
                    include_interaction=True,
                )

            result = target_tensor.clone().numpy().transpose(2, 1, 0)  # Transpose results from (x, y, z) back to (z, y, x)
            self.session.reset_interactions()

            # Calculate the metrics for the current object
            gt_mask = (gt == color).astype(np.uint8)
            obj_mask = result.astype(np.uint8) # Required so they have the same unique values
            
            # viewer.add_labels(result, name=f"Prediction {label}")

            dsc, iou, nsd = utils.compute_metrics(obj_mask, gt_mask)
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
        # viewer.show()
        # napari.run()

        end_time = time.time()

        # Add "Time" column to metrics
        metrics["Time"] = round(end_time - start_time, 2)

        return metrics
    
    
    def analyze_image_segmentation_multiobject(self, data) -> pd.DataFrame:
        """
        Segment the image using the nnInteractive model and calculate the metrics for the whole image after all objects have been segmented.
        
        Args:
            data (dict): A dictionary containing the image, ground truth, and prompts.
        Returns:
            pd.DataFrame: A DataFrame containing the segmentation metrics for the whole image.
        """

        # Load the image
        image, gt, prompts = data["img"], data["gt"], data["prompts"]

        if image.ndim != 3:
            raise ValueError(f"nnInteractive3D only supports 3D grayscale images! Found image with shape {image.shape}")

        depth, height, width = image.shape[0], image.shape[1], image.shape[2]
        image = np.transpose(image, (2, 1, 0))[None, ...]
        
        # Set the image and prepare the target buffer (-> result array)
        self.session.set_image(image)
        target_tensor = torch.zeros(image.shape[1:], dtype=torch.uint8)  # Must be 3D (x, y, z)
        self.session.set_target_buffer(target_tensor)

        # Create an empty array to store the segmentation results
        metrics = pd.DataFrame(columns=["Image", "Object", "DSC", "IoU", "NSD"])

        start_time = time.time()

        # Cluster prompts by object ID -> Returns 2D list of prompts
        # Each sublist contains prompts for a specific object
        # Example: [[prompt1, prompt2], [prompt3], ...]
        prompts_by_obj = defaultdict(list)
        for prompt in prompts:
            prompts_by_obj[prompt.obj_id].append(prompt)

        # Initialize the result array BEFORE the loop
        result = np.zeros((depth, height, width), dtype=np.uint8)
        
        for obj_id, prompts in prompts_by_obj.items():
            label = prompts[0].class_label
            color = prompts[0].color if prompts[0].color is not None else obj_id
            
            print(f"Adding {len(prompts)} prompts for object '{label}' (ID: {obj_id} - Color: {color})")
            
            for prompt in prompts:
                box = prompt.box # (4, ) in xyxy format
                z = prompt.z # Scalar value indicating the slice index
                # ATTENTION: nnInteractive has different bbox format than SAM
                # BBOX_COORDINATES must be specified as [[x1, x2], [y1, y2], [z1, z2]] (half-open intervals).
                
                box = np.array([
                    [box[0], box[2]],  # x1, x2
                    [box[1], box[3]],  # y1, y2
                    [z, z + 1]         # z1, z2 (half-open interval)
                ])
                
                # Add the prompts to the predictor
                self.session.add_bbox_interaction(
                    bbox_coords=box,
                    include_interaction=True,
                )

            pred_mask = target_tensor.clone().numpy().transpose(2, 1, 0)  # Transpose results from (x, y, z) back to (z, y, x)
            result[pred_mask > 0] = color
            self.session.reset_interactions()

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
        df.insert(0, "Model", "nnInteractive")
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

