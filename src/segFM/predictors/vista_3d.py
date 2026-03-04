import numpy as np
import copy, os
import torch
import pandas as pd
import time

from segFM.predictors.VISTA.vista3d.scripts.infer import InferClass
from segFM import utils, color_classes
from segFM.logger import logger

class VISTA3DPredictor:
    """
    A class to handle the VISTA3D model for 3D segmentation tasks.
    It initializes the model and provides methods to segment images.

    Args:
        config_file (list): List of configuration files for the VISTA3D model.
        mode (str): The mode of operation. Options are "point", "label", and "zero-shot".
                     - "point": Use point prompts for segmentation.
                     - "label": Use label prompts for segmentation.
                     - "zero-shot": Use a point prompt with an unsupported class for zero-shot segmentation.
    """

    def __init__(self, config_file=["./vista3d.yaml"], mode="point"):
        self.inferer = InferClass(config_file=config_file)
        self.mode = mode

    def segment_image(self, data: dict, dataset) -> tuple[dict, np.ndarray, np.ndarray]:
        """
        Segment the image using the VISTA3D model.
        Args:
            data (dict): A dictionary containing the image, ground truth, and prompts.
            label_prompt (bool): If True, don't use a generated prompt, but a label prompt (e.g. 1 for liver).
        """
        # Load the image
        gt, prompts = data["gt"], data["prompts"]

        # Create an empty array to store the segmentation results
        metrics = pd.DataFrame(columns=["Image", "Object", "DSC", "IoU", "NSD", "Time"])

        img_path = os.path.join(dataset.imgs_path, data["name"])

        self.inferer.clear_cache()

        start_time = time.time()
        
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            # Print device
            print(f"Using device: {self.inferer.device}")
            for i, prompt in enumerate(prompts):
                self.inferer.clear_cache()  # Clear cache for each prompt
                logger.info(f"Processing prompt {i + 1}/{len(prompts)}: {prompt.class_label}")
                
                label = prompt.class_label # Class, e.g. "liver"
                color = prompt.color # Color in GT, e.g. 1

                # Variables needed for VISTA3D inference
                points = None # wants (y, x, z) format
                point_labels = None # 0 for positive, 1 for negative
                label_prompt = None # Label prompt to use, e.g. 1 for liver (not always the same as color!)
                prompt_class = None # Prompt class (the same as label_prompt, idk why it is needed)

                if self.mode=="label" :
                    # Use a label prompt (e.g. 1 for liver)
                    vista_label = color_classes.label_to_vista_id[label]
                    label_prompt = np.array([vista_label])[np.newaxis, ...]
                    prompt_class = copy.deepcopy(label_prompt)
                elif self.mode=="point" or self.mode=="zero-shot":
                    # Use a generated prompt
                    bbox = prompt.box
                    z = prompt.z
                    point_labels = [1]

                    # Convert bbox to a point prompt expected by VISTA by taking the center of the bbox
                    x1, y1, x2, y2 = bbox
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    bbox = np.array([[center_y, center_x , z]])

                    # Check if the point is 1
                    gt2 = (gt == color).astype(np.uint8)
                    point_val = gt2[z, center_y, center_x]  # Get the value at the point
                    if point_val == 0:
                        print(f"Warning: Point at ({center_x}, {center_y}, {z}) is not in the ground truth for {label}. Using a point prompt instead.")
                        # Use a random prompt instead

                        gt2_slice = gt2[z, :, :]  # Get the ground truth mask for the current slice

                        # Select a random nonzero point in the ground truth mask and this are the coordinates
                        nonzero_points = np.argwhere(gt2_slice > 0)
                    
                        random_point = nonzero_points[np.random.choice(nonzero_points.shape[0])]
                        y, x = random_point[0], random_point[1]

                        points = [[x, y, z]]
                    else:
                        points = [[center_x, center_y, z]]

                    points = np.array(points)[np.newaxis, ...]  # Add batch dimension
                    point_labels = np.array(point_labels)[np.newaxis, ...]  # Add batch dimension

                    if self.mode=="zero-shot":
                        label_prompt = np.array([135])[np.newaxis, ...]  # Use an unsupported class_id for zero-shot
                        prompt_class = copy.deepcopy(label_prompt)
                else:
                    raise ValueError(f"Unknown mode: {self.mode}. Supported modes are 'point', 'label', and 'zero-shot'.")
                
                mask = self.inferer.infer(
                    image_file={"image": img_path},
                    point=points,
                    point_label=point_labels,
                    label_prompt=label_prompt,
                    prompt_class=prompt_class,
                    save_mask=True,
                )[0]

                # Convert the mask to a binary mask
                mask = (mask.data.cpu().numpy() > 0.5).astype(np.uint8) * 255
                mask = np.transpose(mask, (2, 1, 0))  # Transpose to (z, y, x) for visualization

                gt_mask = (gt == prompt.color).astype(np.uint8) * 255

                # Compute metrics
                dsc, iou, nsd = utils.compute_metrics(mask, gt_mask)

                metrics = pd.concat([
                    metrics,
                    pd.DataFrame([{
                        "Image": data["id"],
                        "Object": prompt.class_label,
                        "DSC": dsc,
                        "IoU": iou,
                        "NSD": nsd,
                    }])
                ], ignore_index=True)

                print(f"Metrics for {prompt.class_label}: DSC={dsc}, IoU={iou}, NSD={nsd}")

        end_time = time.time()
        metrics["Time"] = round(end_time - start_time, 2)

        return metrics
    
    def evaluate_model(self, dataset, n_images=100) -> pd.DataFrame:
        """
        Evaluate the VISTA3D model on a dataset.
        Args:
            dataset: The dataset to evaluate on.
            n_images (int): The number of images to evaluate.
        Returns:
            df (pd.DataFrame): A DataFrame containing the evaluation metrics.
        """
        images = dataset.get_n_nonduplicate_images(n_images)

        # Each image has n_object rows with DSC, IoU, and NSD metrics in the columns
        columns = ["Image", "Object", "DSC", "IoU", "NSD"]
        df = pd.DataFrame(columns=columns)

        # Iterate over the images and segment them
        for i, data in enumerate(images):
            logger.info(f"Processing image {i + 1}/{n_images}... {data['img'].shape}")

            # Segment the image
            metrics = self.segment_image(data, dataset)

            print(metrics)

            # Create a row for the dataframe
            df = pd.concat([df, metrics], ignore_index=True)

        # Add the model name to the dataframe
        df.insert(0, "Model", "VISTA3D")

        # Add the prompt type to the dataframe
        df.insert(1, "Mode", self.mode)

        logger.info(f"Successfully evaluated {len(df)} images of the {str(dataset)} dataset.")

        return df


