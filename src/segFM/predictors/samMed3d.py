from time import time
import medim
import numpy as np
import os.path as osp
import pandas as pd
import torch
import copy
import time

from segFM import utils
from segFM.logger import logger
import segFM.predictors.SAMMed3D.utils.infer_utils as infer_utils


class SamMed3dPredictor:
    """
    A class to handle the SAM-Med3D model for 3D segmentation tasks.
    This model is different compared to MedSAM2 in the sense that it is a true 3D segmentation foundation model.
    This also means it can *only* process 3D images, not videos.
    It initializes the model and provides methods to segment images.
    """

    def __init__(self):
        ckpt_path = (
            "https://huggingface.co/blueyo0/SAM-Med3D/blob/main/sam_med3d_turbo.pth"
        )
        self.seed = 11
        self.model = medim.create_model(
            "SAM-Med3D", pretrained=True, checkpoint_path=ckpt_path
        )

    def segment_image(self, dataset, data):
        img_path = dataset.imgs_path
        gt_path = dataset.labels_path
        img_path = osp.join(img_path, data["name"])
        gt_path = osp.join(gt_path, data["gt_name"])

        gt = data["gt"]

        pred_mask = self.validate_paired_img_gt(
            self.model, img_path, gt_path, num_clicks=1
        )

        colors = np.unique(gt)
        colors = colors[colors != 0]

        # Create an empty array to store the segmentation results
        metrics = pd.DataFrame(columns=["Image", "Object", "DSC", "IoU", "NSD"])

        for color in colors:
            obj = (gt == color).astype(np.uint8) * 255
            pred = (pred_mask == color).astype(np.uint8) * 255

            # Compute metrics
            dsc, iou, nsd = utils.compute_metrics(pred, obj)

            metrics = pd.concat([
                    metrics,
                    pd.DataFrame([{
                        "Image": data["id"],
                        "Object": dataset.color_to_label[color],
                        "DSC": dsc,
                        "IoU": iou,
                        "NSD": nsd,
                    }])
            ], ignore_index=True)

        return metrics, pred_mask, gt

    def validate_paired_img_gt(self, model, img_path, gt_path, num_clicks=1):
        # Set seed within the function
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        exist_categories, final_pred_numpy_original_grid = (
            infer_utils.get_category_list_and_zero_mask(gt_path)
        )
        subject, meta_info = infer_utils.get_subject_and_meta_info(img_path, gt_path)

        for category_index in exist_categories:
            logger.info(
                f"Processing category {category_index} with {num_clicks} clicks."
            )
            category_specific_subject = copy.deepcopy(subject)
            category_specific_meta_info = copy.deepcopy(meta_info)
            # roi_image is (1,1,D,H,W), roi_label is (1,1,D,H,W)
            # meta_info contains all necessary affines and shapes
            roi_image, roi_label, meta_info = infer_utils.data_preprocess(
                category_specific_subject,
                category_specific_meta_info,
                category_index=category_index,
                target_spacing=(1.5, 1.5, 1.5),
                crop_size=128,
                # points=gen_prompt,
            )

            roi_pred_numpy, _ = infer_utils.sam_model_infer(
                model,
                roi_image,
                roi_gt=roi_label,
                prompt_generator=infer_utils.random_sample_next_click,
                num_clicks=num_clicks,
                prev_low_res_mask=None,
            )

            cls_pred_original_grid = infer_utils.data_postprocess(
                roi_pred_numpy, meta_info
            )
            final_pred_numpy_original_grid[cls_pred_original_grid == 1] = category_index

        return final_pred_numpy_original_grid

    def get_point_prompt_from_bbox(self, prompt):
        """
        Convert a bounding box to 3d point prompt.
        The point prompt is the center of the bounding box in 3D space.
        Args:
            prompt: A prompt object containing the bounding box.
        Returns:
            A tuple of torch tensors representing coords and label.
        """
        box = prompt.box  # 2d bounding box in the format [x1, y1, x2, y2]
        z = prompt.z  # z coordinate for the 2d bounding box
        coords = torch.tensor([[[
                        z,  # z coordinate
                        (box[1] + box[3]) / 2,  # y coordinate
                        (box[0] + box[2]) / 2,  # x coordinate
                    ]]],dtype=torch.float32,)

        label = torch.tensor([[1]], dtype=torch.long)
        return coords, label

    def evaluate_model(self, dataset, n_images=1):
        """
        Evaluate the model on a dataset.
        Args:
            dataset: The dataset to evaluate on.
            n_images: The number of images to evaluate.
        Returns:
            A DataFrame with the evaluation results.
        """

        images = dataset.get_n_nonduplicate_images(n_images)

        # Create the final result dataframe
        columns = ["Image", "Object", "DSC", "IoU", "NSD", "Time"]
        df = pd.DataFrame(columns=columns)

        for data in images:
            logger.info(f"Processing image {data['name']} with ID {data['id']}")

            try:
                start_time = time.time()
                metrics, seg, gt = self.segment_image(dataset, data)
                end_time = time.time()
            except Exception as e:
                logger.error(f"Error processing image {data['name']}: {e}")
                continue

            elapsed_time = end_time - start_time
            metrics["Time"] = elapsed_time

            # Append the row to the dataframe
            df = pd.concat([df, metrics], ignore_index=True)


        # Add the model name to the dataframe
        df.insert(0, "Model", "SAM-Med3D")

        # Add the prompt type to the dataframe
        df.insert(1, "Prompt", "random point")  # SAM-Med3D

        return df
