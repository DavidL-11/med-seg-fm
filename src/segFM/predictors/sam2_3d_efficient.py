import matplotlib.pyplot as plt
from vedo import show, Volume
from PIL import Image
import pandas as pd
import numpy as np
import torch
import cv2
import os
from sam2.build_sam import build_sam2_video_predictor

from segFM.utils import compute_dice_coefficient
from segFM import checkpoints

torch.set_float32_matmul_precision("high")

class SAM2Predictor3D:
    """
    A class to handle the SAM2 model for 3D segmentation tasks.
    This model is different compared to MedSAM2 in the sense that it is a true 3D segmentation foundation model.
    It initializes the model and provides methods to segment images.
    """

    def __init__(self, checkpoint=checkpoints.SAM2_tiny, model_cfg=checkpoints.SAM2_tiny_cfg):
        self.checkpoint = checkpoint
        self.model_cfg = model_cfg
        self.predictor = build_sam2_video_predictor(model_cfg, checkpoint)
        np.random.seed(11)

    def resize_grayscale_to_rgb(self, array):
        """
        Resize a 3D grayscale NumPy array to an RGB image.

        Parameters:
            array (np.ndarray): Input array of shape (d, h, w).

        Returns:
            np.ndarray: Resized array of shape (d, 3, image_size, image_size).
        """
        d, h, w = array.shape
        resized_array = np.zeros((d, 3, h, w), dtype=np.uint8)

        for i in range(d):
            img_pil = Image.fromarray(array[i].astype(np.uint8))
            img_rgb = img_pil.convert("RGB")
            img_array = np.array(img_rgb).transpose(
                2, 0, 1
            )  # (3, image_size, image_size)
            resized_array[i] = img_array

        return resized_array


    def get_largest_bounding_boxes(self, mask, plot_prompt=False):
        """
        Returns as many bounding boxes as there are objects in the mask.
        Format is z, List([x1, y1, x2, y2])
        The mask is expected to be a 3D numpy array.
        The bounding box is a 2D array obtained by slicing the mask along the z-axis at a good coordinate
        and then finding the bounding box of the ground truth.
        """
        object_colors = np.unique(mask)  # Retrieve all the unique colors in the mask
        # Remove the background color (0)
        object_colors = object_colors[object_colors > 0]

        boxes = {}

        # Iterate over the unique colors and find the bounding box for each color
        for color in object_colors:
            color_mask = (mask == color).astype(np.uint8)

            # Find the slice with the most pixels
            z = np.argmax(np.sum(color_mask, axis=(1, 2)))
            # Find the bounding box of the color mask
            x1, y1, w, h = cv2.boundingRect(color_mask[z])
            # Add it to the dictionary
            boxes[color] = (z, np.array([x1, y1, x1 + w, y1 + h]))

        if plot_prompt:  # Plot the bounding box
            for color in object_colors:
                z, (x1, y1, x2, y2) = boxes[color]
                rect = cv2.rectangle(mask[z].copy(), (x1, y1), (x2, y2), (255, 0, 0), 2)
                plt.imshow(rect, cmap="gray")
                plt.show()
        return boxes


    def convert_3d_image_to_frames(self, image, folder):
        """
        Convert a 3D image to a list of 2D frames and save them in a given directory.
        Args:
            image (np.ndarray): 3D image of shape (depth, rgb_channels, height, width).
        """
        if not os.path.exists(folder):
            os.makedirs(folder)

        # Check if the image is grayscale and convert it to RGB + resize
        image = self.resize_grayscale_to_rgb(image)

        depth = image.shape[0]

        for i in range(depth):
            frame = np.transpose(
                image[i], (1, 2, 0)
            )  # Convert to (height, width, channels)
            # Convert the frame to numpy array from torch tensor and save it
            cv2.imwrite(os.path.join(folder, f"{i:03d}.jpg"), frame)


    def segment_image(self, data, dataset):
        """
        Segment the image using the MedSAM2 model.
        """
        # Load the image
        image = data["img"]
        gt = data["gt"]

        width, height, depth = image.shape[::-1]
        print("Image size:", image.shape[::-1])

        # Get the multiple bounding boxes as a dictionary
        bboxes = self.get_largest_bounding_boxes(gt, plot_prompt=False)

        # Create an empty array to store the segmentation results
        result = np.zeros(image.shape, dtype=np.uint8)

        print("Image resized shape:", image.shape)

        folder = "temp_frames"

        self.convert_3d_image_to_frames(image, folder)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            inference_state = self.predictor.init_state(video_path="temp_frames")

            for color, (z, bbox) in bboxes.items():
                # Propagate forward in z
                _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=z,
                    obj_id=color,
                    box=bbox,
                )

            # Set the result array to 1 where the mask logits are greater than 0
            for (
                out_frame_idx,
                out_obj_ids,
                out_mask_logits,
            ) in self.predictor.propagate_in_video(inference_state):
                for obj_id, mask_logits in zip(out_obj_ids, out_mask_logits):
                    # Set the result array to the color where the mask is True
                    result[out_frame_idx, (mask_logits[0] > 0.0).cpu().numpy()] = obj_id

            # Reset the predictor state
            self.predictor.reset_state(inference_state)

            for color, (z, bbox) in bboxes.items():
                # Propagate backwards in z
                _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=z,
                    obj_id=color,
                    box=bbox,
                )

            # Set the result array to 1 where the mask logits are greater than 0
            for (out_frame_idx, out_obj_ids, out_mask_logits) in self.predictor.propagate_in_video(inference_state, reverse=True):
                for obj_id, mask_logits in zip(out_obj_ids, out_mask_logits):
                    # Set the result array to the color where the mask is True
                    result[out_frame_idx, (mask_logits[0] > 0.0).cpu().numpy()] = obj_id

        dataset.cleanup_temp_dir(folder)

        dice_scores = {label: np.nan for label in dataset.color_to_label.values()}

        # Compute the Dice coefficients for each organ
        for color, (z, bbox) in bboxes.items():
            gt_mask = (gt == color).astype(np.uint8)
            pred_mask = (result == color).astype(np.uint8)
            dice = compute_dice_coefficient(gt_mask, pred_mask)
            dice_scores[dataset.color_to_label[color]] = dice

        return dice_scores, result, gt
    
    def evaluate_model(self, dataset, n_images=1, plot_results=False):
        imgs = dataset.get_n_nonduplicate_images(n_images)

        labels = dataset.color_to_label.values()
        columns = ["Model", "Image"] + list(labels)
        df = pd.DataFrame(columns=columns)

        for data in imgs:
            dice, seg, gt = self.segment_image(data, dataset)

            # Create a row for the dataframe
            row = {"Model": str(self.checkpoint).split("/")[-1], "Image": data["id"]}

            # Add the dice dict to the row dict
            for label in labels:
                row[label] = dice.get(label, np.nan)

            # Append the row to the dataframe
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

            if plot_results:
                res = Volume(seg)
                gt = Volume(gt)
                show(gt, axes=1, viewup="z", interactive=True)
                show(res, axes=1, viewup="z", interactive=True, new=True)

        return df