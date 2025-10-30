from argparse import Namespace
import numpy as np
import torch
import os

from segFM.predictors.SAM_Med2D.segment_anything import sam_model_registry
from segFM.predictors.SAM_Med2D.segment_anything.predictor_sammed import SammedPredictor
from segFM.predictors.BaseSegmenter import BaseSegmenter
from segFM.logger import logger
from segFM import utils


class SAMMed2DImageSegmenter(BaseSegmenter):
    def __init__(self):
        # Setup device depending on hardware (CPU, CUDA, MPS)
        self.device = utils.setup_device()

        args = Namespace()
        args.image_size = 256
        args.encoder_adapter = True
        args.sam_checkpoint = os.path.join(
            os.path.dirname(__file__), "SAM_Med2D", "sam-med2d_b.pth"
        )
        self.model = sam_model_registry["vit_b"](args).to(self.device)
        self.predictor = SammedPredictor(self.model)
        

    def segment_image_multiprompt(self, image, prompts):
        """
        Predicts the segmentation masks for a given image and multiple prompts.
        This has the advantage that it only needs to calculate the image embedding once,
        which is more efficient than calling segment_image for each prompt separately.
        """
        segmasks = []
        scores = []
        prompts_new = []

        # Perform inference
        with torch.no_grad():
            self.predictor.set_image(image)


            for prompt in prompts:
                point_coords = prompt.point
                point_labels = prompt.label
                box = prompt.box

                _masks, _scores, _logits = self.predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=box,
                    multimask_output=True,
                )

                logger.debug(f"Predicted {len(_masks)} masks with scores: {_scores}")

                # Select the best mask based on the scores
                max_score_index = np.argmax(_scores)

                segmasks.append(utils.create_binary_segmentation(_masks[max_score_index]))
                scores.append(_scores[max_score_index])
                prompts_new.append(prompt)

                del _masks, _scores, _logits  # Clear memory
                del max_score_index

            self.predictor.reset_image()

        return segmasks, scores, prompts_new

    # def evaluate_model(self, n_images, plot_results=False, save_dataset_name=False):
    #     """
    #     Evaluates the model on a random set of images.

    #     Parameters:
    #         n_images (int): Number of images to evaluate.
    #         mode (str): Mode of evaluation ('box', 'point', or 'both').
    #         bbsize (int): The padding size to add around the bounding box.
    #         prompt_finder (str): The method to find the prompt ('best', 'canny', 'corners', 'random', or 'histogram').
    #     Returns:
    #         tuple: The average dice score, surface distance, and IoU for the evaluated images.
    #     """
    #     # Prepare lists to store the results
    #     dsc = []
    #     surface_dice = []
    #     iou = []
    #     names = []
    #     class_labels = []

    #     # Get n images from the dataset
    #     data_list = self.dataset.get_n_nonduplicate_images(n_images)

    #     for i, data in enumerate(data_list):

    #         if i % (max(1, n_images // 10)) == 0:
    #             if i == 0:
    #                 start_time = time.time()
    #                 logger.info("[0s] Starting evaluation of images...")
    #             else:
    #                 elapsed_time = time.time() - start_time
    #                 logger.info(f"[{elapsed_time:.1f}s] Evaluating image {i} of {n_images}... - Current DSC: {np.mean(dsc):.3f}")

    #         # Extract the information from the data dictionary
    #         image = data["img"]
    #         gt = data["gt"]
    #         prompts = data["prompts"]

    #         if gt.sum() == 0:
    #             # No valid ground truth was found, skip this image / segmentation mask is empty
    #             continue
            
    #         # Store the actual objects from the ground truth mask (= the unique colors in the mask)
    #         unique_colors = list(np.unique(gt)[1:])  # Exclude background (0)

    #         # If the GT is a labelmap (3D mask with multiple channels for different objects),
    #         # still only create a single channel for each object
    #         # and use the channel index as the color
    #         shape = gt.shape if gt.ndim == 2 else (gt.shape[1], gt.shape[2])

    #         results = {
    #             color: np.zeros(shape, dtype=np.uint8) for color in unique_colors
    #         }

    #         total_segmasks, total_scores = self.segment_image_multiprompt(image, prompts)

    #         # Enumerate over all prompts generated
    #         for i, prompt in enumerate(prompts):

    #             if gt.ndim == 2: #Normal 2D ground truth mask with different colors for different objects
    #                 color = prompt.color
    #             elif gt.ndim == 3: # Labelmap ground truth mask with multiple channels for different objects
    #                 color = prompt.channel + 1  # Channel is 0-indexed, but colors are 1-indexed

    #             results[color] = np.where(total_segmasks[i] > 0.5, 1, results[color])

    #         for c in unique_colors:
    #             if gt.ndim == 2: #Normal 2D ground truth mask with different colors for different objects
    #                 object_gt = (gt == c).astype(np.uint8)
    #             elif gt.ndim == 3: # Labelmap ground truth mask with multiple channels for different objects
    #                 object_gt = (gt[c - 1, :, :] > 0).astype(np.uint8)

    #             object_pred = results[c]  # Get the predicted mask for the object

    #             # Compute the dice score
    #             _dsc, _iou, _nsd = utils.compute_metrics(object_pred, object_gt)

    #             dsc.append(_dsc)
    #             iou.append(_iou)
    #             surface_dice.append(_nsd)
    #             class_labels.append(self.dataset.color_to_label.get(c, "unknown"))
    #             names.append(data["name"])
                
    #             if plot_results:
    #                 image_plt = image.copy()
    #                 for prompt in prompts:
    #                     if prompt.color == c or (prompt.channel is not None and prompt.channel + 1 == c):
    #                         if hasattr(prompt, "point") and prompt.point is not None:
    #                             # Plot the points on the image
    #                             for point in prompt.point:
    #                                 cv2.circle(image_plt, tuple(point), 5, (255, 0, 0), -1)
    #                         if hasattr(prompt, "box") and prompt.box is not None:
    #                             # Plot the bounding box on the image
    #                             x1, y1, x2, y2 = prompt.box
    #                             cv2.rectangle(image_plt, (x1, y1), (x2, y2), (0, 255, 0), 2)

    #                 utils.plot_img_gt_pred(
    #                     img=image_plt,
    #                     gt=object_gt,
    #                     pred=object_pred,
    #                     name=data["name"],
    #                     object_id= c,
    #                     dsc=_dsc,
    #                     nsd=_nsd,
    #                 )

    #     elapsed_time = time.time() - start_time
    #     logger.info(f"-> Finished evaluating {n_images} images in {elapsed_time:.2f} seconds")

    #     # Return a dataframe with the results
    #     df = pd.DataFrame({
    #         "Image": names,
    #         "Object": class_labels,
    #         "DSC": dsc,
    #         "NSD": surface_dice,
    #         "IoU": iou,
    #     })

    #     # Add some metadata to the dataframe
    #     model_name = self.get_model_name()
    #     df.insert(0, "Model", model_name)
    #     df.insert(1, "n_pos", self.dataset.n_pos)
    #     df.insert(2, "n_neg", self.dataset.n_neg)
    #     df.insert(3, "bbsize", self.dataset.bbsize)
    #     df.insert(4, "Mode", self.dataset.mode)
    #     df.insert(5, "Prompt Finder", self.dataset.prompt_finder)

    #     # Clear the memory after each image to avoid CUDA out of memory errors
    #     try:
    #         del image, gt, prompts, total_segmasks, total_scores, results, data
    #         del object_gt, object_pred
    #         del _dsc, _iou, _nsd
    #         del names, class_labels, dsc, surface_dice, iou
    #         del data_list
    #     except Exception as e:
    #         logger.error(f"Error clearing memory: {e}")

    #     logger.info(
    #         f" DSC - Average: {np.mean(df['DSC']):.4f}, Median: {np.median(df['DSC']):.4f}"
    #     )
    #     logger.info(
    #         f" IOU - Average: {np.mean(df['IoU']):.4f}, Median: {np.median(df['IoU']):.4f}"
    #     )
    #     logger.info(
    #         f" NSD - Average: {np.mean(df['NSD']):.4f}, Median: {np.median(df['NSD']):.4f}"
    #     )

    #     return df

    def get_model_name(self):
        """
        Returns the name of the model.
        """
        return "SAM-Med2D"