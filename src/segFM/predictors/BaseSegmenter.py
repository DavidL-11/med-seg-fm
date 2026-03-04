import cv2
import time
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

from segFM.logger import logger
from segFM import utils
from segFM.prompts import Prompt

class BaseSegmenter:
    def __init__(self):
        pass

    def evaluate_model(self, dataset, n_images, plot_results=False, save_dataset_name=False) -> pd.DataFrame:
        """
        Evaluates the model on a random set of images.

        Returns:
            pd.DataFrame: A DataFrame containing the evaluation results.
        """
        # Prepare lists to store the results
        dsc = []
        surface_dice = []
        iou = []
        names = []
        class_labels = []
    
        # Get n images from the dataset
        data_list = dataset.get_n_nonduplicate_images(n_images)

        for i, data in enumerate(data_list):
            if i % max(n_images // 10, 1) == 0:
                if i == 0:
                    start_time = time.time()
                    logger.info("[0s] Starting evaluation of images...")
                else:
                    elapsed_time = time.time() - start_time
                    logger.info(f"[{elapsed_time:.1f}s] Evaluating image {i} of {n_images}... - Current DSC: {np.mean(dsc):.3f}")

            # Extract the information from the data dictionary
            image = data["img"]
            gt = data["gt"]
            prompts = data["prompts"]

            if gt.sum() == 0:
                # No valid ground truth was found, skip this image / segmentation mask is empty
                continue
            
            # Store the actual objects from the ground truth mask (= the unique colors in the mask)
            unique_colors = list(np.unique(gt)[1:])  # Exclude background (0)

            # If the GT is a labelmap (3D mask with multiple channels for different objects),
            # still only create a single channel for each object
            # and use the channel index as the color
            shape = gt.shape if gt.ndim == 2 else (gt.shape[1], gt.shape[2])

            results = {
                np.int16(color): np.zeros(shape, dtype=np.uint8) for color in unique_colors
            }

            segmasks, scores, total_prompts = self.segment_image_multiprompt(image, prompts)
            
            if segmasks is None:
                logger.warning(f"No segmentation masks were returned for image {data['name']}, skipping...")
                continue

            # Enumerate over all prompts generated
            for i, prompt in enumerate(total_prompts):
                color = np.int16(prompt.color)

                if color in results.keys():
                    # Set the image in the results dict to 255 everywhere the segmentation mask is > 0
                    # We do this since maybe multiple prompts are generated for the same color, so results are accumulated
                    results[color] = np.where(segmasks[i] > 0.5, 1, results[color])
                else:
                    logger.warning(f"Color {color} / Object {prompt.class_label}[{prompt.class_id}] not found in results, skipping...")

            # Calculate the metrics for each unique color in the results
            for c in unique_colors:
                if gt.ndim == 2: #Normal 2D ground truth mask with different colors for different objects
                    object_gt = (gt == c).astype(np.uint8)
                elif gt.ndim == 3: # Labelmap ground truth mask with multiple channels for different objects
                    object_gt = (gt[c - 1, :, :] > 0).astype(np.uint8)

                object_pred = results[c]  # Get the predicted mask for the object

                _dsc, _iou, _nsd = utils.compute_metrics(object_pred, object_gt)

                dsc.append(_dsc)
                iou.append(_iou)
                surface_dice.append(_nsd)
                class_labels.append(dataset.color_to_label.get(c, f"Unknown"))
                names.append(data["name"])

                if plot_results:
                    image_plt = image.copy()
                    for prompt in prompts:
                        if prompt.color == c or (prompt.channel is not None and prompt.channel + 1 == c):
                            if hasattr(prompt, "point") and prompt.point is not None:
                                # Plot the points on the image
                                for point in prompt.point:
                                    cv2.circle(image_plt, tuple(point), 5, (255, 0, 0), -1)
                            if hasattr(prompt, "box") and prompt.box is not None:
                                # Plot the bounding box on the image
                                x1, y1, x2, y2 = prompt.box
                                cv2.rectangle(image_plt, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    utils.plot_img_gt_pred(
                        img=image_plt,
                        gt=object_gt,
                        pred=object_pred,
                        name=data["name"],
                        object_id=dataset.color_to_label.get(c, f"{c}"),
                        dsc=_dsc,
                        nsd=_nsd,
                    )

        elapsed_time = time.time() - start_time
        logger.info(f"-> Finished evaluating {n_images} images in {elapsed_time:.2f} seconds")

        # Return a dataframe with the results
        df = pd.DataFrame({
            "Image": names,
            "Object": class_labels,
            "DSC": dsc,
            "NSD": surface_dice,
            "IoU": iou,
        })

        # Add some metadata to the dataframe
        model_name = self.get_model_name()
        df.insert(0, "Model", model_name)
        df.insert(1, "n_pos", dataset.n_pos)
        df.insert(2, "n_neg", dataset.n_neg)
        df.insert(3, "bbsize", dataset.bbsize)
        df.insert(4, "Mode", dataset.prompt_finder if "BOB" in dataset.prompt_finder else dataset.mode)
        df.insert(5, "Prompt Finder", dataset.prompt_finder)

        if save_dataset_name:
            df.insert(6, "Dataset", dataset.dataset_name.replace("_", " "))

        logger.info(f" DSC - Average: {np.mean(df['DSC']):.4f}, Median: {np.median(df['DSC']):.4f}")
        logger.info(f" IOU - Average: {np.mean(df['IoU']):.4f}, Median: {np.median(df['IoU']):.4f}")
        logger.info(f" NSD - Average: {np.mean(df['NSD']):.4f}, Median: {np.median(df['NSD']):.4f}")

        return df
    
    def segment_image_multiprompt(self, image, prompts: list[Prompt]) -> tuple[list[np.ndarray], list[float], list[Prompt]]:
        raise NotImplementedError("Subclasses should implement this method.")
    
    def get_model_name(self) -> str:
        raise NotImplementedError("Subclasses should implement this method.")
    

    def _calculate_tp_fp_for_image(self, predicted_prompts: list[Prompt], gt_prompts: list[Prompt], iou_threshold: float = 0.5) -> tuple[list, list, list]:
        """
        Calculate TP/FP for a single image and return confidence scores.
        
        Args:
            predicted_prompts: List of predicted prompts with bounding boxes and confidence scores
            gt_prompts: List of ground truth prompts with bounding boxes
            iou_threshold: IoU threshold for considering a detection as true positive
            
        Returns:
            tuple: (tp_list, fp_list, confidence_list) for this image
        """
        if not predicted_prompts or not gt_prompts:
            return [], [], []
            
        # Sort predicted prompts by confidence (descending)
        predicted_prompts = sorted(predicted_prompts, key=lambda p: getattr(p, 'confidence', 1.0), reverse=True)
        
        # Track which ground truth boxes have been matched
        gt_matched = [False] * len(gt_prompts)
        tp = []  # True positives
        fp = []  # False positives
        confidences = []  # Confidence scores
        
        for pred_prompt in predicted_prompts:
            confidence = getattr(pred_prompt, 'confidence', 1.0)
            confidences.append(confidence)
            
            if not hasattr(pred_prompt, 'box') or pred_prompt.box is None:
                fp.append(1)
                tp.append(0)
                continue
                
            best_iou = 0.0
            best_gt_idx = -1
            
            # Find best matching ground truth box
            for gt_idx, gt_prompt in enumerate(gt_prompts):
                if gt_matched[gt_idx] or not hasattr(gt_prompt, 'box') or gt_prompt.box is None:
                    continue
                    
                iou = self._calculate_box_iou(pred_prompt.box, gt_prompt.box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # Check if it's a true positive
            if best_iou >= iou_threshold and best_gt_idx != -1:
                tp.append(1)
                fp.append(0)
                gt_matched[best_gt_idx] = True
            else:
                tp.append(0)
                fp.append(1)
        
        return tp, fp, confidences
    

    def _calculate_box_iou(self, box1, box2) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            box1: First bounding box [x1, y1, x2, y2]
            box2: Second bounding box [x1, y1, x2, y2]
            
        Returns:
            float: IoU score
        """
        # Convert to numpy arrays if needed
        box1 = np.array(box1)
        box2 = np.array(box2)
        
        # Calculate intersection area
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
            
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union area
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _generate_gt_prompts_from_mask(self, gt_mask, dataset) -> list[Prompt]:
        """
        Generate ground truth prompts (bounding boxes) from a segmentation mask.
        
        Args:
            gt_mask: Ground truth segmentation mask
            dataset: Dataset object containing color mappings
            
        Returns:
            list[Prompt]: List of ground truth prompts with bounding boxes
        """
        gt_prompts = []
        unique_colors = list(np.unique(gt_mask)[1:])  # Exclude background (0)
        
        for color in unique_colors:
            if gt_mask.ndim == 2:
                object_mask = (gt_mask == color).astype(np.uint8)
            elif gt_mask.ndim == 3:
                object_mask = (gt_mask[color - 1, :, :] > 0).astype(np.uint8)
            else:
                continue
                
            if object_mask.sum() == 0:
                continue
                
            # Find bounding box
            coords = np.where(object_mask > 0)
            if len(coords[0]) == 0:
                continue
                
            y_min, y_max = coords[0].min(), coords[0].max()
            x_min, x_max = coords[1].min(), coords[1].max()
            
            # Create ground truth prompt
            gt_prompt = Prompt(
                box=[x_min, y_min, x_max, y_max],
                color=color,
                class_label=dataset.color_to_label.get(color, f"Class_{color}"),
                generated_by="ground_truth",
                confidence=1.0
            )
            gt_prompts.append(gt_prompt)
            
        return gt_prompts

    def evaluate_bob(self, dataset, n_images, model_name="BOB D-FINE-N", save_dataset_name=False, iou_threshold=0.5) -> pd.DataFrame:
        """
        Evaluates BOB's performance using mAP calculated from precision-recall curve.
        Collects TP/FP across all images and calculates overall mAP as area under PR curve.
        
        Args:
            dataset: Dataset to evaluate on
            n_images: Number of images to evaluate
            plot_results: Whether to plot results for each image
            save_dataset_name: Whether to include dataset name in results
            iou_threshold: IoU threshold for TP/FP calculation
            
        Returns:
            pd.DataFrame: DataFrame with evaluation results
        """
        # Prepare lists to collect results across all images
        all_tp = []
        all_fp = []
        all_confidences = []
        total_gt_objects = 0
        
        # Per-image statistics for DataFrame
        names = []
        avg_confidences = []
        n_predicted_prompts = []
        n_gt_objects = []
        
        # Get n images from the dataset
        data_list = dataset.get_n_nonduplicate_images(n_images)
        
        for i, data in enumerate(data_list):
            if i % max(n_images // 10, 1) == 0:
                if i == 0:
                    start_time = time.time()
                    logger.info("[0s] Starting BOB evaluation...")
                else:
                    elapsed_time = time.time() - start_time
                    # Calculate interim mAP for progress reporting using sklearn
                    if all_tp and total_gt_objects > 0:
                        try:
                            current_map = average_precision_score(np.array(all_tp), np.array(all_confidences))
                        except:
                            current_map = 0.0
                    else:
                        current_map = 0.0
                    logger.info(f"[{elapsed_time:.1f}s] Evaluating image {i} of {n_images}... - Current mAP: {current_map:.3f}")
            
            # Extract the information from the data dictionary
            image = data["img"]
            gt = data["gt"]
            predicted_prompts = data["prompts"]  # BOB-generated prompts
            
            if gt.sum() == 0:
                # No valid ground truth was found, skip this image
                continue
            
            # Generate ground truth prompts from the segmentation mask
            gt_prompts = self._generate_gt_prompts_from_mask(gt, dataset)
            
            if not gt_prompts:
                continue
            
            # Calculate TP/FP for this image
            tp_list, fp_list, confidences = self._calculate_tp_fp_for_image(predicted_prompts, gt_prompts, iou_threshold)
            
            # Add to overall lists
            all_tp.extend(tp_list)
            all_fp.extend(fp_list)
            all_confidences.extend(confidences)
            total_gt_objects += len(gt_prompts)
            
            # Calculate per-image statistics
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            # Store per-image results
            names.append(data["name"])
            avg_confidences.append(avg_confidence)
            n_predicted_prompts.append(len(predicted_prompts))
            n_gt_objects.append(len(gt_prompts))
            
        
        # Calculate precision-recall curve and mAP using sklearn
        if len(all_tp) > 0 and total_gt_objects > 0:
            # Convert to binary classification format for sklearn
            y_true = np.array(all_tp)  # 1 for TP, 0 for FP
            y_scores = np.array(all_confidences)  # confidence scores
            
            # Calculate precision-recall curve and mAP using sklearn
            precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_scores)
            ap = average_precision_score(y_true, y_scores)
        else:
            ap = 0.0
            precision_curve = np.array([])
            recall_curve = np.array([])
        
        elapsed_time = time.time() - start_time if 'start_time' in locals() else 0
        logger.info(f"-> Finished evaluating BOB on {n_images} images in {elapsed_time:.2f} seconds")
        
        # Create DataFrame with per-image results
        df = pd.DataFrame({
            "Image": names,
            "Avg_Confidence": avg_confidences,
            "N_Predicted": n_predicted_prompts,
            "N_GT_Objects": n_gt_objects,
        })
        # Get all confidences for TP
        tp_confidences = [conf for tp, conf in zip(all_tp, all_confidences) if tp == 1]
        fp_confidences = [conf for fp, conf in zip(all_fp, all_confidences) if fp == 1]
        average_confidence = np.mean(tp_confidences) if tp_confidences else 0.0
        average_confidence_bottom_20 = np.mean(sorted(tp_confidences)[:max(1, int(0.2 * len(tp_confidences)))]) if tp_confidences else 0.0

        # Calculate the bottom 20% average confidence for FP
        average_confidence_bottom_20_fp = np.mean(sorted(fp_confidences)[:max(1, int(0.2 * len(fp_confidences)))]) if fp_confidences else 0.0

        logger.info(f" Average Confidence - Average: {average_confidence:.4f}")
        logger.info(f" TP Average Confidence - Bottom 20%: {average_confidence_bottom_20:.4f}")
        logger.info(f" FP Average Confidence - Bottom 20%: {average_confidence_bottom_20_fp:.4f}")

        # Add overall mAP as a column (same value for all rows)
        df.insert(1, "mAP", ap)
        
        # Add some metadata to the dataframe
        df.insert(0, "Model", model_name)
        df.insert(5, "Mode", dataset.prompt_finder if "BOB" in dataset.prompt_finder else dataset.mode)
        df.insert(6, "Prompt Finder", dataset.prompt_finder)
        df.insert(7, "IoU_Threshold", iou_threshold)
        
        if save_dataset_name:
            df.insert(8, "Dataset", dataset.dataset_name.replace("_", " "))
        
        logger.info(f" Overall mAP: {ap:.4f}")
        logger.info(f" Avg Confidence - Average: {average_confidence:.4f}")
        logger.info(f" N Predicted - Average: {np.mean(df['N_Predicted']):.1f}, Total: {np.sum(df['N_Predicted'])}")
        logger.info(f" N GT Objects - Average: {np.mean(df['N_GT_Objects']):.1f}, Total: {np.sum(df['N_GT_Objects'])}")
        logger.info(f" Total TP: {np.sum(all_tp)}, Total FP: {np.sum(all_fp)}")

        self.plot_precision_recall_curve(precision_curve, recall_curve, ap, average_confidence, title=f"Precision-Recall Curve for {model_name}", filename=f"pr_curve_{model_name}_{dataset.dataset_name}.png")
        
        # Plot recall-confidence curve if we have data
        if len(all_tp) > 0 and total_gt_objects > 0:
            self.plot_recall_confidence_curve(
                y_true=np.array(all_tp), 
                y_scores=np.array(all_confidences), 
                total_gt_objects=total_gt_objects,
                title=f"Recall-Confidence Curve for {model_name}",
                filename=f"recall_conf_curve_{model_name}_{dataset.dataset_name}.png"
            )
        
        return df

    def plot_precision_recall_curve(self, precision_curve: np.ndarray, recall_curve: np.ndarray, ap: float, average_confidence: float, title: str = "Precision-Recall Curve", filename: str = "pr_curve.png"):
        """
        Plot the precision-recall curve using matplotlib.
        
        Args:
            precision_curve: Array of precision values
            recall_curve: Array of recall values  
            map_score: mAP score to display in the title
            title: Title for the plot
        """
        if len(precision_curve) == 0 or len(recall_curve) == 0:
            logger.warning("No data to plot precision-recall curve")
            return
        
        precision_monotonic = np.maximum.accumulate(precision_curve)

        plt.step(recall_curve, precision_monotonic, where='post', label='Precision', color='red')
        plt.xlabel("Recall", fontsize=16)
        plt.ylabel("Precision", fontsize=16)
        plt.legend()
        plt.title(f'{title}\nAP = {ap:.3f} | Avg TP Conf = {average_confidence:.2f}', fontsize=16)
        plt.grid()
        plt.legend(fontsize=12)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        
        # Save the plot
        plt.savefig(filename)
        plt.show()

    def plot_recall_confidence_curve(self, y_true: np.ndarray, y_scores: np.ndarray, total_gt_objects: int, title: str = "Recall-Confidence Curve", filename: str = "recall_conf_curve.png"):
        """
        Plot the recall-confidence curve showing how recall varies with confidence threshold.
        
        Args:
            y_true: Binary array indicating TP (1) or FP (0) for each detection
            y_scores: Confidence scores for each detection
            total_gt_objects: Total number of ground truth objects across all images
            title: Title for the plot
            filename: Filename to save the plot
        """
        if len(y_true) == 0 or len(y_scores) == 0:
            logger.warning("No data to plot recall-confidence curve")
            return
        
        # Sort detections by confidence (descending)
        sorted_indices = np.argsort(y_scores)[::-1]
        sorted_y_true = y_true[sorted_indices]
        sorted_confidences = y_scores[sorted_indices]
        
        # Calculate cumulative TP and recall at different confidence thresholds
        cumulative_tp = np.cumsum(sorted_y_true)
        recall_values = cumulative_tp / total_gt_objects if total_gt_objects > 0 else np.zeros_like(cumulative_tp)
        
        # Create confidence thresholds (unique sorted confidence values)
        unique_confidences = np.unique(sorted_confidences)[::-1]  # Descending order
        recall_at_thresholds = []
        
        for conf_threshold in unique_confidences:
            # Find detections above this threshold
            above_threshold = sorted_confidences >= conf_threshold
            tp_above_threshold = np.sum(sorted_y_true[above_threshold])
            recall_at_threshold = tp_above_threshold / total_gt_objects if total_gt_objects > 0 else 0
            recall_at_thresholds.append(recall_at_threshold)
        
        recall_at_thresholds = np.array(recall_at_thresholds)
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(unique_confidences, recall_at_thresholds, 'b-', linewidth=2, label='Recall')
        plt.xlabel("Confidence Threshold", fontsize=16)
        plt.ylabel("Recall", fontsize=16)
        plt.title(f'{title}\nMax Recall = {np.max(recall_at_thresholds):.3f}', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        logger.info(f"Recall-confidence curve saved as {filename}")
