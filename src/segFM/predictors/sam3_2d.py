from transformers import Sam3Processor, Sam3Model, Sam3TrackerProcessor, Sam3TrackerModel
import torch
import numpy as np
from segFM.prompts import Prompt
from segFM import utils
from segFM.predictors.BaseSegmenter import BaseSegmenter

class SAM3ImageSegmenter(BaseSegmenter):
    def __init__(
        self,
        mode: str = "visual",
    ):
        self.mode = mode

        # Setup device depending on hardware (CPU, CUDA, MPS)
        self.device = utils.setup_device()

        # load the model
        if mode in ["text", "context"]:
            self.sam3 = Sam3Model.from_pretrained("facebook/sam3").to(self.device)
            self.processor = Sam3Processor.from_pretrained("facebook/sam3")
        elif mode == "visual":
            self.sam3 = Sam3TrackerModel.from_pretrained("facebook/sam3").to(self.device)
            self.processor = Sam3TrackerProcessor.from_pretrained("facebook/sam3")
        else:
            raise ValueError(f"Unsupported mode {mode} for SAM3ImageSegmenter - choose from ['text', 'context', 'visual'].")


    def segment_image_multiprompt(self, image, prompts: list[Prompt]):
        """
        Predicts the segmentation mask for a given image and a list of prompts.
        """
        segmasks = []
        scores = []
        prompts_new = []
        
        if prompts is None or len(prompts) == 0:
            return None, None, None
        
        if self.mode == "visual":
            mask = self.predict_multiprompt_visual(image, prompts)
            assert len(mask) == len(prompts), "Number of masks must match number of prompts."
            for i in range(len(mask)):
                # Get the segmentation mask from the predicted masks
                #print(f"Mask shape: {mask[i].cpu().numpy().shape}")
                segmask = utils.create_binary_segmentation(mask[i].cpu().numpy())
                segmasks.append(segmask)
            return segmasks, None, prompts

        for prompt in prompts:
            if self.mode == "text":
                mask, box, score = self.process_text_prompt(image, prompt)
                # plot mask and GT
            elif self.mode == "context":
                mask, box, score = self.predict_context(image, prompt)
            elif self.mode == "visual":
                mask, box, score = self.predict_visual(image, prompt)
            else:
                raise ValueError(f"Unsupported mode {self.mode} for SAM3ImageSegmenter - choose from ['text', 'context', 'visual'].")


            for i in range(len(mask)):
                # Get the segmentation mask from the predicted masks
                #print(f"Mask shape: {mask[i].cpu().numpy().shape}")
                segmask = utils.create_binary_segmentation(mask[i].cpu().numpy())
                segmasks.append(segmask)
                scores.append(score)
                prompts_new.append(prompt)

        return segmasks, scores, prompts_new

    def process_text_prompt(self, image, prompt):
        text_prompt = prompt.class_label.replace("_", " ")
        inputs = self.processor(images=image, text=text_prompt, return_tensors="pt").to(self.device)
        
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
        
        print(f"Predicted {len(score)} masks for text prompt '{text_prompt}' with scores: {score}")
        return mask, box, score
    
    def predict_context(self, image, prompt: Prompt):
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
    
    def predict_visual(self, image, prompt: Prompt):
        """
        Predicts the segmentation mask for a given visual prompt.
        """
        if prompt.is_empty():
            return None, None, None
        
        # Turn [[array([1, 1, 1, 1, 0, 0])]] into [[[1, 1, 1, 1, 0, 0]]]
        labels = [[[int(l) for l in prompt.label]]] if prompt.label is not None else None
        
        # Turn [[array([[176,   1], [132,  50], [147, 168], ...]])]] into [[[176,   1], [132,  50], [147, 168], ...]]
        points = [[[[int(coord) for coord in point] for point in prompt.point ]]] if prompt.point is not None else None
        

        inputs = self.processor(
            images=image,
            input_boxes=[[prompt.box]] if prompt.box is not None else None,
            input_points=points,
            input_labels=labels,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.sam3(**inputs)

        #print(inputs.get("original_sizes"))

        # Post-process results
        # Output: (batch_size, num_channels, height, width)
        masks = self.processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]

        # Convert to binary masks with (h, w) shape
        # The model outputs multiple mask predictions ranked by quality score
        # Only the first channel includes the best mask
        masks = masks[:, 0, :, :]

        return masks, None, None
    
    def predict_multiprompt_visual(self , image, prompts: list[Prompt]):
        # Labels is a 3D list: # 1 for positive click, 0 for negative click, 3 dimensions (image_dim, object_dim, point_label)
        if len(prompts) < 1 or prompts[0].is_empty():
            return None
        labels = [[[int(l) for l in prompt.label] for prompt in prompts]] if prompts[0].label is not None else None
        
        # Single point click, 4 dimensions (image_dim, object_dim, point_per_object_dim, coordinates)
        points = [[[[int(coord) for coord in point] for point in prompt.point ] for prompt in prompts]] if prompts[0].point is not None else None
        
        # Boxes is a 3D list: (image_dim, object_dim, box_coordinates)
        boxes = [[prompt.box for prompt in prompts]] if prompts[0].box is not None else None
        
        inputs = self.processor(
            images=image,
            input_boxes=boxes,
            input_points=points,
            input_labels=labels,
            return_tensors="pt"
        ).to(self.device)
        
        
        with torch.no_grad():
            outputs = self.sam3(**inputs)
            
        # Post-process results
        masks = self.processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0] # Shape (object_dim, num_masks, height, width)
        
        scores = outputs.iou_scores.cpu().numpy() # Shape (image_dim, object_dim, num_masks) - usually (1, n_obj, 3)
        
        import matplotlib.pyplot as plt
        
        best_masks = []
        for i in range(masks.shape[0]):  # For each object
            # # Plot all masks for this object
            # for j in range(masks.shape[1]):
            #     plt.subplot(1, masks.shape[1], j+1)
            #     plt.imshow(masks[i, j, :, :])
            #     plt.title(f"Mask {j} - Score: {scores[0, i, j]:.2f}")
            # plt.show()
            # Find best mask (max score)
            best_idx = np.argmax(scores[0, i, :])
            best_masks.append(masks[i, best_idx, :, :])
        return best_masks
        

    def get_model_name(self):
        """
        Generates the model name based on the checkpoint and fine-tuning status.

        Returns:
            str: The name of the model and if it is fine-tuned.
        """
        # The checkpoint name without fine-tuning
        # e.g. "sam2.1_hiera_small.pt" -> "S", large -> "L" etc.
        return "SAM3 " + self.mode