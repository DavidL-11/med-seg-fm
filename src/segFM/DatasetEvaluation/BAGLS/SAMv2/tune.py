import os
import pandas as pd
import cv2
import torch
import torch.nn.utils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from segFM.DataLoaders.bagls import BAGLSImagesFull
from segFM import checkpoints

SAM2_CHECKPOINT = checkpoints.SAM2_tiny
MODEL_CONFIG = checkpoints.SAM2_tiny_cfg
dataset = BAGLSImagesFull(mode="point", n_pos=1, n_neg=0, bbsize=0)


def read_batch(img, binary_mask, prompt, visualize_data=False):
    # Initialize a single binary mask with one channel
    #binary_mask = np.zeros_like(img[..., 0], dtype=np.uint8)
    #points = []

    # Get binary masks and combine them into a single mask
    #mask = cv2.resize(mask, (int(mask.shape[1] * r), int(mask.shape[0] * r)), interpolation=cv2.INTER_NEAREST)
    #binary_mask = np.maximum(binary_mask, binary_mask)  # Combine with the existing binary mask
    #points.append(prompt.point[0])  # Assuming one point per prompt

    #points = np.array(points)
    points = prompt.point

    if visualize_data:
        # Plotting the images and points
        plt.figure(figsize=(15, 5))

        # Original Image
        plt.subplot(1, 3, 1)
        plt.title('Original Image')
        plt.imshow(img)
        plt.axis('off')

        # Segmentation Mask (binary_mask)
        plt.subplot(1, 3, 2)
        plt.title('Binarized Mask')
        plt.imshow(binary_mask, cmap='gray')
        plt.axis('off')

        # Mask with Points in Different Colors
        plt.subplot(1, 3, 3)
        plt.title('Binarized Mask with Points')
        plt.imshow(binary_mask, cmap='gray')

        # Plot points in different colors
        colors = list(mcolors.TABLEAU_COLORS.values())
        for i, point in enumerate(points):
            plt.scatter(point[0], point[1], c=colors[i % len(colors)], s=100, label=f'Point {i+1}')  # Corrected to plot y, x order

        # plt.legend()
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    binary_mask = np.expand_dims(binary_mask, axis=-1)  # Now shape is (1024, 1024, 1)
    binary_mask = binary_mask.transpose((2, 0, 1))
    points = np.expand_dims(points, axis=1)

    # Return the image, binarized mask, points, and number of masks
    return img, binary_mask, points, 1


# Visualize the data
test_data = dataset[0]
test_img = test_data["img"]
test_gt = test_data["gt"]
test_prompts = test_data["prompts"]
for prompt in test_prompts:
    test_obj_gt = test_gt[prompt.channel]
    img1, masks1, points1, num_masks = read_batch(test_img, test_obj_gt, prompt, visualize_data=True)

sam2_model = build_sam2(MODEL_CONFIG, SAM2_CHECKPOINT, device="cuda")
predictor = SAM2ImagePredictor(sam2_model)

# Train mask decoder.
predictor.model.sam_mask_decoder.train(True)

# Train prompt encoder.
predictor.model.sam_prompt_encoder.train(True)

# Configure optimizer.
optimizer=torch.optim.AdamW(params=predictor.model.parameters(),lr=0.0001,weight_decay=1e-4) #1e-5, weight_decay = 4e-5

# Mix precision.
scaler = torch.amp.GradScaler()

# No. of steps to train the model.
NO_OF_STEPS = 1000 # @param 

# Fine-tuned model name.
FINE_TUNED_MODEL_NAME = "bagls_sam2t"

# Initialize scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.2) # 500 , 250, gamma = 0.1
accumulation_steps = 4  # Number of steps to accumulate gradients before updating

indeces = dataset.get_n_nonduplicate_indeces(NO_OF_STEPS)

for step in range(1, NO_OF_STEPS + 1):
    with torch.amp.autocast(device_type="cuda"):
        data_obj = dataset[indeces[step - 1]]
        gt = data_obj["gt"]
        prompts = data_obj["prompts"]
        img = data_obj["img"]
        for prompt in prompts:
            obj_gt = gt[prompt.channel]

            image, mask, input_point, num_masks = read_batch(img, obj_gt, prompt, visualize_data=False)
            if image is None or mask is None or num_masks == 0:
                continue

            input_label = np.ones((num_masks, 1))
            if not isinstance(input_point, np.ndarray) or not isinstance(input_label, np.ndarray):
                continue

            if input_point.size == 0 or input_label.size == 0:
                continue

            predictor.set_image(image)
            mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(input_point, input_label, box=None, mask_logits=None, normalize_coords=True)
            if unnorm_coords is None or labels is None or unnorm_coords.shape[0] == 0 or labels.shape[0] == 0:
                continue

            sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                points=(unnorm_coords, labels), boxes=None, masks=None,
            )

            batched_mode = unnorm_coords.shape[0] > 1
            high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
            low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
                image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
                repeat_image=batched_mode,
                high_res_features=high_res_features,
            )
            prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])

            gt_mask = torch.tensor(mask.astype(np.float32)).cuda()
            prd_mask = torch.sigmoid(prd_masks[:, 0])
            seg_loss = (-gt_mask * torch.log(prd_mask + 0.000001) - (1 - gt_mask) * torch.log((1 - prd_mask) + 0.00001)).mean()

            inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
            iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
            score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
            loss = seg_loss + score_loss * 0.05

            # Apply gradient accumulation
            loss = loss / accumulation_steps
            scaler.scale(loss).backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(predictor.model.parameters(), max_norm=1.0)

            if step % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                predictor.model.zero_grad()

        # Update scheduler
        scheduler.step()

        if step % 500 == 0 or step == NO_OF_STEPS:
            FINE_TUNED_MODEL = FINE_TUNED_MODEL_NAME + "_" + str(step) + ".torch"
            # Place in ./finetuned/ directory
            FINE_TUNED_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "finetuned", FINE_TUNED_MODEL)
            torch.save(predictor.model.state_dict(), FINE_TUNED_MODEL)

        if step == 1:
            mean_iou = 0

        mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())

        if step % 100 == 0:
            print("Step " + str(step) + ":\t", "Accuracy (IoU) = ", mean_iou)