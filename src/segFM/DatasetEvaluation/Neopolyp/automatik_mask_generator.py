from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import matplotlib.pyplot as plt
import numpy as np

from segFM.DataLoaders.neopolyp import NeopolypDataset
from segFM import checkpoints, utils


def show_anns(anns, img, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3)*255])
        # Set every pixel where m is True to the color
        img[m] = color_mask
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1), thickness=1) 

    return img

dataset = NeopolypDataset(
    bbsize=0,
    mode="box",
    prompt_finder="random",
    n_pos=1,
    n_neg=0,
    plot_prompts=False
)

device = utils.setup_device()

sam2_checkpoint = checkpoints.SAM2_base
model_cfg = checkpoints.SAM2_base_cfg

sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)

mask_generator = SAM2AutomaticMaskGenerator(
    model=sam2,
    points_per_side=32,
    points_per_batch=32,
    pred_iou_thresh=0.7,
    stability_score_thresh=0.9,
    stability_score_offset=0.7,
    crop_n_layers=1,
    box_nms_thresh=0.7,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=120.0,
    use_m2m=True,
)
imgs_arr = []
masks_arr = []

for i in range(5):
    print(f"Processing image {i+1}/5")
    data = dataset.get_random_image()
    image = data["img"]

    masks = mask_generator.generate(image)

    for mask in masks:
        # dict_keys(['segmentation', 'area', 'bbox', 'predicted_iou', 'point_coords', 'stability_score', 'crop_box'])
        print(f"Mask area: {mask['area']}, Predicted IoU: {mask['predicted_iou']:.2f}, Stability score: {mask['stability_score']:.2f}")

    mask = show_anns(masks, np.zeros_like(image), borders=True)
    imgs_arr.append(data["gt"])
    masks_arr.append(mask)

# Plot the images and masks
fig, axs = plt.subplots(2, 5, figsize=(20, 8))
for i in range(5):
    axs[0, i].imshow(imgs_arr[i])
    axs[0, i].set_title(f"Ground Truth #{i+1}")
    axs[0, i].axis("off")

    axs[1, i].imshow(masks_arr[i])
    axs[1, i].set_title(f"Generated Mask #{i+1}")
    axs[1, i].axis("off")
plt.tight_layout()
plt.show()