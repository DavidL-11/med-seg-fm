import numpy as np
import cv2
import matplotlib.pyplot as plt
from numba import njit, prange
from sklearn.cluster import AgglomerativeClustering

from segFM.predictors.sam2_2d import SAM2ImageSegmenter
from segFM.DataLoaders.neopolyp import NeopolypDataset
from segFM.DataLoaders.endoscapes import EndoscapesDataset
from segFM.DataLoaders.imed361 import IMed361
from segFM.DataLoaders.bagls import BAGLS_Images
from segFM import checkpoints
from segFM.prompts import Prompt


def generate_prompt_point_grid(img, n_rows, n_cols):
    """
    Generate a grid of points for the given image.

    Args:
        img (np.ndarray): The input image.
        n_rows (int): Number of rows in the grid.
        n_cols (int): Number of columns in the grid.

    Returns:
        np.ndarray: An array of points in the format [[x1, y1], [x2, y2], ...].
    """
    print(img.shape)
    h, w = img.shape[:2]
    print(h)
    print(w)
    x_coords = np.linspace(0, w - 1, n_cols)
    y_coords = np.linspace(0, h - 1, n_rows)
    print(x_coords)
    print(y_coords)
    # Remove the border points that are too close to the edges
    x_coords = x_coords[1:-1].astype(np.int32)  # Remove first and last column
    y_coords = y_coords[1:-1].astype(np.int32)  # Remove first and last row

    points = np.array(np.meshgrid(x_coords, y_coords)).T.reshape(-1, 2)

    print(points.shape)

    print(points)

    # Labels are just a list of 1s, one for each point
    labels = np.ones(points.shape[0], dtype=np.int32)

    return points.astype(np.float32), labels

# ============================================================================
# 1) NUMBA‐ACCELERATED VERSION
# ============================================================================
# 

@njit(parallel=True)
def compute_dice_numba(flat_imgs, sums):
    """
    flat_imgs : uint8 array of shape (N, P) with values 0 or 1
    sums      : int32 array of shape (N,) precomputed sums per image
    Returns   : float32 Dice matrix of shape (N, N)
    """
    N, P = flat_imgs.shape
    dice = np.zeros((N, N), dtype=np.float32)
    for i in prange(N):
        for j in range(i, N):
            # fast bitwise AND count
            inter = 0
            for k in range(P):
                inter += flat_imgs[i, k] & flat_imgs[j, k]
            d = 2.0 * inter / (sums[i] + sums[j]) if (sums[i] + sums[j]) > 0 else 0.0
            dice[i, j] = d
            dice[j, i] = d
    return dice

def cluster_and_pick(dice_mat, thresh=0.1):
    """
    dice_mat : (N,N) matrix of pairwise Dice
    thresh   : maximum 1−Dice distance for clustering
    Returns  : labels (N,), list of dominant indices per cluster
    """
    # distance = 1 - overlap
    dist = 1.0 - dice_mat
    # cluster: no preset n_clusters, just threshold
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric='precomputed',
        linkage='average',
        distance_threshold=thresh
    ).fit(dist)
    labels = clustering.labels_
    dominants = []
    for lab in np.unique(labels):
        idx = np.where(labels == lab)[0]
        sub = dice_mat[np.ix_(idx, idx)]
        # medoid = argmax of sum of similarities
        scores = sub.sum(axis=1)
        dominants.append(idx[np.argmax(scores)])
    return labels, dominants

def dominant_images_numba(imgs, cluster_thresh=0.1):
    """
    imgs           : bool or uint8 array of shape (N, H, W)
    cluster_thresh : threshold on 1−Dice for clustering
    Returns        : (labels, dominants, dice_matrix)
    """
    N, H, W = imgs.shape
    flat = imgs.reshape(N, H*W).astype(np.uint8)
    sums = flat.sum(axis=1).astype(np.int32)
    dice = compute_dice_numba(flat, sums)
    labels, dominants = cluster_and_pick(dice, thresh=cluster_thresh)
    return labels, dominants, dice

# ============================================================================

dataset = BAGLS_Images()
dataset = EndoscapesDataset()
dataset = IMed361(dataset_id=2)

segmenter = SAM2ImageSegmenter(
    checkpoint=checkpoints.MedSAM2_latest,
    config=checkpoints.MedSAM_cfg,
    fine_tuned_weights=None
)


data = dataset.get_n_nonduplicate_images(15)[1]
img = data['img']
gt = data['gt']


points, labels = generate_prompt_point_grid(img, 40, 20)

total_masks = []


prompts = [Prompt(point=np.array([point]), label=np.array([label]), box=None) for point, label in zip(points, labels)]
    
segmasks, scores, prompts_new = segmenter.segment_image_multiprompt(
    image=img,
    prompts=prompts
)

for mask, score in zip(segmasks, scores):
    obj_mask = mask  # Get the mask
    obj_mask[obj_mask > 0] = 1  # Ensure binary mask
    obj_mask = obj_mask.astype(np.uint8)  # Convert to uint8
    if score > 0.1:
        total_masks.append(obj_mask)

### Combination of all masks before clustering for comparison
combined_masks = np.zeros_like(total_masks[0], dtype=np.uint8)
colors = np.linspace(0, 255, len(total_masks), dtype=np.uint8)
# Combine all masks into an image
for mask, color in zip(total_masks, colors):
    combined_masks[mask > 0] = color

### Combination of all masks after clustering
# Use dice score clustering to find dominant masks
labels, dominants, dice = dominant_images_numba(np.array(total_masks), cluster_thresh=0.7)

# Combine the dominant masks into a single mask
total_mask = np.zeros_like(total_masks[0], dtype=np.uint8)
colors = np.linspace(0, 255, len(dominants), dtype=np.uint8)
for i, dominant in enumerate(dominants):
    total_mask[total_masks[dominant] > 0] = colors[i]

### Visualization of the prompts on the image
for point in points:
    cv2.circle(img, tuple(point.astype(int)), 2, (0, 255, 0), -1)

# Plot the final mask
plt.figure(figsize=(10, 3))
plt.axis('off')
plt.subplot(1, 4, 1)
plt.imshow(img)
plt.title("Image")
plt.axis('off')

if len(gt.shape) == 3 and gt.shape[-1] > 3: # Convert labelmap C, H, W to single channel with random color
    # GT single is RGB image
    gt_single = np.zeros(gt.shape[1:] + (3,), dtype=np.uint8)
    for c in range(gt.shape[0]):
        gt_single[gt[c] > 0] = [np.random.randint(0,256), np.random.randint(0,256), np.random.randint(0,256)]
else:
    gt_single = gt

plt.subplot(1, 4, 2)
# Remove coordinates and axes for saving
plt.axis('off')
plt.imshow(gt_single, cmap='gray')
plt.title("GT")

plt.subplot(1, 4, 3)
plt.imshow(combined_masks, cmap='jet')
plt.title("Before Clustering")
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(total_mask, cmap='jet')
plt.title("After Clustering")
plt.axis('off')


plt.tight_layout()
plt.savefig("medsam2_automatic_mask_generation.png", bbox_inches='tight', dpi=300)
plt.savefig("medsam2_automatic_mask_generation.pdf", bbox_inches='tight')
plt.savefig("medsam2_automatic_mask_generation.svg", bbox_inches='tight')

plt.show()