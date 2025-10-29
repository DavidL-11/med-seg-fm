import nibabel as nib
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import zoom
import os
import pandas as pd

from segFM.logger import logger

from surface_distance.metrics import (
    compute_surface_distances,
    compute_surface_dice_at_tolerance,
    compute_dice_coefficient
)
from surface_distance import metrics


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def save_dataframe_to_csv(file, df: pd.DataFrame, file_name:str):
    path = os.path.dirname(os.path.abspath(file)) + f"/{file_name}"
    # Check if the csv already exists
    if os.path.exists(path):
        # Append the new data to the existing csv
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        # Create a new csv file with the data
        df.to_csv(path, index=False)

def compute_dice(gt, seg):
    """
    Computes the Dice coefficient between the ground truth and the segmentation mask.
    The Dice coefficient is a measure of overlap between two sets.
    It is represented by a value between 0 and 1, where 1 indicates perfect overlap.
    
    Parameters:
        gt (numpy.ndarray): The ground truth segmentation mask.
        seg (numpy.ndarray): The predicted segmentation mask.
        
    Returns:
        float: The Dice coefficient.
    """
    if gt.sum() == 0 and seg.sum() == 0:
        return 1.0
    return metrics.compute_dice_coefficient(gt, seg)

def compute_metrics(seg, gt):
    if seg.sum() == 0 and gt.sum() == 0:
        return 1.0, 1.0, 1.0
    elif seg.sum() == 0:
        return 0.0, 0.0, 0.0
    
    assert ((np.unique(seg)==np.unique(gt)).all() if np.unique(seg).shape == np.unique(gt).shape else True), f"Seg and GT must have the same unique values - found {np.unique(seg)} and {np.unique(gt)}"
    assert len(np.unique(seg)) <= 2, f"Segmentation must be binary (0 and 1) but found {np.unique(seg)}"
    assert len(np.unique(gt)) <= 2, f"Ground truth must be binary (0 and 1) but found {np.unique(gt)}"
    
    dsc = compute_dice(gt, seg)
    #iou = intersection_over_union(gt, seg)
    # Compute IoU using the dice score, saves computation time
    iou = dsc / (2 - dsc)
    nsd = normalized_surface_distance(gt, seg)

    if np.isnan(nsd):
        nsd = None
    return dsc, iou, nsd

def setup_device() -> torch.device:
    """
    Sets up the GPU device for computation.
    It checks if CUDA or MPS is available and sets the device accordingly.
    """
    # select the device for computation
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda").__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        logger.warning(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )

    return device

# def nifti_to_numpy_old(nii_image_path):
#     """
#     This used to work until I tried to load some annoying images with wrong orientation.
#     Times were simpler back then... *takes a sip of coffee*
#     """
#     nii_image = sitk.ReadImage(nii_image_path)
#     nii_image_data = sitk.GetArrayFromImage(nii_image)

#     nii_image_data = (
#         (nii_image_data - np.min(nii_image_data))
#         / (np.max(nii_image_data) - np.min(nii_image_data))
#         * 255.0
#     )

def nifti_to_numpy2(filepath, preprocess="None"):
    import SimpleITK as sitk
    sitk_image = sitk.ReadImage(filepath)
    arr = sitk.GetArrayFromImage(sitk_image) # Z, Y, X

    if preprocess != "None":
        # Remove outliers by clipping to 0.5/99.5 percentiles
        p05 = np.percentile(arr, 0.5)
        p995 = np.percentile(arr, 99.5)
        arr = np.clip(arr, p05, p995)
    
    # Perform windowing for CT images - Window level, window width - Clips only certain Hounsfield units
    if "CT" in preprocess:
        if "Brain" in preprocess:
            WL = 40
            WW = 80
        elif "Abdomen" in preprocess:
            WL = 40
            WW = 350
        elif "Liver" in preprocess:
            WL = 30
            WW = 150
        elif "Bone" in preprocess:
            WL = 300
            WW = 1500
        elif "Lung" in preprocess:
            WL = -600
            WW = 1500
        else:
            WL = 40
            WW = 400
        arr = np.clip(arr, WL - WW / 2, WL + WW / 2)

    if preprocess != "None":
        # Subtract mean and divide by stddev
        arr = (arr - np.mean(arr)) / np.std(arr)

    if min(np.unique(arr)) < 0 or max(np.unique(arr)) > 255:
        arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX)
    
    arr = arr.astype(np.uint8)

    return arr

def nifti_to_numpy(filepath, interpolate_spacing=False, remove_4d_dim=True):
    """
    Read a NIfTI file and convert it to a numpy array. Converts to RAS orientation.
    The output array is in the shape (Z, Y, X) and of type uint8.
    If the input data is in the shape (X, Y, Z, channels), like MedicalDecathlon, the last dimension is removed.
    If the data is not in the range of 0-255, it is normalized to this range.
    If interpolate_spacing is True, the image is resampled to isotropic spacing.
    """
    nii = nib.load(filepath)
    nii_ras = nib.as_closest_canonical(nii)
    data_ras = nii_ras.get_fdata()

    # If the data is in the shape (x, y, z, channels), like MedicalDecathlon, remove the last dimension
    if remove_4d_dim and data_ras.ndim == 4 and data_ras.shape[-1] >= 1:
        data_ras = data_ras[..., 0]

    # If the data is not in the range of 0-255, normalize it
    if min(np.unique(data_ras)) < 0 or max(np.unique(data_ras)) > 255:
        data_ras = cv2.normalize(data_ras, None, 0, 255, cv2.NORM_MINMAX)

    
    if interpolate_spacing:
        ### Not really used for anything, was just interesting to try out
        affine_ras = nii_ras.affine

        orig_spacing = np.sqrt((affine_ras[:3, :3] ** 2).sum(axis=0))

        target_spacing = min(orig_spacing)
        target_spacing = np.array(target_spacing)

        zoom_factors = orig_spacing / target_spacing

        # Resample the image to isotropic spacing
        resampled = zoom(data_ras, zoom=zoom_factors, order=1)
    else:
        # No resampling, just convert to uint8
        resampled = data_ras
    
    # Convert to SAR by swapping axes
    # Change from (x, y, z) to (z, y, x)
    resampled = np.transpose(resampled, range(resampled.ndim)[::-1])
    resampled = np.ascontiguousarray(resampled, dtype=np.uint8)
    
    return resampled


def grayscale_to_rgb(image):
    """
    Converts a grayscale image to RGB format by stacking the grayscale image across three channels.
    
    Parameters:
        image (numpy.ndarray): The input grayscale image.
        
    Returns:
        numpy.ndarray: The RGB image.
    """
    if image.shape[-1] != 1:
        # If the image is of shape (H, W) or (D, H, W) for 3D images, stack it at the last dimension
        return np.stack((image,)*3, axis=-1)
    else: 
        # If grayscale image is in shape (H, W, 1), convert it to (H, W, 3)
        return np.concatenate([image]*3, axis=-1)


def dice_score(seg, gt) -> float:
    """
    Computes the dice score between two binary images.
    The dice score is a measure of overlap between two sets.
    It is represented by a value between 0 and 1, where 1 indicates perfect overlap.
    This can be used to compare a predicted segmentation with a ground truth segmentation.
    """
    assert (
        len(np.unique(seg)) == 2 or len(np.unique(seg)) == 1
    ), "Segmentation must be binary"
    assert (
        len(np.unique(gt)) == 2 or len(np.unique(gt)) == 1
    ), "Ground truth must be binary"
    assert seg.shape == gt.shape, "Image and segmentation must have the same shape"
    assert seg.dtype == gt.dtype, "Image and segmentation must have the same dtype"

    overlap = np.count_nonzero(seg * gt)
    union = np.count_nonzero(seg) + np.count_nonzero(gt)

    # No overlap is still considered a perfect score if all pixels are 0
    if union == 0:
        return 1.0

    # to avoid possible division by zero
    return (2.0 * overlap + 1e-6) / (union + 1e-6)


def intersection_over_union(seg_mask, gt, smooth=1e-6) -> float:
    """
    Computes the intersection over union (IoU) between the predicted segmentation mask and the ground truth.
    The IoU is a measure of overlap between two sets.
    It is represented by a value between 0 and 1, where 1 indicates perfect overlap.

    Parameters:
        seg_mask (numpy.ndarray): The predicted segmentation mask.
        gt (numpy.ndarray): The ground truth segmentation mask.
    Returns:
        float: The IoU score.
    """

    # Normalize the images to 0-1 range
    seg_mask = seg_mask / 255.0
    gt = gt / 255.0

    # Calculate the intersection and union
    intersection = np.logical_and(seg_mask, gt)
    union = np.logical_or(seg_mask, gt)

    # Calculate the IoU
    iou = (np.sum(intersection) + smooth) / (np.sum(union) + smooth)

    return iou


def normalized_surface_distance(gt, seg_mask) -> float:
    """
    Computes the normalized surface distance (NSD) between the predicted segmentation mask and the ground truth.
    The NSD is a measure of the distance between two sets.
    It is represented by a value between 0 and 1, where 1 indicates perfect overlap.

    Parameters:
        seg_mask (numpy.ndarray): The predicted segmentation mask.
        gt (numpy.ndarray): The ground truth segmentation mask.
    Returns:
        float: The NSD score.
    """
    ndim = gt.ndim

    if gt.sum() == 0 and seg_mask.sum() == 0:
        return 1.0

    # Calculate the surface distances
    surface_distances = compute_surface_distances(
        gt > 0, seg_mask > 0, spacing_mm=(1,) * ndim
    )

    # Calculate the spacing depending on the size of the image
    tolerance = max(int(max(gt.shape) / 128), 1)

    # Calculate the NSD
    nsd = compute_surface_dice_at_tolerance(surface_distances, tolerance_mm=tolerance)

    return nsd


def create_binary_segmentation(mask):
    """
    Creates a binary image from the predicted mask.

    Parameters:
        mask (numpy.ndarray): The predicted mask to be converted to binary.
    Returns:
        numpy.ndarray: The binary image.
    """

    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w)

    # Ensure the mask is binary
    mask_image = (mask_image > 0).astype(np.uint8) * 255

    return mask_image

def convert_multiobject_to_multicolor(image):
    """
    Converts a multi-object segmentation mask to a multi-color (grayscale) image.
    Each connected component in the mask is assigned a unique grayscale value between 10 and 250.
    """
    # Find connected components in the image
    num_labels, labels_im = cv2.connectedComponents(image.astype(np.uint8), connectivity=4)

    # Clean connected components by removing small objects
    min_size = 50  # Minimum size of connected components to keep
    for label in range(1, num_labels):  # Skip background label 0
        component_mask = (labels_im == label)
        if np.sum(component_mask) < min_size:
            labels_im[component_mask] = 0

    # Create a color map for the unique labels
    color_map = np.linspace(10, 250, num_labels - 1, dtype=np.uint8)

    assert len(color_map) == num_labels - 1, "Color map length does not match number of labels"

    # Create an empty grayscale image to hold the mask
    multi_color_image = np.zeros_like(image, dtype=np.uint8)
    for label in range(1, num_labels):  # Skip background label 0
        # Assign a unique grayscale value to each connected component
        multi_color_image[labels_im == label] = color_map[label - 1]
    
    return multi_color_image

def generate_yolo_label(data, min_area=0.0016):
    """
    Generates a YOLO label from the bounding boxes in the data.
    Args:
        data (dict): Dictionary containing the image and bounding boxes.
        class_label (int): The class label specified in the dataset.yaml file.
    Returns:
        str: YOLO formatted label string.
    """
    prompt = data["prompts"]
    image = data["img"]
    image_width, image_height = image.shape[1], image.shape[0]

    result = ""

    for p in prompt:
        # Extract bounding box coordinates
        x1, y1, x2, y2 = p.box

        # Convert bounding box coordinates to YOLO format
        x_center = ((x1 + x2) / 2) / image_width
        y_center = ((y1 + y2) / 2) / image_height
        width = (x2 - x1) / image_width
        height = (y2 - y1) / image_height

        area = width * height # Area relative to the image size

        # Skip tiny boxes with less than 0.1% of the image area
        # Invalid IDs (= unknown objects) are marked with negative class ID
        if p.class_id >= 0 and area > min_area:
            # For teeth: check mean intensity within the box, only keep i>100
            if p.class_id == 18:
                img_patch=image[x1:x2,y1:y2]
                mean_intensity = img_patch.mean()
                if mean_intensity < 80 and area < 1.5*min_area:
                    logger.debug(f"Discarding box with intensity {mean_intensity} - Area: {area}")

            # Append the result in YOLO format
            result += f"{p.class_id} {x_center} {y_center} {width} {height}\n"

    # If the result contains some prompt for 18, at least 4 prompts have to be present, else discard all
    # Split by lines and check how many 18 are present
    res_split = result.split("\n")
    if len(res_split) > 1:
        count_18 = sum(1 for line in res_split if line.startswith("18 "))
        if count_18 > 0 and count_18 < 4:
            logger.info(f"Discarding all boxes because only {count_18} boxes for class 18 are present.")
            return ""
        
    return result

def split_dataset_indeces(indices, train_ratio=0.8):
    """
    Randomly splits the dataset indices into training and validation sets based on the specified ratio.
    
    Args:
        indices (list): List of dataset indices to be split.
        train_ratio (float): Ratio of training set size to total dataset size.
        
    Returns:
        tuple: Two lists containing the training and validation indices.
    """
    np.random.shuffle(indices)
    split_index = int(len(indices) * train_ratio)
    train_indices = indices[:split_index]
    val_indices = indices[split_index:]
    
    return train_indices, val_indices

def resize_and_normalize(image):
    """
    This function resizes a 3D grayscale numpy array to a 4D numpy array of shape (d, 3, image_size, image_size)
    and normalizes it to the range [0, 1] with ImageNet mean and std.
    The input image is expected to be in the shape (d, h, w).

    Args:
        image (numpy.ndarray): Input 3D grayscale image array of shape (d, h, w).
    Returns:
        torch.Tensor: A 4D tensor of shape (d, 3, image_size, image_size) with normalized pixel values.
    """
    img_resized = transpose_and_resize(image, 512)

    img_resized = img_resized / 255.0
    img_resized = torch.from_numpy(img_resized).cuda()
    img_mean = (0.485, 0.456, 0.406)  # ImageNet mean
    img_std = (0.229, 0.224, 0.225)
    img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None].cuda()
    img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None].cuda()
    img_resized -= img_mean
    img_resized /= img_std
    return img_resized

def transpose_and_resize(array, image_size):
    """
    Resize a 3D grayscale numpy array to a 4D numpy array of shape (d, 3, image_size, image_size).
    """
    d, h, w = array.shape
    array = grayscale_to_rgb(array)  # Ensure the array is in RGB format
    resized_array = np.zeros((d, 3, image_size, image_size), dtype=np.float32)

    for i in range(d):
        img_pil = Image.fromarray(array[i].astype(np.uint8))
        img_rgb = img_pil.convert("RGB")
        img_resized = img_rgb.resize((image_size, image_size))
        img_array = np.array(img_resized).transpose(
            2, 0, 1
        )  # (3, image_size, image_size)
        resized_array[i] = img_array

    return resized_array

def plot_img_gt_pred(img, gt, pred, name, object_id, dsc, nsd):
    # Plot the results for each object
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title(f"{name} - {object_id}")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(gt, cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(pred, cmap="gray")
    plt.title(f"Pred - DSC: {dsc:.3f} - NSD: {nsd:.3f}")
    plt.axis("off")

    # SAve the figure
    plt.savefig(f"res.png", bbox_inches="tight", dpi=300)
    plt.savefig(f"res.pdf", bbox_inches="tight", dpi=300)
    plt.savefig(f"res.svg", bbox_inches="tight", dpi=300)

    plt.show()

if __name__ == "__main__":
    raise NotImplementedError(
        "This module is not meant to be run directly. It is meant to be imported and used in other modules."
    )
