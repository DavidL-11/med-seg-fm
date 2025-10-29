import os, cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from surface_distance.metrics import compute_dice_coefficient

RESULT_FOLDER = "src/BAGLS/SAMv2/videos/results"
GT_FOLDER = "Datasets/BAGLS/videos"
MEDSAM = False

def dice_score(img, seg) -> float:
    """
    Computes the dice score between two binary images.
    The dice score is a measure of overlap between two sets.
    It is represented by a value between 0 and 1, where 1 indicates perfect overlap.
    This can be used to compare a predicted segmentation with a ground truth segmentation.
    """

    overlap = np.count_nonzero(img * seg)
    union = np.count_nonzero(img) + np.count_nonzero(seg)

    #No overlap is still considered a perfect score if all pixels are 0
    if union == 0:
        return 1.0

    return (2.0 * overlap + 1e-6) / (union + 1e-6) # to avoid possible division by zero

def intersection_over_union(seg_mask, gt) -> float:
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
    # Conver to PIL Image



    #Calculate the intersection and union
    intersection = np.logical_and(seg_mask, gt)
    union = np.logical_or(seg_mask, gt)

    #Calculate the IoU
    iou = np.sum(intersection) / np.sum(union)

    return iou

compare_box = []
compare_point = []

files_names = [
    p for p in os.listdir(RESULT_FOLDER)
    if p.endswith(".mp4") and ("m2d" if MEDSAM else "_t") in p
]
files_names.sort(key=lambda x: int(x.split("_")[0]))

# Read src/BAGLS/SAMv2/videos/*.mp4
for filename in files_names:
    gt_name = filename.split("_")[0] + "_seg.mp4"
    gt_path = os.path.join(GT_FOLDER, gt_name)

    if "box" in filename:
        compare_box.append((os.path.join(RESULT_FOLDER, filename), gt_path))
    elif "point" in filename:
        compare_point.append((os.path.join(RESULT_FOLDER, filename), gt_path))

# Iterate over the gts and compare each frame with the corresponding frame in the video and evaluate its performance using the DSC and IOU metrics
dsc_overall = []
iou_overall = []

for seg, gt in compare_point:
    # Read the video
    video = cv2.VideoCapture(seg)
    gt_video = cv2.VideoCapture(gt)

    # Get the number of frames
    n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Evaluating video {seg}...")
    single_video_dsc = []
    single_video_iou = []

    # Iterate over the frames
    for i in range(n_frames):
        ret, frame = video.read()
        ret, gt_frame = gt_video.read()

        if not ret:
            break

        frame[frame < 130] = 0
        frame[frame >= 130] = 255
        gt_frame[gt_frame < 130] = 0
        gt_frame[gt_frame >= 130] = 255

        # Convert to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gt_frame = cv2.cvtColor(gt_frame, cv2.COLOR_BGR2GRAY)

        # Calculate DSC and IOU
        dsc = dice_score(frame, gt_frame)
        dsc2 = compute_dice_coefficient(frame, gt_frame)
        iou = intersection_over_union(frame, gt_frame)

        single_video_dsc.append(dsc)
        single_video_iou.append(iou)

        # if dsc < 0.5:
        #     # Save both frames
        #     cv2.imwrite(f"{i}_seg.png", frame)
        #     cv2.imwrite(f"{i}_gt.png", gt_frame)

        #     # Create an rgb image with frame as red and gt_frame as green
        #     rgb_frame = cv2.merge((frame, gt_frame, np.zeros_like(frame)))
        #     # Show the frame
        #     plt.imshow(rgb_frame)
        #     plt.title(f"Frame {i} - DSC: {dsc:.4f} - IOU: {iou:.4f}")
        #     plt.axis("off")
        #     plt.show()
    
    print(f"Average DSC: {np.mean(single_video_dsc):.4f}")
    print(f"Average IOU: {np.mean(single_video_iou):.4f}")
    print("#######################################################")
    dsc_overall.append(single_video_dsc)
    iou_overall.append(single_video_iou)

    video.release()
    gt_video.release()

result = {
    "dsc": dsc_overall,
    "iou": iou_overall
}

print(f"{'MedSAM2' if MEDSAM else 'SAMv2'} Evaluation Results")
print(f"Overall Average DSC: {np.mean(dsc_overall):.3f}")
print(f"Overall Average IOU: {np.mean(iou_overall):.3f}")
print(f"Overall Median DSC: {np.median([np.median(x) for x in dsc_overall]):.3f}")
print(f"Overall Median IOU: {np.median([np.median(x) for x in iou_overall]):.3f}")


