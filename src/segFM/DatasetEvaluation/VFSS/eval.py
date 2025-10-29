from sam2.build_sam import build_sam2_video_predictor
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np
import cv2
import os

from segFM.DataLoaders.vfss import VFSSVideoDataset
from segFM import checkpoints, prompts, utils


def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    ax.imshow(mask_image)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")

    plt.show()

def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def frames_to_video(frames, orig_video_name, fps=15):
    """
    Converts a list of frames to a video file.
    """
    
    # Combine the mask_images into a single video
    video_name = os.path.basename(orig_video_name).split(".")[0]

    output_video_folder = f"{os.path.dirname(__file__)}/videos"
    output_video_path = f"{output_video_folder}/{video_name}_seg_{"MS2" if USE_MEDSAM2 else "SAM2"}.mp4"

    height, width, _ = frames[0].shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    for mask_image in frames:
        mask_image = np.array(mask_image)
        out.write(mask_image)
    out.release()
    print(f"Segmented video saved at {output_video_path}")

def convert_masks(masks, gt_masks, temp_dir, orig_video_name, first_frame_idx=0):
    """
    Converts the masks obtained by the model into a list of overlayed images (Depending on 
    TP/FP/FN) by using the original image, the GT and the predicted masks.
    The function takes the masks, the temporary directory where the original images are stored, and the original video name.
    It returns a list of overlayed images and a list of binary masks.
    """
    overlays = []
    binary_masks = []
    orig_image = np.array(Image.open(f"{temp_dir}/{orig_video_name}.jpg"))

    for out_frame_idx in range(first_frame_idx, len(masks)):
        for out_obj_id, out_mask in masks[out_frame_idx].items():
            h, w = out_mask.shape[-2:]
            out_mask = out_mask.reshape(h, w)
            mask = out_mask.astype(np.uint8) * 255

            binary_masks.append(mask)

            # Overlay the mask on the original image
            overlayed_image = orig_image.copy()

            # Get the ground truth mask for the current frame
            gt_mask = gt_masks[out_frame_idx]

            # TP are Green, FP are Red, FN are Blue
            tp = (mask == 255) & (gt_mask == 255)
            fp = (mask == 255) & (gt_mask == 0)
            fn = (mask == 0) & (gt_mask == 255)

            overlayed_image[tp] = [0, 255, 0]       # True Positive - Green
            overlayed_image[fp] = [255, 0, 0]       # False Positive - Red
            overlayed_image[fn] = [0, 0, 255]       # False Negative - Blue
            overlays.append(overlayed_image)

    return overlays, binary_masks

def segment_frames(predictor, medsam, n_videos=1, render_first_frame=False, render_results=False, save_video=True, save_frame_plot=False, prompt_every_n_frames=5, n_pos=3, nneg=6):
    """
    Segments frames from the VFSS dataset using the provided predictor.
    Parameters:
    - predictor: The video segmentation model predictor.
    - medsam: Boolean indicating if MedSAM2 is used.
    - n_videos: Number of videos to process.
    - render_first_frame: Whether to plot the first frame with prompts.
    - render_results: Whether to plot the segmentation results.
    - save_video: Whether to save the segmented video as a .mp4 file.
    - save_frame_plot: Whether to show a plot with a few selected frames and their segmentation results at the end.
    - prompt_every_n_frames: Interval of frames to provide prompts.
    - n_pos: Number of positive prompts.
    - nneg: Number of negative prompts.
    Returns:
    - df: A pandas DataFrame containing the evaluation metrics for each video.
    """
    
    dataset = VFSSVideoDataset(n_pos=n_pos, nneg=nneg)

    df = pd.DataFrame(columns=["Model", "prompt_step", "n_pos", "n_neg", "DSC", "IoU", "NSD"])

    import random
    random.seed(11)
    np.random.seed(11)

    # Move the video frames to the new directory
    for data in dataset.get_n_nonduplicate_videos(n_videos):
        # Extract the data from the dictionary
        video_name = data['video_name']
        temp_dir = data['temp_dir']
        frames = data['frames']
        gt_masks = data['gt']
        prompts = data['prompts']

        print(f"Processing video: {video_name}")

        inference_state = predictor.init_state(video_path=temp_dir)
        predictor.reset_state(inference_state)

        # The first frame that receives a prompt
        first_annotated_frame = 1 
        
        # Generate prompts ever n frames
        for i in np.arange(start=first_annotated_frame, stop=len(frames), step=prompt_every_n_frames):
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=i,
                obj_id=1,
                points=prompts[i].point,
                labels=prompts[i].label,
            )

            if render_first_frame:
                # Show the prompt points and result on the first frame
                plt.figure(figsize=(9, 6))
                plt.title(f"frame {i}")
                plt.imshow(Image.open(f"{temp_dir}/{frames[i]}.jpg"))
                show_points(prompts[i].point, prompts[i].label, plt.gca())
                show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])

                plt.show()

        # run propagation throughout the video and collect the results in a dict
        video_segments = {}  # video_segments contains the per-frame segmentation results
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        overlays, binary_masks = convert_masks(masks=video_segments, gt_masks=gt_masks, temp_dir=temp_dir, orig_video_name=frames[out_frame_idx], first_frame_idx=first_annotated_frame)

        if save_frame_plot:
            # show a few selected segmentation results at the end
            plt.close("all")
            n_plots = 6
            frame_stride = max(1, len(overlays) // (n_plots - 1))
            plot_ids = list(range(0, len(overlays) - 1, frame_stride))
            
            plt.figure(figsize=(2.5 * n_plots, 4))
            for plot_idx, out_frame_idx in enumerate(plot_ids):
                plt.subplot(1, n_plots, plot_idx + 1)
                plt.title(f"frame {out_frame_idx} segmentation")
                plt.imshow(overlays[out_frame_idx])
                plt.axis('off')

            plt.tight_layout()
            plt.savefig(f"{os.path.dirname(__file__)}/examples/{video_name}_{"MS2" if medsam else "SAM2"}.png", dpi=300, bbox_inches='tight')
            plt.savefig(f"{os.path.dirname(__file__)}/examples/{video_name}_{"MS2" if medsam else "SAM2"}.pdf", dpi=300, bbox_inches='tight')
            plt.savefig(f"{os.path.dirname(__file__)}/examples/{video_name}_{"MS2" if medsam else "SAM2"}.svg", dpi=300, bbox_inches='tight')
            plt.show()
            
        if render_results:
            # render the segmentation results every few frames
            vis_frame_stride = 5
            plt.close("all")
            for out_frame_idx in range(first_annotated_frame, len(frames), vis_frame_stride):
                plt.figure(figsize=(6, 4))
                plt.title(f"frame {out_frame_idx}")

                original_frame = Image.open(f"{temp_dir}/{frames[out_frame_idx]}.jpg")

                plt.imshow(original_frame)

                for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                    show_mask(out_mask, plt.gca(), obj_id=out_obj_id)

        # Convert the masks to a video
        if save_video:
            frames_to_video(overlays, video_name)

        # Delete the temporary frames directory
        dataset.cleanup_temp_dir(temp_dir)

        dscs, ious, nsds = [], [], []
        # Compute metrics
        for gt, pred in zip(gt_masks[first_annotated_frame:], binary_masks):
            assert gt.shape == pred.shape, f"Ground truth and prediction shapes do not match: {gt.shape} vs {pred.shape}"
            dsc, iou, nsd = utils.compute_metrics(seg=pred, gt=gt)
            dscs.append(dsc)
            ious.append(iou)
            nsds.append(nsd)

        new_df = pd.DataFrame({
            "Model": ["MedSAM2" if medsam else "SAM2"],
            "prompt_step": [prompt_every_n_frames],
            "n_pos": [n_pos],
            "n_neg": [nneg],
            "DSC": [np.mean(dscs)],
            "IoU": [np.mean(ious)],
            "NSD": [np.mean(nsds)]
        })

        print(new_df)

        df = pd.concat([df, new_df], ignore_index=True)

    print(df)

    return df


if __name__ == "__main__":
    device = utils.setup_device()

    USE_MEDSAM2 = False

    if USE_MEDSAM2:
        CHECKPOINT = checkpoints.MedSAM2_latest
        CONFIG = checkpoints.MedSAM_cfg
    else:
        CHECKPOINT = checkpoints.SAM2_tiny
        CONFIG = checkpoints.SAM2_tiny_cfg

    

    predictor = build_sam2_video_predictor(CONFIG, CHECKPOINT, device=device, vos_optimized=USE_MEDSAM2)

    #for n_pos in [1, 2, 3, 4, 5, 7, 10]:
    #    for nneg in [0, 1, 2, 3, 4, 5, 7, 10]:

    #for prompt_step in [1, 2, 3, 4, 5, 10, 15, 30]:
    df = segment_frames(
        predictor, 
        medsam=USE_MEDSAM2,
        n_videos=5, 
        render_first_frame=False, 
        render_results=False, 
        save_video=True, 
        save_frame_plot=True,
        prompt_every_n_frames=5,
        n_pos=8, 
        nneg=4
    )

    # utils.save_dataframe_to_csv(__file__, df, "results/vfss_promptstep.csv")