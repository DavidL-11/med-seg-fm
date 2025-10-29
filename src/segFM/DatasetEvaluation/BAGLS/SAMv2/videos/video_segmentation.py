from sam2.build_sam import build_sam2_video_predictor
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
from segFM import utils, prompts, checkpoints

USE_MEDSAM2 = False

if USE_MEDSAM2:
    CHECKPOINT = checkpoints.MedSAM2_latest
    CONFIG = checkpoints.MedSAM_cfg
else:
    CHECKPOINT = checkpoints.SAM2_tiny
    CONFIG = checkpoints.SAM2_tiny_cfg

class VideoSegmenter:
    def __init__(self):
        self.device = utils.setup_device()
        self.predictor = build_sam2_video_predictor(CONFIG, CHECKPOINT, device=self.device)
    

    def convert_video_to_frames(self, video_path):
        """Converts a video file into a list of frames and stores them in a temporary directory."""

        video = cv2.VideoCapture(video_path)

        fps = video.get(cv2.CAP_PROP_FPS)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create a new temporary directory to store the frames
        temp_dir = "temp_frames"
        os.makedirs(temp_dir, exist_ok=True)

        count = 0
        sucess, frame = video.read()

        while sucess:
            # Convert the frame to RGB format
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #Save the frame
            frame_path = os.path.join(temp_dir, f"{count:03d}.jpg")
            cv2.imwrite(frame_path, frame)
            count += 1

            sucess, frame = video.read()
            
        print(f"Extracted {count} frames from the video.")

        # Release the video capture object
        video.release()
        return fps, width, height
    
    def show_mask(self, mask, ax, obj_id=None, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            cmap = plt.get_cmap("tab10")
            cmap_idx = 0 if obj_id is None else obj_id
            color = np.array([*cmap(cmap_idx)[:3], 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)
        ax.axis("off")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

        plt.show()

    def show_points(self, coords, labels, ax, marker_size=200):
        pos_points = coords[labels==1]
        neg_points = coords[labels==0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

    def show_box(self, box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

    def segment_video(self, video_path, box, points, labels, mode = "point", show_plot=False):
        """Segments a video into frames and applies the SAM2 model to each frame.

        Args:
            video_path (str): Path to the video file.
        Returns:
            list: A list of segmented frames, where each frame is a numpy array in RGB format.
        """
        # Convert video to frames
        fps, width, height = self.convert_video_to_frames(video_path)
        frame_dir = "temp_frames"
        
        frame_names = [
            p for p in os.listdir(frame_dir)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

        inference_state = self.predictor.init_state(video_path=frame_dir)
        self.predictor.reset_state(inference_state)

        ann_frame_idx = 0  # the frame index we interact with
        ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=points if mode == "point" else None,
            box=box if mode == "box" else None,
            labels=labels if mode == "point" else None,
        )

        if show_plot:
            # show the results on the current (interacted) frame
            plt.figure(figsize=(9, 6))
            plt.title(f"frame {ann_frame_idx}")
            plt.imshow(Image.open(os.path.join(frame_dir, frame_names[ann_frame_idx])))
            self.show_points(points, labels, plt.gca())
            self.show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])

            plt.show()

        masks = []

        # run propagation throughout the video and collect the results in a dict
        video_segments = {}  # video_segments contains the per-frame segmentation results
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
            # Save the segmentation results
            for obj_id, mask in video_segments[out_frame_idx].items():
                mask = mask.astype(np.uint8) * 255
                mask_image = Image.fromarray(mask.squeeze(), mode='L')
                mask_image.save(os.path.join(frame_dir, f"mask_{out_frame_idx:03d}.png"))
                masks.append(mask_image)

        # Delete the temporary frames directory
        for image in os.listdir(frame_dir):
            image_path = os.path.join(frame_dir, image)
            if os.path.isfile(image_path):
                os.remove(image_path)
        os.rmdir(frame_dir)

        # Combine the mask_images into a single video
        video_name = os.path.basename(video_path).split(".")[0]

        output_video_folder = "src/BAGLS/SAMv2/videos/results"
        output_video_path = f"{output_video_folder}/{video_name}_seg_{mode}_{'m2d' if USE_MEDSAM2 else 't'}.mp4"

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        for mask_image in masks:
            mask_image = np.array(mask_image)
            mask_image = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2BGR)
            out.write(mask_image)
        out.release()
        print(f"Segmented video saved at {output_video_path}")

    
    def get_prompts_from_video(self, video_path):
        """Extracts a point prompt from a video file by taking the center of mass of the ground truth mask.
        Args:
            video_path (str): Path to the video file.
        Returns:
            tuple: A tuple containing the x and y coordinates of the point prompt.
        """
        # Load the video
        video = cv2.VideoCapture(video_path.split(".")[0] + "_seg.mp4")

        # Get the first frame
        ret, frame = video.read()
        if not ret:
            print("Failed to read the video")
            return None
        
        # Release the video capture object
        video.release()
        
        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Binarize the frame
        gray_frame[gray_frame < 130] = 0
        gray_frame[gray_frame >= 130] = 255

        #Find the bounding box of the white regions in the segmented image
        box = prompts.box_prompt_from_gt(gray_frame)

        points, labels = prompts.random_prompts_from_binary_gt(gray_frame, npos=1, nneg=5, neg_size=30)

        # Return the point prompt as a numpy array
        return box, points, labels


if __name__ == "__main__":
    segmenter = VideoSegmenter() 

    for i in range(30):
        video_path = f"Datasets/BAGLS/videos/{i}.mp4"
        box, points, labels = segmenter.get_prompts_from_video(video_path)
        segmenter.segment_video(video_path, box, points, labels, mode="point", show_plot=False)