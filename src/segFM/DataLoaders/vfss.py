from PIL import Image
import pandas as pd
import numpy as np
import os
from collections import defaultdict

from segFM.DataLoaders.base_dataset import BaseImageDataset, BaseVideoDataset
from segFM import prompts


class VFSSImageDataset(BaseImageDataset):
    """
    This class is for loading the images from the VFSS-Segmentation dataset.
    The image, its segmentation masks as well as point/box prompts are returned
    as a dictionary when accessing the dataset.
    """

    def __init__(self, transform=None):
        super(VFSSImageDataset, self).__init__(transform=transform)
        self.img_path = "Datasets/VFSS/imgs"
        self.label_path = "Datasets/VFSS/masks"
        self.n_images = 6400  # Total number of images in the dataset

    def __getitem__(self, idx):
        """
        Get the image, label, and prompt for the given index.
        """
        if idx < 320 or idx >= self.n_images:
            raise IndexError(
                f"This dataset only contains images from index 320 to {self.n_images - 1}. Index {idx} is out of range."
            )


class VFSSVideoDataset(BaseVideoDataset):
    """
    This class is for loading the videos from the VFSS-Segmentation dataset.
    The video, its segmentation masks as well as point/box prompts are returned as a dictionary of lists when accessing the dataset.
    """

    def __init__(self, transform=None, n_pos=1, nneg=1):
        super(VFSSVideoDataset, self).__init__(transform=transform)
        self.img_path = "Datasets/VFSS/imgs"
        self.label_path = "Datasets/VFSS/masks"
        self.videos, self.video_names = self.retrieve_video_frames()
        self.n_videos = len(self.videos)
        self.n_pos = n_pos  # Number of positive prompts
        self.nneg = nneg  # Number of negative prompts
        self.color_to_label = {1: "Bolus", 255: "Bolus"}


    def __getitem__(self, idx):
        """
        Get the video frames, label, and prompt for the given index.
        """
        if idx < 0 or idx >= self.n_videos:
            raise IndexError(
                f"This dataset only contains {self.n_videos} videos. Index {idx} is out of range."
            )

        video_name = self.video_names[idx]
        frames = self.videos[idx]

        prmpts = []

        # Move frames to a temporary directory for processing
        temp_dir = self.move_frames_to_temp_dir(self.img_path, video_name, frames)

        # Load the ground truth masks for the frames
        gt_masks = self.load_ground_truth_masks(frames)

        for i, frame in enumerate(frames):
            prompt = prompts.random_prompts_from_binary_gt(gt=gt_masks[i], dataset=self, npos=self.n_pos, nneg=self.nneg, neg_size=50)
            prmpts.append(prompt)

        # Create a dictionary to hold the data
        data = {
            "video_name": video_name,
            "temp_dir": temp_dir,
            "frames": frames,
            "gt": gt_masks,
            "prompts": prmpts,
        }
        return data

    def retrieve_video_frames(self):
        """
        Returns a dictionary with video names as keys and lists of frame indices as values.
        The function reads the CSV file located at "Datasets/VFSS/data_overview.csv",
        which contains information about the video frames.
        """

        info_file = "Datasets/VFSS/data_overview.csv"

        # Read the CSV file
        df = pd.read_csv(info_file)

        # Group all items that have the same video_name
        grouped = df.groupby("video_name")

        videos = []
        video_names = []

        for video_name, group in grouped:
            frames = group["frame_idx"].tolist()
            videos.append(frames)
            video_names.append(video_name)

        return videos, video_names

    def move_frames_to_temp_dir(self, frame_dir, video_name, frames):
        """
        Moves the frames of a video to a temporary directory for processing.
        """
        temp_dir = f"Datasets/VFSS/frames/{video_name.replace('.mp4', '')}"

        os.makedirs(temp_dir, exist_ok=True)
        for frame in frames:
            # Move the frame to the new directory
            src = f"{frame_dir}/{frame}.png"
            dst = os.path.join(temp_dir, f"{frame}.jpg")

            # Save the image as jpg
            img = Image.open(src)
            img = img.convert("RGB")  # Convert to RGB if not already
            img.save(dst, "JPEG")

        return temp_dir

    def load_ground_truth_masks(self, frame_names):
        """
        Loads the ground truth masks from the specified path and returns a list of masks.
        """
        gt_masks = []

        for frame in frame_names:
            # Load the ground truth mask
            img = Image.open(f"{self.label_path}/{frame}_bolus.png")
            img = img.convert("L")
            gt = np.array(img)
            gt[gt < 130] = 0
            gt[gt >= 130] = 255
            gt_masks.append(gt)

        gt_masks = np.array(gt_masks)
        return gt_masks
