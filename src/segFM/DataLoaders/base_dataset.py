from torch.utils.data import Dataset
import numpy as np
import random
import os

from segFM import utils
from segFM.logger import logger

class BaseImageDataset(Dataset):
    """
    Base class for all datasets.
    Specific datasets should inherit from this class and implement the __getitem__ method.
    """

    def __init__(self, transform=None):
        self.transform = transform
        self.n_images = 0
        # Name is class name
        self.name = self.__class__.__name__
        np.random.seed(11)
        random.seed(11)

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        raise NotImplementedError("Subclasses should implement this method.")

    def get_random_image(self):
        """
        Returns a random image from the dataset.
        """
        idx = np.random.randint(1, self.n_images)
        return self.__getitem__(idx)

    def get_n_nonduplicate_indeces(self, n_images):
        """
        Get n non-duplicate indices from the dataset.
        """
        if n_images > self.n_images:
            n_images = self.n_images
            logger.warning(
                f"Requested more images than available. Reducing n_images to {self.n_images}."
            )

        # Get n random indices
        indices = random.sample(range(self.n_images), n_images)

        return indices

    def get_n_nonduplicate_images(self, n_images):
        """
        Get n non-duplicate images from the dataset.
        Each item is a dictionary containing the image, segmentation mask, and prompts.
        """
        if n_images > self.n_images:
            n_images = self.n_images
            logger.warning(
                f"Requested more images than available. Reducing n_images to {self.n_images}."
            )

        # Get n random indices
        indices = self.get_n_nonduplicate_indeces(n_images)
        images = []

        logger.info(f"Loading images and preparing prompts for {n_images} images...")
        for i, idx in enumerate(indices):
            utils.printProgressBar(i + 1, n_images, prefix='Progress:', suffix='Complete', length=70)
            images.append(self.__getitem__(idx))

        return images

    def get_img_list(self, path, filetype, n_images=-1):
        """
        Get the list of nifti images in the given path.
        If n_images is specified, only that many images will be returned.
        """
        img_list = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(filetype) and not file.startswith("."):
                    if n_images != -1 and len(img_list) >= n_images:
                        break
                    img_list.append(file)

        return img_list
    
    def cleanup_temp_dir(self, temp_dir):
        """
        Cleans up the temporary directory by removing all files and the directory itself.
        """
        if not os.path.exists(temp_dir):
            return
        
        # Remove all files in the temporary directory
        for image in os.listdir(temp_dir):
            image_path = os.path.join(temp_dir, image)
            if os.path.isfile(image_path):
                os.remove(image_path)
        os.rmdir(temp_dir)


class BaseVideoDataset(Dataset):
    """
    Base class for all datasets.
    """

    def __init__(self, transform=None):
        self.transform = transform
        self.n_videos = 0

    def __len__(self):
        return self.n_videos

    def __getitem__(self, idx):
        raise NotImplementedError("Subclasses should implement this method.")

    def get_random_video(self):
        """
        Returns a random image from the dataset.
        """
        idx = np.random.randint(1, self.n_videos)
        return self.__getitem__(idx)

    def get_n_nonduplicate_indeces(self, n_videos):
        """
        Get n non-duplicate indices from the dataset.
        """
        if n_videos > self.n_videos:
            raise ValueError(
                f"Cannot get {n_videos} non-duplicate indices from a dataset of size {self.n_videos}."
            )

        # Get n random indices
        indices = random.sample(range(self.n_videos), n_videos)

        return indices

    def get_n_nonduplicate_videos(self, n_videos):
        """
        Get n non-duplicate videos from the dataset.
        Each item is a dictionary containing the video frames, segmentation masks, and prompts.
        """
        if n_videos > self.n_videos:
            raise ValueError(
                f"Cannot get {n_videos} non-duplicate videos from a dataset of size {self.n_videos}."
            )

        # Get n random indices
        indices = self.get_n_nonduplicate_indeces(n_videos)
        videos = []

        logger.info(f"Loading videos and preparing prompts for {n_videos} videos...")
        for i, idx in enumerate(indices):
            utils.printProgressBar(i + 1, n_videos, prefix='Progress:', suffix='Complete', length=70)
            videos.append(self.__getitem__(idx))
        
        return videos

    def cleanup_temp_dir(self, temp_dir):
        """
        Cleans up the temporary directory by removing all files and the directory itself.
        """
        for image in os.listdir(temp_dir):
            image_path = os.path.join(temp_dir, image)
            if os.path.isfile(image_path):
                os.remove(image_path)
        os.rmdir(temp_dir)

    def compute_metrics(self, gt_masks, pred_masks):
        """
        Computes the metrics for the predicted masks against the ground truth masks.
        """
        dscs = []
        for gt, pred in zip(gt_masks, pred_masks):
            if gt.sum() == 0 and pred.sum() == 0:
                dsc = 1.0
            else:
                dsc = utils.compute_dice_coefficient(gt, pred)
            dscs.append(dsc)
        dscs = np.array(dscs)
        print(f"Mean Dice Coefficient: {np.mean(dscs)}")
        return np.mean(dscs)
