from PIL import Image
import numpy as np
import os
from ultralytics import YOLO
import random

from segFM.DataLoaders.base_dataset import BaseImageDataset
from segFM import prompts
from segFM import color_classes
from segFM.logger import logger
from segFM.BOB.src.BOB.prompt_generator import BOB

from collections import defaultdict

class BAGLS_Images(BaseImageDataset):
    """
    This class is for loading the images from the BAGLS dataset.
    The image, its segmentation masks as well as point/box prompts are returned as a dictionary when accessing the dataset.
    Find out more at [BAGLS.org](https://bagls.org/).
    """

    def __init__(
        self,
        type="test",
        transform=None,
        mode="point",
        prompt_finder="random",
        n_pos=1,
        n_neg=0,
        bbsize=0,
        noisy=False,
        prompt_gen:BOB=None,
    ):
        super(BAGLS_Images, self).__init__(transform=transform)
        self.n_images = 55250 if type == "training" else 3499
        self.training_folder = f"Datasets/BAGLS/{type}/"
        self.test_folder = "Datasets/BAGLS/test/"
        self.mode = mode
        self.prompt_finder = prompt_finder
        self.n_pos = n_pos
        self.n_neg = n_neg
        self.bbsize = bbsize
        self.random_noise = noisy
        self.prompt_gen = prompt_gen
        self.class_to_color = defaultdict(lambda: 255)
        self.color_to_label = defaultdict(lambda: "Glottis")
        self.dataset_name = "BAGLS_Images"

    def __getitem__(self, idx):
        # Check if the index is valid
        if idx < 0 or idx >= self.n_images:
            raise IndexError(f"This dataset only contains {self.n_images} images!")

        # Load the image and segmentation mask
        image = np.array(Image.open(f"{self.training_folder}{idx}.png"))
        gt = np.array(Image.open(f"{self.training_folder}{idx}_seg.png").convert("L"))

        # Transform the image if a transform is provided
        if self.transform:
            image = self.transform(image)

        if self.random_noise:
            # Add random noise to the image
            noise = np.random.normal(0, 0.1, image.shape).astype(np.float32)
            image = np.clip(image + noise * 255, 0, 255).astype(np.uint8)

        if self.prompt_finder == "yolo":
            prompt = self.prompt_gen.generate_prompt(
                img=image,
                id_to_color=self.class_to_color,
                plot_prompts=False,
            )
        else:
            # Get the prompt for the image
            prompt = prompts.get_prompt(
                gt=gt,
                image=image,
                dataset=self,
                bbsize=self.bbsize,
                mode=self.mode,
                prompt_finder=self.prompt_finder,
                n_pos=self.n_pos,
                n_neg=self.n_neg
            )

        # Return the image, segmentation mask, and prompt as a dictionary
        data = {
            "img": image,
            "gt": gt,
            "id": idx,
            "prompts": [prompt],
            "name": f"{idx}.png",
        }
        return data


class BAGLS_Videos(BaseImageDataset):
    """
    This class is for loading the videos from the BAGLS dataset.
    The path to the video, as well as the frames and their segmentation masks are returned as a dictionary.
    The video is loaded using OpenCV and the frames are extracted.
    The segmentation masks are loaded using PIL and converted to numpy arrays.
    Find out more at [BAGLS.org](https://bagls.org/).
    """

    def __init__(self, transform=None):
        self.n_videos = 558

    def __getitem__(self, idx):
        video_path = f"Datasets/BAGLS/videos/{idx}.mp4"
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video {video_path} not found.")
        
class BAGLSImagesFull(BaseImageDataset):
    """
    The full BAGLS image dataset includes ground truths for the glottis as well as the vocal folds.
    """
    def __init__(
        self,
        transform=None,
        mode="box",
        prompt_finder="random",
        n_pos=1,
        n_neg=0,
        bbsize=0,
        noisy=False,
        prompt_gen: BOB = None,
    ):
        super(BAGLSImagesFull, self).__init__(transform=transform)
        self.path = "Datasets/BAGLS_full/"
        self.dataset_name = "BAGLSImagesFull"
        self.img_path = self.path + "images/"
        self.gt_path = self.path + "masks/"
        self.n_images = len(os.listdir(self.img_path))
        self.img_names = self.get_img_names()
        self.mode = mode
        self.prompt_finder = prompt_finder
        self.n_pos = n_pos
        self.n_neg = n_neg
        self.bbsize = bbsize
        self.random_noise = noisy
        self.prompt_gen = prompt_gen
        self.id_to_color = color_classes.bagls_id_to_color
        self.color_to_id = color_classes.bagls_color_to_id
        self.color_to_label = color_classes.bagls_color_to_label

    def get_img_names(self):
        """
        Returns a list of image names in the dataset.
        """
        files = os.listdir(self.img_path)
        return [f for f in files if f.endswith('.png')]

    def __getitem__(self, idx):
        # Check if the index is valid
        if idx < 0 or idx >= self.n_images:
            raise IndexError(f"This dataset only contains {self.n_images} images!")

        # Load the image and segmentation mask
        image_name = self.img_names[idx]
        mask_name = "mask_" + image_name

        image = np.array(Image.open(os.path.join(self.img_path, image_name)))
        gt = np.array(Image.open(os.path.join(self.gt_path, mask_name)))

        gt = gt.transpose(2, 0, 1)

        # Set all 1s in each channel to the channel's index
        for i in range(gt.shape[0]):
            gt[i, :, :] = np.where(
                gt[i, :, :] > 0, i + 1, 0
            )  # Set the channel index as the color value

        # Transform the image if a transform is provided
        if self.transform:
            image = self.transform(image)

        if "BOB" in self.mode:
            self.prompt_finder = self.mode
            prompt = self.prompt_gen.generate_prompt(
                img=image,
                dataset=self,
                plot_prompts=False,
                confidence=0.5,
                allowed_classes=[0, 1, 2]
            )
        else:
            # Get the prompt for the image
            prompt = prompts.get_labelmap_prompt(
                gt=gt,
                image=image,
                dataset=self,
                bbsize=self.bbsize,
                mode=self.mode,
                prompt_finder=self.prompt_finder,
                n_pos=self.n_pos,
                n_neg=self.n_neg,
                plot_prompts=False,
            )

        # Return the image, segmentation mask, and prompt as a dictionary
        data = {
            "img": image,
            "gt": gt,
            "id": idx,
            "prompts": prompt,
            "name": image_name,
        }
        return data