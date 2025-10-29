from PIL import Image
import numpy as np
import os
import cv2
from collections import defaultdict

from segFM.DataLoaders.base_dataset import BaseImageDataset
from segFM import prompts, utils


class NeopolypDataset(BaseImageDataset):
    """
    Neopolyp Dataset for polyp segmentation in endoscopic images.
    """
    def __init__(
        self,
        transform=None,
        bbsize=0,
        mode="box",
        prompt_finder="random",
        n_pos=2,
        n_neg=3,
        plot_prompts=False,
        prompt_gen=None,
        split="train",
    ):
        super(NeopolypDataset, self).__init__(transform)
        self.dataset_name = "NEOPOLYP"

        if split == "train":
            self.image_path = "Datasets/bkai-igh-neopolyp/train/train/"
            self.mask_path = "Datasets/bkai-igh-neopolyp/train_gt/train_gt/"
        else:
            self.image_path = "Datasets/bkai-igh-neopolyp/bob_test/"
            self.mask_path = "Datasets/bkai-igh-neopolyp/bob_test_gt/"

        self.bbsize = bbsize
        self.mode = mode
        self.prompt_finder = prompt_finder
        self.n_pos = n_pos
        self.n_neg = n_neg
        self.plot_prompts = plot_prompts
        self.prompt_gen = prompt_gen

        # Not all images have masks, so we use the mask path to get the list of images
        self.image_list = self.get_img_list(self.image_path, ".jpeg")
        self.n_images = len(self.image_list)
        self.id_to_color = {
            4: 255,
        }
        self.color_to_id = {
            1: 4,
            255: 4,
        }
        self.color_to_label = {
            255: "Polyp",
            1: "Polyp",
        }

    def __getitem__(self, idx):
        """
        Get the image and its corresponding mask for the given index.
        """
        name = self.image_list[idx]
        mask_path = os.path.join(self.mask_path, name)
        img_path = os.path.join(self.image_path, name)

        # Load the image and mask
        image = np.array(Image.open(img_path).convert("RGB"))
        gt = np.array(Image.open(mask_path).convert("L"))

        # Ensure the ground truth mask is binary
        gt = (gt > 60).astype(np.uint8) * 255

        # Perform binary erosion on the ground truth mask to remove small jpeg artifacts
        gt = cv2.erode(gt, np.ones((3, 3)), iterations=1)

        if self.transform:
            image = self.transform(image)

        if "BOB" in self.mode:
            prompt = self.prompt_gen.generate_prompt(image, dataset=self, confidence=0.1)
        else:
            # Get the bounding box and points for the image
            prompt = prompts.get_multiobject_prompt(
                gt,
                image,
                dataset=self,
                bbsize=self.bbsize,
                mode=self.mode,
                prompt_finder=self.prompt_finder,
                n_pos=self.n_pos,
                n_neg=self.n_neg,
                plot_prompts=self.plot_prompts,
            )

        data = {
            "img": image,
            "gt": gt,
            "id": idx,
            "name": name,
            "prompts": prompt,
        }
        return data
