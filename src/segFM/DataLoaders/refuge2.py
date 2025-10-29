from PIL import Image
import numpy as np
import os
import cv2
from ultralytics import YOLO
from collections import defaultdict

from segFM.DataLoaders.base_dataset import BaseImageDataset
from segFM import prompts, utils


class Refuge2Dataset(BaseImageDataset):
    """
    Refuge2 Dataset for optic disc and cup segmentation.
    """
    def __init__(
        self,
        transform=None,
        bbsize=0,
        mode="box",
        prompt_finder="random",
        n_pos=1,
        n_neg=0,
        plot_prompts=False,
        prompt_gen=None,
        split="train",
    ):
        super(Refuge2Dataset, self).__init__(transform)

        self.dataset_name = "REFUGE2"
        self.image_path = f"Datasets/REFUGE2/{split}/images/"
        self.mask_path = f"Datasets/REFUGE2/{split}/mask/"

        self.bbsize = bbsize
        self.mode = mode
        self.prompt_finder = prompt_finder
        self.n_pos = n_pos
        self.n_neg = n_neg
        self.plot_prompts = plot_prompts

        # Not all images have masks, so we use the mask path to get the list of images
        self.image_list = self.get_img_list(self.image_path, ".jpg")
        self.n_images = len(self.image_list)
        self.color_to_id = {
            255: 24, # Optic cup
            128: 23, # Optic disc
            0: -1  # Background
        }
        self.id_to_color = {
            24: 255,
            23: 128,
        }
        self.color_to_label = {
            0: "background",
            128: "optic_disc",
            255: "optic_cup"
        }
        self.prompt_gen = prompt_gen

    def __getitem__(self, idx):
        """
        Get the image and its corresponding mask for the given index.
        """
        name = self.image_list[idx]
        img_path = os.path.join(self.image_path, name)
        gt_name = name.replace(".jpg", ".png" if "V" in name else ".bmp")
        mask_path = os.path.join(self.mask_path, gt_name)  # GT is Bitmap

        # Load the image and mask
        image = np.array(Image.open(img_path).convert("RGB"))
        gt = np.array(Image.open(mask_path).convert("L"))

        # Invert the GT colors (255->0, 0->255, 128->128)
        new_gt = np.zeros_like(gt)
        new_gt[gt == 255] = 0
        new_gt[gt == 0] = 255
        new_gt[gt == 128] = 128
        gt = new_gt

        if self.transform:
            image = self.transform(image)

        if "BOB" in self.mode:
            prompt = self.prompt_gen.generate_prompt(
                image,
                dataset=self,
                confidence=0.5
            )
        else:
            # Get the bounding box and points for the image
            prompt = prompts.get_multicolor_prompt(
                gt=gt,
                image=image,
                dataset=self,
                mode=self.mode,
                bbsize=self.bbsize,
                prompt_finder=self.prompt_finder,
                n_pos=self.n_pos,
                n_neg=self.n_neg,
                plot_prompts=self.plot_prompts
            )

        data = {
            "img": image,
            "gt": gt,
            "id": idx,
            "name": name,
            "prompts": prompt,
        }
        return data

if __name__ == "__main__":
    dataset = Refuge2Dataset(plot_prompts=True)
    for i in range(len(dataset)):
        data = dataset[i]
        print(f"Image: {data['name']}, Prompts: {data['prompts']}")
        # Optionally visualize the image and prompts here