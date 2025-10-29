from PIL import Image
import numpy as np
import os
from collections import defaultdict

from segFM.DataLoaders.base_dataset import BaseImageDataset
from segFM import prompts, color_classes


class EndoscapesDataset(BaseImageDataset):
    """
    Endoscapes Dataset for multi-object segmentation in endoscopic images.
    """
    def __init__(
        self,
        transform=None,
        bbsize=0,
        mode="point",
        prompt_finder="random",
        n_pos=1,
        n_neg=0,
        plot_prompts=False,
    ):
        super(EndoscapesDataset, self).__init__(transform)

        self.image_path = "/media/david/SHARED/endoscapes/all/"
        self.mask_path = "/media/david/SHARED/endoscapes/semseg/"

        self.bbsize = bbsize
        self.mode = mode
        self.prompt_finder = prompt_finder
        self.n_pos = n_pos
        self.n_neg = n_neg
        self.plot_prompts = plot_prompts

        # Not all images have masks, so we use the mask path to get the list of images
        self.mask_list = self.get_img_list(self.mask_path, ".png")
        self.n_images = len(self.mask_list)
        self.color_to_id = {
            0: -1,  # Background
            1: -1,   # Cystic plate
            2: -1,   # Calot triangle
            3: -1,   # Cystic artery
            4: -1,   # Cystic duct
            5: 15,   # Gallbladder
            6: 5,   # Tool
        }

        self.color_to_label = color_classes.endoscapes_color_to_label

    def __getitem__(self, idx):
        """
        Get the image and its corresponding mask for the given index.
        """
        mask_name = self.mask_list[idx]
        mask_path = os.path.join(self.mask_path, mask_name)

        img_name = mask_name.replace(".png", ".jpg")
        img_path = os.path.join(self.image_path, img_name)

        # Load the image and mask
        image = np.array(Image.open(img_path).convert("RGB"))
        gt = np.array(Image.open(mask_path).convert("L"))

        if self.transform:
            image = self.transform(image)

        # Get the bounding box and points for the image
        prompt = prompts.get_multiobject_multicolor_prompt(
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
            "name": img_name,
            "prompts": prompt,
        }
        return data
