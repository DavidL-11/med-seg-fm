from segFM.DataLoaders.base_dataset import BaseImageDataset
from segFM import utils, prompts
from segFM import color_classes

import numpy as np

class Amos22(BaseImageDataset):
    """
    Amos22 Dataset for multi-organ segmentation.
    """

    def __init__(
        self,
        transform=None,
        mode="box",
        prompt_gen=None,
        plot_prompts=False,
        split="val",
        preprocess="auto",
    ):
        super(Amos22, self).__init__(transform=transform)
        if split == "train":
            self.imgs_path = "/media/david/SSD1TB/Datasets/amos22/imagesTr/"
            self.labels_path = "/media/david/SSD1TB/Datasets/amos22/labelsTr/"
            self.dataset_name = "Amos22_train"
        elif split == "val":
            self.imgs_path = "/media/david/SSD1TB/Datasets/amos22/imagesVa/"
            self.labels_path = "/media/david/SSD1TB/Datasets/amos22/labelsVa/"
            self.dataset_name = "Amos22_test"
        else:
            raise ValueError("Mode must be 'train' or 'val'.")
        self.img_list = self.get_img_list(self.imgs_path, ".nii.gz")
        self.n_images = len(self.img_list)
        self.plot_prompts = plot_prompts
        self.id_to_color = color_classes.amos22_id_to_color
        self.color_to_id = color_classes.amos22_color_to_id
        self.color_to_label = color_classes.amos22_color_to_label
        self.prompt_gen = prompt_gen
        self.mode = mode
        self.preprocess = preprocess

    def __getitem__(self, idx):
        # Check if the index is valid
        if idx < 0 or idx > self.n_images:
            raise IndexError(
                f"This dataset only contains [1, {self.n_images + 1}] images. Index {idx} is out of range."
            )

        img_name = self.img_list[idx]

        # Load the image
        # Check if image is CT or MRI and preprocess accordingly
        # Images with name amos_0XXX are MRI if XXX is larger than 500
        if self.preprocess.lower() == "auto":
            if int(img_name.split("_")[-1].split(".")[0]) > 500:
                preprocess = "Abdomen MRI"
            else:
                preprocess = "Liver CT"
        elif self.preprocess.lower() == "random":
            if int(img_name.split("_")[-1].split(".")[0]) > 500:
                preprocess = np.random.choice(["Abdomen MRI", "None"], p=[0.5, 0.5])
            else:
                preprocess = np.random.choice(["Liver CT", "Abdomen CT", "None"], p=[0.2, 0.4, 0.4])
        else:
            preprocess = self.preprocess

        img = utils.nifti_to_numpy2(self.imgs_path + img_name, preprocess=preprocess)
        gt = utils.nifti_to_numpy2(self.labels_path + img_name, preprocess="None")

        if "BOB" in self.mode:
            prompt = self.prompt_gen.generate_prompt(
                img,
                dataset=self,
                confidence=0.4,
                iou=0.7,
                n_prompts_per_obj=5,
                multiprompt_z_spacing=5,
                max_z_distance=20,
                allow_multiobject_3d=False,
                allowed_classes=None,
            )
        else:
            prompt = prompts.multicolor_box_prompt_3d(
                gt, dataset=self, plot_prompt=self.plot_prompts, img=img
            )

        if self.transform:
            img, gt = self.transform(
                img, gt
            )

        data = {
            "img": img,
            "gt": gt,
            "name": img_name,
            "gt_name": img_name,
            "id": idx,
            "prompts": prompt,
        }

        return data
