import numpy as np
import random
import os
from collections import defaultdict

from segFM.DataLoaders.base_dataset import BaseImageDataset
from segFM import utils, prompts
from segFM.BOB.src.BOB.prompt_generator import BOB

class BratsMSDGlioma(BaseImageDataset):
    """
    This class is for loading the images from the Medical Decathlon dataset.
    The image, its segmentation masks as well as point/box prompts are returned as a dictionary when accessing the dataset.
    Find out more at [Medical Decathlon](http://medicaldecathlon.com/).
    """

    def __init__(self, yolo=False, transform=None, modality=None):
        """
        Initializes the BRATS dataset.

        Args:
            transform: Transform that is applied to image and ground truth
            modality (int): Which channel/modality of the multi-channel dataset should be kept (FLAIR, T1w, T1gd, T2w)
        """

        self.yolo = yolo
        super(BratsMSDGlioma, self).__init__(transform=transform)

        self.img_path, self.label_path = self.get_folder_path()
        self.img_list = self.get_img_list(self.img_path, ".nii.gz")
        self.n_images = len(self.img_list)
        self.prompt_gen = BOB()
        self.color_to_id = defaultdict(lambda: 22)
        self.color_to_label = defaultdict(lambda: "Glioma")
        self.id_to_color = defaultdict(lambda: 255)
        self.yolo = yolo
        self.modality = modality

    def biased_random(self):
        # Assign equal weights to 0, 2, 3
        # Assign smaller weight to 1 (T1W) since it's not very useful for tumor detection
        weights = [3, 1, 3, 3]  # corresponds to [0,1,2,3]
        return random.choices([0, 1, 2, 3], weights=weights, k=1)[0]

    def __getitem__(self, idx):
        """
        Get the image, label, and prompt for the given index.
        """
        img_name = self.img_list[idx]
        img_path = os.path.join(self.img_path, img_name)
        gt_pth = os.path.join(self.label_path, img_name)

        # Load the image and label
        img = utils.nifti_to_numpy(img_path, remove_4d_dim=False)
        gt = utils.nifti_to_numpy(gt_pth)

        # Binarize the GT to only contain "Glioma" and "Not Glioma" instead of the 3 classes edema, non-enhancing/enhancing tumor
        gt[gt > 1] = 255
        gt[gt <= 1] = 0

        if self.transform:
            img = self.transform(img)

        # Get a random modality with P(1) < P(0) = P(2) = P(3)
        if self.modality is None:
            modality = self.biased_random()

            # Take a random modality from the 4 available ones (FLAIR, T1w, T1gd, T2w)
            img_slice = img[modality]
             # Skip empty/dark slices
            while img_slice.sum() < 5555555:
                modality = self.biased_random()
                img_slice = img[modality]
        else:
            modality = self.modality
            img_slice = img[modality]  # Random slice

        prompt = prompts.multicolor_box_prompt_3d(gt, dataset=self, plot_prompt=False)

        data = {
            # Return a random modality for each sample
            "img": img_slice,
            "name": f"M{modality}_" + img_name,
            "gt": gt,
            "prompts": prompt,
        }
        
        return data

    def get_folder_path(self):
        """
        Get the folder path based on the mode.
        """
        img_path = "Datasets/MedicalDecathlon/Task01_BrainTumour/imagesTr"
        label_path = "Datasets/MedicalDecathlon/Task01_BrainTumour/labelsTr"

        return img_path, label_path
