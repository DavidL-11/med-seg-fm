import numpy as np
import os
from collections import defaultdict

from segFM.DataLoaders.base_dataset import BaseImageDataset
from segFM import utils, prompts

BINARIZE_MASKS = True

class MedicalDecathlonDataset(BaseImageDataset):
    """
    This class is for loading the images from the Medical Decathlon dataset.
    The image, its segmentation masks as well as point/box prompts are returned as a dictionary when accessing the dataset.
    Find out more at [Medical Decathlon](http://medicaldecathlon.com/).
    """

    def __init__(self, task="Heart", mode="box", bob=None, transform=None, plot_prompts=False):
        assert task in [
            "Glioma",
            "Heart",
            "Spleen",
            "Hippocampus",
            "Prostate",
            "HepaticVessel",
        ], "Task must be either 'Glioma', 'Heart', 'Spleen', 'Hippocampus' or 'Prostate'."

        super(MedicalDecathlonDataset, self).__init__(transform=transform)

        task_to_class_id = {
            "Glioma": 22, # Brain Tumor
            "Heart": 29, # Heart Atrium Left
            "Spleen": 9, # Spleen
            "Hippocampus": -1,
            "Prostate": 20, # Prostate
            "HepaticVessel": -1,
        }
        task_to_preprocess = {
            "Glioma": "Brain MRI",
            "Heart": "Heart MRI",
            "Spleen": "Liver CT",
            "Hippocampus": "Brain MRI",
            "Prostate": "Prostate MRI",
            "HepaticVessel": "Abdomen CT",
        }
        self.task = task
        self.preprocess = task_to_preprocess[task]
        self.imgs_path, self.labels_path = self.get_folder_path(task)
        self.img_list = self.get_img_list(self.imgs_path, ".nii.gz")
        self.n_images = len(self.img_list)
        self.color_to_id = {
            0: 0,
            1: -1,
            2: -1,
            3: -1,
            4: -1,
            5: -1,
            255: task_to_class_id[task],
        }
        if self.task == "HepaticVessel":
            self.color_to_label = {
                0: "background",
                1: "HepaticVessel",
                2: "HepaticTumor",
                255: "HepaticVessel",
            }
        elif self.task == "Glioma":
            self.color_to_label = {
                0: "background",
                1: "edema",
                2: "non-enhancing tumor",
                3: "enhancing tumor",
                255: "Glioma",
            }
        self.id_to_label = {
            task_to_class_id[task]: task,
        }
        self.id_to_color = {
            task_to_class_id[task]: 255,
            0: 0,
        }
        self.bob = bob
        self.mode = mode
        self.plot_prompts = plot_prompts


    def __getitem__(self, idx):
        """
        Get the image, label, and prompt for the given index.
        """
        img_name = self.img_list[idx]
        img_path = os.path.join(self.imgs_path, img_name)
        gt_pth = os.path.join(self.labels_path, img_name)

        # Load the image and label
        image = utils.nifti_to_numpy2(img_path, preprocess=self.preprocess)
        gt = utils.nifti_to_numpy2(gt_pth)

        if BINARIZE_MASKS:
            gt[gt > 0] = 255  # Convert all non-zero values to 255 for binary segmentation

        # Reduce to random channel if multichannel image (channel, Z, Y, X)
        if image.ndim == 4 and image.shape[3] > 3:
            CHANNEL = 0 # Choose a channel to use for segmentation
            image = image[CHANNEL]

        if self.transform:
            image = self.transform(image)

        # Get the bounding box and points for the image
        if "BOB" in self.mode:
            # prompt = prompts.multicolor_box_prompt_3d(gt, dataset=self, plot_prompt=False)
            prompt = self.bob.generate_prompt(img=image, 
                                                     dataset=self,
                                                     confidence=0.25,
                                                     n_prompts_per_obj = 4,
                                                     multiprompt_z_spacing = 5,
                                                     allow_multiobject_3d = False,
                                                     plot_prompts=False,
                                                     allowed_classes = [self.color_to_id[255]]
                                            )
        else:
            # Generate prompts using GT
            prompt = prompts.multicolor_box_prompt_3d(gt=gt, dataset=self, plot_prompt=self.plot_prompts, img=image)
            
        data = {
            "img": image, 
            "gt": gt, 
            "prompts": prompt,
            "id": idx,
            "name": img_name,
            "gt_name": img_name,
        }
        
        return data

    def get_folder_path(self, mode):
        """
        Get the folder path based on the mode.
        """
        if mode == "Heart":
            img_path = "Datasets/MedicalDecathlon/Task02_Heart/imagesTr"
            label_path = "Datasets/MedicalDecathlon/Task02_Heart/labelsTr"
        elif mode == "Glioma":
            img_path = "Datasets/MedicalDecathlon/Task01_BrainTumour/imagesTr"
            label_path = "Datasets/MedicalDecathlon/Task01_BrainTumour/labelsTr"
        elif mode == "Spleen":
            img_path = "Datasets/MedicalDecathlon/Task09_Spleen/imagesTr"
            label_path = "Datasets/MedicalDecathlon/Task09_Spleen/labelsTr"
        elif mode == "Hippocampus":
            img_path = "Datasets/MedicalDecathlon/Task04_Hippocampus/imagesTr"
            label_path = "Datasets/MedicalDecathlon/Task04_Hippocampus/labelsTr"
        elif mode == "Prostate":
            img_path = "Datasets/MedicalDecathlon/Task05_Prostate/imagesTr"
            label_path = "Datasets/MedicalDecathlon/Task05_Prostate/labelsTr"
        elif mode == "HepaticVessel":
            img_path = "Datasets/MedicalDecathlon/Task08_HepaticVessel/imagesTr"
            label_path = "Datasets/MedicalDecathlon/Task08_HepaticVessel/labelsTr"
        else:
            raise ValueError("Invalid mode. Choose either 'Heart', 'Glioma', 'Spleen', 'Hippocampus', 'Prostate' or 'HepaticVessel'.")
        return img_path, label_path
