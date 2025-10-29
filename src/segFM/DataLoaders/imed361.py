from segFM.DataLoaders.base_dataset import BaseImageDataset
from segFM import prompts
import json
import os
from collections import defaultdict
import numpy as np
from PIL import Image
from scipy.sparse import load_npz
from segFM.logger import logger
import matplotlib.pyplot as plt

class IMed361(BaseImageDataset):
    """
    IMed361 Dataset containing many sub-datasets to evaluate the performance of segmentation models.
    """

    def __init__(
        self,
        dataset_id=0,
        split="training",
        transform=None,
        plot_prompts=False,
        mode="point",
        prompt_finder="random",
        n_pos=1,
        n_neg=0,
        bbsize=0,
        prompt_gen=None,
    ):
        """
        Initialize the IMed361 dataset.

        Args:
            dataset_id (int): ID of the dataset to load (0 to n_datasets-1).
            split (str): "training" or "test" split of the dataset.
            transform (callable, optional): A function/transform that takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            plot_prompts (bool): Whether to plot the prompts on the images.
            mode (str): The prompt mode to use. Options are "point", "box", "BOB D-FINE-N", "BOB YOLOv12n", etc.
            prompt_finder (str): The method to find prompts. Options are "random", "center", "rim"
            n_pos (int): Number of positive point prompts to use.
            n_neg (int): Number of negative point prompts to use.
            bbsize (int or str): Bounding box size for box mode. If int, it is the pixel size. If str, it is the percentage of the image size (e.g. "10%").
            prompt_gen (object): The prompt generator model to use for BOB mode.
        """
        super(IMed361, self).__init__(transform=transform)
        # Other path because this dataset requires a lot of space
        self.dataset_path = "/media/david/SHARED/IMed-361M"

        self.split = split  # "training" or "test"
        self.plot_prompts = plot_prompts
        self.mode = mode
        self.prompt_finder = prompt_finder
        self.n_pos = n_pos
        self.n_neg = n_neg
        self.bbsize = bbsize
        self.prompt_gen = prompt_gen

        # Walk through the dataset path to find all datasets as directories
        self.datasets = [
            d
            for d in os.listdir(self.dataset_path)
            if os.path.isdir(os.path.join(self.dataset_path, d))
        ]

        print(f"Found {len(self.datasets)} datasets in {self.dataset_path}.")
            
        self.n_datasets = len(self.datasets)
        self.dataset_name = self.datasets[dataset_id]

        logger.info(
            f"Initialized IMed361 dataset {dataset_id} [{self.dataset_name}] out of {self.n_datasets} datasets."
        )

        self.load_single_dataset(self.dataset_name)


    def load_single_dataset(self, dataset_name):
        """
        Load a single dataset from the dataset path.
        """
        # Read the dataset.json file and extract necessary information
        with open(
            os.path.join(self.dataset_path, dataset_name, "dataset.json"), "r"
        ) as f:
            dataset_info = json.load(f)

        # Get the image and label paths
        labels = dataset_info.get("labels", [])
        modality = dataset_info.get("modality", {}).values()

        self.color_to_label = {
            int(k): v for k, v in labels.items() if isinstance(k, str) and k.isdigit()
        }

        self.label_to_color = {v: int(k) for k, v in labels.items()}
        self.color_to_id = defaultdict(lambda: -1)  # Default color ID for unknown colors

        #ID mapping for MedPAM Prompt Generator (see dataset.yaml)
        self.label_to_id = self.get_label_to_id()
        self.id_to_color = self.get_id_to_color()
        self.color_to_id = {
            v: k for k, v in self.id_to_color.items()
        }  # Invert the id_to_color mapping

        # Prepare the image list
        img_list = dataset_info.get(self.split, [])
        self.images = self.extract_images(img_list)
        self.n_images = len(self.images)

        # Store the dataset modalities as string (e.g. "CT | MRI")
        self.modality = " | ".join(modality) if modality else "Unknown"

    def extract_images(self, img_list):
        """
        Extract image names from the img list that can be found in the dataset.json file.
        The img_list is a list of dictionaries, each containing the key "image", "label" and "imask".
        The label (.npz) and imask (.npy) differ from each other in the senses that the label is the
        ground truth labelmap per object and the imask is an image containing possible segmentation masks.

        Returns:
            list(tuple): A list of tuples, each containing the image name, label name, and imask name.
        """
        images = map(lambda x: (x["image"], x["label"], x["imask"]), img_list)
        return list(images)

    def __getitem__(self, idx):
        # Check if the index is valid
        if idx < 0 or idx > self.n_images:
            raise IndexError(
                f"This dataset only contains [1, {self.n_images + 1}] images. Index {idx} is out of range."
            )

        # Get the image name, label name, and imask name
        image_name, label_name, imask_name = self.images[idx]

        # Since the image names are paths like image/x/patient058_frame01_3.jpg,
        # we can use the last part as the image ID.
        image_name_clean = image_name.split("/")[-1] # Only keep the last part of the path

        # Prepare the data paths
        image_path = os.path.join(self.dataset_path, self.dataset_name, image_name)
        label_path = os.path.join(self.dataset_path, self.dataset_name, label_name)
        # imask_path = os.path.join(self.dataset_path, self.dataset_name, imask_name) # .npy

        # Load the image and label
        image = np.array(Image.open(image_path).convert("RGB"))
        gts = load_npz(label_path) # Label is stored as a sparse matrix (.npz)
        gts = gts.toarray()  # Convert sparse matrix to dense array
        # Convert flattened to a 2D labelmap with n_classes channels
        n_classes = len(self.color_to_label) - 1  # Exclude background class
        gts = gts.reshape(
            (n_classes, image.shape[0], image.shape[1])
        )  # Reshape to (3, height, width)

        # Set all 1s in each channel to the channel's index
        for i in range(n_classes):
            gts[i, :, :] = np.where(
                gts[i, :, :] > 0, i + 1, 0
            )  # Set the channel index as the color value

        # If the dataset is "gamma", set the first three rows in each channel of the GT to 0 (corrupted)
        if "gamma" in self.dataset_name.lower():
            gts[:, :3, :] = 0

        if "BOB" in self.mode:
            if self.prompt_gen is None:
                raise ValueError("Prompt generator model is not specified for BOB mode.")
            prompt = self.prompt_gen.generate_prompt(image, dataset=self, confidence=0.3)
        else:
            # Get the bounding box and points for the image
            prompt = prompts.get_labelmap_prompt(
                gts,
                image,
                dataset=self,
                bbsize=self.bbsize,
                mode=self.mode,
                prompt_finder=self.prompt_finder,
                n_pos=self.n_pos,  # Number of positive prompts
                n_neg=self.n_neg,  # Number of negative prompts
                plot_prompts=self.plot_prompts,
            )

        data = {
            "img": image,
            "gt": gts,
            "id": idx,
            "name": image_name_clean,
            "prompts": prompt,
        }

        return data
    

    def get_label_to_id(self):
        """
        Get the color to ID mapping.
        """
        return {
            "lung": 3,
            "polyp": 4,
            "liver": 7,
            "kidney_right": 8,
            "spleen": 9,
            "aorta": 11,
            "inferior_vena_cava": 12,
            "gallbladder": 15,
            "esophagus": 16,
            "stomach": 17,
            "kidney_left": 19,
            "prostate": 99 if "amos" in self.dataset_name.lower() else 20,  # Special case for AMOS dataset
            "prostate_and_uterus": 20 if "amos" in self.dataset_name.lower() else 99,  # Special case for AMOS dataset
            "skin_lesion": 21,
            "glioma": 22,
            "optic_disc": 23,
            "optic_cup": 24,
            "heart_myocardium": 26,
            "heart_ventricle_left": 27,
            "heart_ventricle_right": 28,
            "heart_atrium_left": 29,
        }
    
    def get_id_to_color(self):
        """
        Get the ID to color mapping.
        """
        id_to_label = {v: k for k, v in self.label_to_id.items()}
        return {k: self.label_to_color.get(v, 255) for k, v in id_to_label.items()}