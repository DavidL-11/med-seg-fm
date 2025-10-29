from segFM.DataLoaders.base_dataset import BaseImageDataset
from segFM import utils, prompts
import json
import napari
import numpy as np

class ToothFairyLabelDict(dict):
    """
    Custom dictionary for ToothFairy3 dataset labels.
    Maps specific tooth-related terms to a unique label ID that is used for training BOB models.
    This is also useful for filtering out non-tooth-related labels.
    """
    def __getitem__(self, key):
        k = key.lower()
        if not "pulp" in k and \
            ("canine" in k or \
            "incisor" in k or \
            "premolar" in k or \
            "molar" in k):
            return 18
        elif "pharynx" in k:
            return 13
        else:
            return super().__getitem__(key)
        
    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

class ToothFairy3(BaseImageDataset):
    """
    ToothFairy3 Dataset for tooth segmentation.
    """

    def __init__(self, transform=None, plot_prompts=False, tooth_only=False):
        """
        Initialize the ToothFairy3 dataset.

        Args:
            transform: Transformations to apply to the images.
            plot_prompts: Whether to plot the prompts using napari.
            tooth_only: If True, only include tooth labels in the ground truth and exclude other labels like pulp.
        """
        super(ToothFairy3, self).__init__(transform=transform)
        self.imgs_path = "Datasets/ToothFairy3/imagesTr/"
        self.labels_path = "Datasets/ToothFairy3/labelsTr/"
        self.coordinates_path = "Datasets/ToothFairy3/clicks/"

        self.plot_prompts = plot_prompts

        self.images = self.get_img_list(self.imgs_path, ".nii.gz")
        self.n_images = len(self.images)
        self.tooth_only = tooth_only

        with open("Datasets/ToothFairy3/dataset.json", "r") as f:
            dataset_info = json.load(f)

        # Color/Label/ID translation dictionaries
        labels = dataset_info.get("labels", [])
        self.label_to_color = labels
        self.color_to_label = {v: k for k, v in labels.items()}

        self.label_to_id = ToothFairyLabelDict()

        self.color_to_id = {color: self.label_to_id.get(label, -1) for label, color in self.label_to_color.items()}
        self.id_to_color = {self.label_to_id.get(label, -1): color for label, color in self.label_to_color.items()}                   

        # Get tooth colors after all dicts have been initialized
        self.tooth_colors = self.get_tooth_pharynx_colors()

    def __getitem__(self, idx):
        # Check if the index is valid
        if idx < 0 or idx > self.n_images:
            raise IndexError(
                f"This dataset only contains [1, {self.n_images + 1}] images. Index {idx} is out of range."
            )
        
        # Img Example: ToothFairy3F_001_0000.nii.gz
        img_name = self.images[idx] 

        # Gt Example: ToothFairy3F_001_gt.nii.gz
        gt_name = img_name.replace("_0000.nii.gz", ".nii.gz")

        # Load the image
        img = utils.nifti_to_numpy(self.imgs_path + img_name)
        gt = utils.nifti_to_numpy(self.labels_path + gt_name)

        if self.tooth_only:
            gt = np.where(np.isin(gt, self.tooth_colors), gt, 0)

        prompt = prompts.multicolor_box_prompt_3d(gt, dataset=self, plot_prompt=False)

        # Custom prompt plotting due to the amount of prompts
        if self.plot_prompts:
            # Plot the image and the prompts using napari
            viewer = napari.Viewer()
            viewer.add_image(img, name="Image")
            viewer.add_labels(gt, name="Ground Truth")
            # Prompt is a list of lists in the format [[z, y, x], ...]
            for pr in prompt:
                p = pr.box #Box is in format [x1, y1, x2, y2]
                point = (p[0] + p[2]) / 2, (p[1] + p[3]) / 2
                z = pr.z
                new_p = [z, point[1], point[0]]  # Convert to [z, y, x] format
                new_p = np.array(new_p, dtype=np.uint16)
                new_p = new_p.reshape(1, 3)

                viewer.add_points(
                    new_p,
                    size=10, 
                    face_color='red', 
                    name=pr.class_label,
                )
            napari.run()

        data = {
            "img": img,
            "gt": gt,
            "name": img_name,
            "gt_name": gt_name,
            "id": idx,
            "prompts": prompt,
        }

        return data
    
    def get_tooth_pharynx_colors(self):
        """
        Get the colors corresponding to tooth labels.
        """
        tooth_colors = []
        for label, color in self.label_to_color.items():
            l = label.lower()
            if not "pulp" in l and \
                ("canine" in l or \
                "incisor" in l or \
                "premolar" in l or \
                "molar" in l or \
                "pharynx" in l):
                tooth_colors.append(color)
        return tooth_colors


class TertiaryToothFairy3(BaseImageDataset):
    """
    ToothFairy3 Dataset for tooth segmentation with three classes: background, pharynx, tooth.
    """

    def __init__(self, transform=None, plot_prompts=False, mode="box", prompt_gen=None):
        """
        Initialize the ToothFairy3 dataset.

        Args:
            transform: Transformations to apply to the images.
            plot_prompts: Whether to plot the prompts using napari.
            tooth_only: If True, only include tooth labels in the ground truth and exclude other labels like pulp.
        """
        super(TertiaryToothFairy3, self).__init__(transform=transform)
        # Hardcoded paths
        self.imgs_path = "Datasets/ToothFairy3/imagesTr/"
        self.labels_path = "Datasets/ToothFairy3/labelsTr/"
        self.coordinates_path = "Datasets/ToothFairy3/clicks/"

        # Arguments
        self.plot_prompts = plot_prompts
        self.prompt_gen = prompt_gen
        self.mode = mode

        # Find images
        self.images = self.get_img_list(self.imgs_path, ".nii.gz")
        self.n_images = len(self.images)

        self._original_dataset = ToothFairy3()

        # Colors in the GT that correspond to tooth or pharynx labels
        self.tp_colors = self._original_dataset.get_tooth_pharynx_colors() 

        # Color/Label/ID translation dictionaries
        self.label_to_color = {
            "Background": 0,
            "Pharynx": 1,
            "Tooth": 2
        }
        self.color_to_label = {
            0: "Background",
            1: "Pharynx",
            2: "Tooth"
        }
        self.label_to_id =  {
            "Background": 0,
            "Pharynx": 13,
            "Tooth": 18
        }
        self.color_to_id = {
            1: 13,
            2: 18 
        }
        self.id_to_color = {
            13: 1,
            18: 2
        }
  

    def __getitem__(self, idx):
        # Check if the index is valid
        if idx < 0 or idx > self.n_images:
            raise IndexError(
                f"This dataset only contains [1, {self.n_images + 1}] images. Index {idx} is out of range."
            )
        
        # Img Example: ToothFairy3F_001_0000.nii.gz
        img_name = self.images[idx] 

        # Gt Example: ToothFairy3F_001_gt.nii.gz
        gt_name = img_name.replace("_0000.nii.gz", ".nii.gz")

        # Load the image
        img = utils.nifti_to_numpy(self.imgs_path + img_name)
        gt = utils.nifti_to_numpy(self.labels_path + gt_name)

        # Remove labels that are not in tooth_pharynx_colors
        tp_colors = self._original_dataset.get_tooth_pharynx_colors()
        gt = np.where(np.isin(gt, tp_colors), gt, 0)

        if "BOB" in self.mode:
            prompt = self.prompt_gen.generate_prompt(img=img, 
                                                     dataset=self, 
                                                     confidence=0.8,
                                                     iou=0.7,
                                                     n_prompts_per_obj=1,
                                                     multiprompt_z_spacing = 5,
                                                     max_z_distance=8,
                                                     plot_prompts=False,
                                                     allow_multiobject_3d=True,
                                                     allowed_classes=[13, 18])
        else:
            prompt = prompts.multicolor_box_prompt_3d(gt, dataset=self._original_dataset, plot_prompt=self.plot_prompts)

            for pr in prompt:
                if pr.color == 7: # Pharynx
                    pr.color = 1
                else:
                    pr.color = 2 # Tooth
                    pr.class_label = "Tooth"

        # Convert the ground truth to three classes: background, pharynx, tooth
        gt = self.convert_gt_to_tertiary(gt)

        data = {
            "img": img,
            "gt": gt,
            "name": img_name,
            "gt_name": gt_name,
            "id": idx,
            "prompts": prompt,
        }

        return data
    
    def convert_gt_to_tertiary(self, gt):
        """
        Convert the ground truth segmentation map to a tertiary classification.
        """
        new_gt = np.zeros_like(gt)

        for tc in self.tp_colors:
            new_gt[gt == tc] = 2  # Tooth and Pharynx
        
        new_gt[gt == 7] = 1  # Overwrite pharynx color in NewGT

        return new_gt