from segFM.DataLoaders.base_dataset import BaseImageDataset
from segFM import utils, prompts
from segFM import color_classes
import numpy as np

class Flare22(BaseImageDataset):
    """
    Flare22 Dataset for multi-organ abdomen segmentation.
    """

    def __init__(self, mode="box", bob=None, plot_prompts=False, transform=None, preprocess="Abdomen CT"):
        """
        Initialize the Flare22 dataset.

        Args:
            mode (str): The mode of prompt generation. Options are "box" or "BOB <model_name>".
            bob (BOB): An instance of the BOB prompt generator. Required if mode contains "BOB".
            plot_prompts (bool): Whether to plot the generated prompts.
            transform: Optional transformations to apply to the images and masks.
        """
        super(Flare22, self).__init__(transform=transform)
        self.imgs_path = "/media/david/SHARED/FLARE22Train/images/"
        self.labels_path = "/media/david/SHARED/FLARE22Train/labels/"
        self.n_images = 49
        self.plot_prompts = plot_prompts
        self.prompt_gen = bob
        self.mode = mode
        self.label_to_color = color_classes.flare22_label_to_color
        self.color_to_label = color_classes.flare22_color_to_label
        self.color_to_id = color_classes.flare22_color_to_id
        self.id_to_color = color_classes.flare22_id_to_color
        self.preprocess = preprocess

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        # Check if the index is valid
        if idx < 0 or idx > self.n_images:
            raise IndexError(
                f"This dataset only contains [1, {self.n_images + 1}] images. Index {idx} is out of range."
            )

        idx += 1  # Adjust index to match the dataset's 1-based indexing

        image_number = str(idx).zfill(2)
        nii_fname = f"FLARE22_Tr_00{image_number}_0000.nii.gz"
        nii_gtname = f"FLARE22_Tr_00{image_number}.nii.gz"

        # Random preprocessing for BOB training is recommended
        if self.preprocess == "random":
            preprocess = np.random.choice(["Abdomen CT", "Liver CT", "None"], p=[0.65, 0.15, 0.2])
        else:
            preprocess = self.preprocess

        # Load the image
        img = utils.nifti_to_numpy2(self.imgs_path + nii_fname, preprocess=preprocess)
        gt = utils.nifti_to_numpy2(self.labels_path + nii_gtname)

        if "BOB" in self.mode:
            prompt = self.prompt_gen.generate_prompt(
                img, 
                dataset=self, 
                confidence=0.5,
                max_det=13,
                n_prompts_per_obj=5,
                multiprompt_z_spacing=15,
                max_z_distance=30,
                allow_multiobject_3d=False,
                plot_prompts=self.plot_prompts)
        elif self.mode == "box":
            prompt = prompts.multicolor_box_prompt_3d(gt, dataset=self, plot_prompt=self.plot_prompts)
        else:
            raise ValueError(f"Mode {self.mode} not recognized. Use 'box' or 'BOB <model_name>'.")
        
        data = {
            "img": img,
            "gt": gt,
            "name": nii_fname,
            "gt_name": nii_gtname,
            "id": idx,
            "prompts": prompt,
        }

        return data
