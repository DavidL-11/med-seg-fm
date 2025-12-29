from segFM.DataLoaders.base_dataset import BaseImageDataset
from segFM import utils, prompts

from collections import defaultdict
import tifffile as tiff

class BrainMicroscopy(BaseImageDataset):
    """
    Brain Electron Microscopy Dataset for segmenting mitochondria.
    """

    def __init__(self, transform=None, plot_prompts=False, mode="box",
                 n_pos=1, n_neg=0, bbsize=0, prompt_finder="random", bob=None):
        super(BrainMicroscopy, self).__init__(transform=transform)
        self.dataset_name = "BrainElectronMicroscopy"
        self.mode = mode
        self.n_pos = n_pos
        self.n_neg = n_neg
        self.bbsize = bbsize
        self.prompt_finder = prompt_finder
        self.prompt_gen = bob

        self.imgs_path = "/run/media/david/SSD1TB/Datasets/ElectronMicroscopy/images/"
        self.labels_path = "/run/media/david/SSD1TB/Datasets/ElectronMicroscopy/masks/"
        self.img_list = self.get_img_list(self.imgs_path, ".tif")
        self.n_images = len(self.img_list)

        self.plot_prompts = plot_prompts
        self.id_to_color = defaultdict(lambda: 255)
        self.color_to_id = defaultdict(lambda: 10)
        self.color_to_label = {1: "Mitochondria", 255: "Mitochondria"}
        self.label_to_id = defaultdict(lambda: 10)
        self.images = []

        self.tiff_to_multiple_imgs()
    
    def tiff_to_multiple_imgs(self):
        for img_name in self.img_list:
            # Load the image
            img = tiff.imread(self.imgs_path + img_name)

            # Load the ground truth mask
            gt = tiff.imread(self.labels_path + img_name)

            # Save each slice as a separate image
            for i in range(img.shape[0]):
                slice_img = img[i, :, :]
                slice_gt = gt[i, :, :]
                self.images.append((slice_img, slice_gt))

        self.n_images = len(self.images)

    def __getitem__(self, idx):
        # Check if the index is valid
        if idx < 0 or idx > self.n_images:
            raise IndexError(
                f"This dataset only contains [1, {self.n_images + 1}] images. Index {idx} is out of range."
            )

        img, gt = self.images[idx]

        if "BOB" in self.prompt_finder:
            prompt = self.prompt_gen.generate_prompt(img,
                                                     dataset=self,
                                                     confidence=0.7,
                                                     n_prompts_per_obj=1,
                                                     multiprompt_z_spacing=5,
                                                     max_z_distance=10,
                                                     allow_multiobject_3d=True,
                                                     allowed_classes=[self.label_to_id["Label"]],
                                                    )
        else:
            prompt = prompts.get_multiobject_prompt(
                gt=gt,
                image=img,
                dataset=self,
                bbsize=self.bbsize,
                mode=self.mode,
                prompt_finder=self.prompt_finder,
                n_pos=self.n_pos,
                n_neg=self.n_neg,
                plot_prompts=self.plot_prompts,
            )

        img = utils.grayscale_to_rgb(img)

        for p in prompt:
            p.class_id = self.label_to_id["Label"]

        data = {
            "img": img,
            "gt": gt,
            "name": f"brain_em_{idx}.png",
            "id": idx,
            "prompts": prompt,
        }

        return data
