import numpy as np

from segFM import utils
from segFM import utils
from segFM.DataLoaders.amos22 import Amos22
from segFM import checkpoints


# Select one of the predictors - Make sure you have selected the corresponding venv 

#### MedSAM2Predictor3D ####
# from segFM.predictors.medsam_3d import MedSAM2Predictor3D
# predictor = MedSAM2Predictor3D(checkpoint=checkpoints.MedSAM2_latest) # Needs .venv_med

#### SAM-Med3D ####
# from segFM.predictors.samMed3d import SamMed3dPredictor
# predictor = SamMed3dPredictor() # Needs .venv

#### SAM2 Predictor 3D ####
# from segFM.predictors.sam2_3d import SAM2Predictor3D
# predictor = SAM2Predictor3D() # Needs .venv

### VISTA3D Predictor ####
# from segFM.predictors.vista_3d import VISTA3DPredictor
# predictor = VISTA3DPredictor(mode="label") # Needs .venv_vista

#### nnInteractive 3D Predictor ####
from segFM.predictors.nnInteractive_3d import nnInteractivePredictor3D
predictor = nnInteractivePredictor3D() # Needs .venv_nnInteractive

# Initialize the dataset
from BOB import BOB
bob = BOB(model="YOLOv12n")
#dataset = Amos22(mode="BOB D-FINE-N", prompt_gen=bob, plot_prompts=True, split="val", preprocess="auto")
dataset = Amos22(mode="BOB YOLOv12n",plot_prompts=False, split="val", preprocess="None", prompt_gen=bob)

print(f"Evaluating AMOS22 dataset with {dataset.n_images} images.")

# Call the evaluate_model method to get a dataframe with evaluation results
df = predictor.evaluate_model(
    dataset=dataset,
    n_images=25,  # Use the maximum number of images in the dataset
    #threads=4,
)

# Print the evaluation results
print(df)


# Save the evaluation results to a CSV file
utils.save_dataframe_to_csv(__file__, df, "results/AMOS22_final.csv")

#### This was used for something, I dont remember what ####
# Create a "flip the image across the x axis" transform
# def x_flip_transform(image, ground_truth):
#     """
#     Flips the image and ground truth across the x axis.
#     Images are of shape (Z, Y, X).
#     """
#     flipped_image = np.flip(image, axis=1)
#     flipped_ground_truth = np.flip(ground_truth, axis=1)
#     return flipped_image, flipped_ground_truth