from segFM import utils

from segFM import checkpoints, utils
from segFM.DataLoaders.flare22 import Flare22
from BOB.prompt_generator import BOB

# Select one of the predictors - Make sure you have selected the corresponding venv 
prompt_gen = BOB(model="D-FINE-N") # D-FINE-N, YOLOv12n, YOLOv12s

# Initialize the dataset
dataset = Flare22(mode="BOB D-FINE-N", bob=prompt_gen, plot_prompts=False, preprocess="Abdomen CT")

#### MedSAM2Predictor3D ####
from segFM.predictors.medsam_3d import MedSAM2Predictor3D
predictor = MedSAM2Predictor3D(checkpoint=checkpoints.MedSAM2_latest) # Needs .venv_med

#### SAM-Med3D ####
# from segFM.predictors.samMed3d import SamMed3dPredictor
# predictor = SamMed3dPredictor() # Needs .venv

#### SAM2 Predictor 3D ####
#from segFM.predictors.sam2_3d import SAM2Predictor3D
#predictor = SAM2Predictor3D() # Needs .venv

#### VISTA3D Predictor ####
# from segFM.predictors.vista_3d import VISTA3DPredictor
# predictor = VISTA3DPredictor(mode="label") # Needs .venv_vista

# Call the evaluate_model method to get a dataframe with evaluation results
df = predictor.evaluate_model(
    dataset=dataset,
    n_images=50,  # Use the maximum number of images in the dataset
)

# Print the evaluation results
print(df)

# Save the evaluation results to a CSV file
utils.save_dataframe_to_csv(__file__, df, "results/FLARE22.csv")