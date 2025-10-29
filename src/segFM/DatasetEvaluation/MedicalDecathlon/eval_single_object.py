import torch

from segFM.DataLoaders.medical_decathlon import MedicalDecathlonDataset
from segFM import checkpoints, utils

torch.set_float32_matmul_precision('high')

def evaluate_sam_med3d(task, n_images=50):
    from segFM.predictors.samMed3d import SamMed3dPredictor

    predictor = SamMed3dPredictor()

    dataset = MedicalDecathlonDataset(task=task, mode="random point")

    df = predictor.evaluate_model(dataset=dataset, n_images=n_images)

    utils.save_dataframe_to_csv(__file__, df, "results/msd.csv")

def evaluate_vista3d(task, n_images=50):
    dataset = MedicalDecathlonDataset(task=task, mode="box",)

    from segFM.predictors.vista_3d import VISTA3DPredictor
    for prompt_type in ["point", "label", "zero-shot"]:
        predictor = VISTA3DPredictor(mode=prompt_type) # Needs .venv_vista

        df = predictor.evaluate_model(dataset=dataset, n_images=n_images)

        utils.save_dataframe_to_csv(__file__, df, "results/msd.csv")

def evaluate_medsam2(task, n_images=50):
    from segFM.predictors.medsam_3d import MedSAM2Predictor3D
    checkpoint = checkpoints.MedSAM2_latest

    predictor = MedSAM2Predictor3D(checkpoint=checkpoint)

    for mode in ["BOB YOLOv12n"]:
        bob = None
        if mode == "BOB YOLOv12n":
            from BOB.prompt_generator import BOB
            bob = BOB(model="YOLOv12n")

        dataset = MedicalDecathlonDataset(task=task, mode=mode, bob=bob, plot_prompts=False)

        df = predictor.evaluate_model(dataset=dataset, n_images=n_images)

        utils.save_dataframe_to_csv(__file__, df, "results/msd.csv")

def evaluate_sam3d(task, n_images=50):
    from segFM.predictors.sam2_3d import SAM2Predictor3D
    checkpoint = checkpoints.SAM2_tiny
    model_cfg = checkpoints.SAM2_tiny_cfg

    predictor = SAM2Predictor3D(checkpoint=checkpoint, config=model_cfg)

    for mode in ["box"]:
        bob = None
        if mode == "BOB D-FINE-N":
            from BOB.prompt_generator import BOB
            bob = BOB(model="D-FINE-N")

        dataset = MedicalDecathlonDataset(task=task, mode=mode, bob=bob)

        df = predictor.evaluate_model(dataset=dataset, n_images=n_images)

        utils.save_dataframe_to_csv(__file__, df, "results/msd.csv")


if __name__ == "__main__":
    TASK = "Spleen"  # Spleen, Hippocampus, Prostate, Glioma, Heart, HepaticVessel
    N_IMAGES = 50  # Number of images to evaluate on. Max is 41 for Spleen

    evaluate_medsam2(task=TASK, n_images=N_IMAGES)
    #evaluate_sam3d(task=TASK, n_images=N_IMAGES)
    #evaluate_vista3d(task=TASK, n_images=N_IMAGES)
    #evaluate_sam_med3d(task=TASK, n_images=N_IMAGES)