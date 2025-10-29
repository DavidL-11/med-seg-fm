from segFM import checkpoints, utils
from segFM.DataLoaders.toothfairy3 import ToothFairy3, TertiaryToothFairy3

from BOB import BOB
bob = BOB(model="D-FINE-N")

# image 95 is good

for cp in [checkpoints.MedSAM2_latest]: #checkpoints.MedSAM2CT
    
    from segFM.predictors.medsam_3d import MedSAM2Predictor3D
    pred = MedSAM2Predictor3D(checkpoint=cp)

    # from segFM.predictors.samMed3d import SamMed3dPredictor
    # pred = SamMed3dPredictor()

    # from segFM.predictors.sam2_3d import SAM2Predictor3D
    # pred = SAM2Predictor3D(checkpoint=checkpoints.SAM2_tiny, model_cfg=checkpoints.SAM2_tiny_cfg)

    # from segFM.predictors.sam2_3d_efficient import SAM2Predictor3D as SAM2Predictor3D_eff
    # pred = SAM2Predictor3D_eff(checkpoint=checkpoints.SAM2_tiny, model_cfg=checkpoints.SAM2_tiny_cfg)

    # from segFM.predictors.vista_3d import VISTA3DPredictor
    # pred = VISTA3DPredictor(mode="point")

    dataset = TertiaryToothFairy3(plot_prompts=False, mode="BOB D-FINE-N 0.8", prompt_gen=bob)

    df = pred.evaluate_model(
        dataset=dataset,
        n_images=15,
        concat_colors=True
    )

    utils.save_dataframe_to_csv(__file__, df, "results/bob_perf/ToothFairy3_bob.csv")
