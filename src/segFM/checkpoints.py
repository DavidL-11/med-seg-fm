"""
This file contains paths to various model checkpoints used in this project.
"""

### SAM2 ### https://github.com/facebookresearch/sam2
SAM2_tiny = "src/segFM/predictors/sam2/checkpoints/sam2.1_hiera_tiny.pt"
SAM2_tiny_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"

SAM2_small = "src/segFM/predictors/sam2/checkpoints/sam2.1_hiera_small.pt"
SAM2_small_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"

SAM2_base = "src/segFM/predictors/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
SAM2_base_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"

SAM2_large = "src/segFM/predictors/sam2/checkpoints/sam2.1_hiera_large.pt"
SAM2_large_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

MedSAM_cfg = "configs/sam2.1_hiera_t512.yaml"
MedSAM2_CT = "src/segFM/predictors/MedSAM2/checkpoints/MedSAM2_CTLesion.pt"
MedSAM2_latest = "src/segFM/predictors/MedSAM2/checkpoints/MedSAM2_latest.pt"
MedSAM2_Heart = "src/segFM/predictors/MedSAM2/checkpoints/MedSAM2_US_Heart.pt"
MedSAM2_Liver = "src/segFM/predictors/MedSAM2/checkpoints/MedSAM2_MRI_LiverLesion.pt"
MedSAM2_2411 = "src/segFM/predictors/MedSAM2/checkpoints/MedSAM2_2411.pt"