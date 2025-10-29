"""
This file contains mappings between color codes, IDs, and labels for various segmentation datasets.
This is needed to convert between Ground Truth Colors, Labels (for human readability) and IDs.
"""

bagls_color_to_id = {
    0: None,  # "Background"
    1: 0,  # "Glottis"
    2: 1,  # "Left Vocal Fold"
    3: 2,  # "Right Vocal Fold"
    4: -1,  # "Unknown"
}
bagls_id_to_color = {
    0: 1,  # "Glottis"
    1: 2,  # "Left Vocal Fold"
    2: 3,  # "Right Vocal Fold"
}
bagls_color_to_label = {
    0: None,  # "Background"
    1: "Glottis",
    2: "Left Vocal Cord",
    3: "Right Vocal Cord",
    4: "Unknown", # Sometimes a fourth channel is present, idk why
}
flare22_color_to_id = {
    1: 7, #"Liver"
    2: 8, #"Right Kidney"
    3: 9, #"Spleen"
    4: -1, #"Pancreas"
    5: 11, #"Aorta"
    6: 12, #"Inferior Vena Cava"
    7: -1, #"Right Adrenal Gland"
    8: -1, #"Left Adrenal Gland"
    9: 15, #"Gallbladder"
    10: 16, #"Esophagus"
    11: 17, #"Stomach"
    12: -1, #"Duodenum"
    13: 19, #"Left Kidney"
}
flare22_id_to_color = {
    7: 1,  # "Liver"
    8: 2,  # "Right Kidney"
    9: 3,  # "Spleen"
    10: 4,  # "Pancreas"
    11: 5,  # "Aorta"
    12: 6,  # "Inferior Vena Cava"
    13: 7,  # "Right Adrenal Gland"
    14: 8,  # "Left Adrenal Gland"
    15: 9,  # "Gallbladder"
    16: 10,  # "Esophagus"
    17: 11,  # "Stomach"
    18: 12,  # "Duodenum"
    19: 13,  # "Left Kidney"
    20: 14,  # "Bladder"
    21: 15,  # "Prostate/Uterus"
}

flare22_label_to_color = {
    "Liver": 1,
    "Right Kidney": 2,
    "Spleen": 3,
    "Pancreas": 4,
    "Aorta": 5,
    "Inferior Vena Cava": 6,
    "Right Adrenal Gland": 7,
    "Left Adrenal Gland": 8,
    "Gallbladder": 9,
    "Esophagus": 10,
    "Stomach": 11,
    "Duodenum": 12,
    "Left Kidney": 13,
    "Bladder": 14,
    "Prostate/Uterus": 15
}
flare22_color_to_label = {
    1: "Liver",
    2: "Right Kidney",
    3: "Spleen",
    4: "Pancreas",
    5: "Aorta",
    6: "Inferior Vena Cava",
    7: "Right Adrenal Gland",
    8: "Left Adrenal Gland",
    9: "Gallbladder",
    10: "Esophagus",
    11: "Stomach",
    12: "Duodenum",
    13: "Left Kidney",
}
amos22_color_to_label = {
    1: "Spleen",
    2: "Right Kidney",
    3: "Left Kidney",
    4: "Gallbladder",
    5: "Esophagus",
    6: "Liver",
    7: "Stomach",
    8: "Aorta",
    9: "Inferior Vena Cava", #= Postcava
    10: "Pancreas",
    11: "Right Adrenal Gland",
    12: "Left Adrenal Gland",
    13: "Duodenum",
    14: "Bladder",
    15: "Prostate/Uterus"
}
amos22_label_to_color = {
    "Spleen": 1,
    "Right Kidney": 2,
    "Left Kidney": 3,
    "Gallbladder": 4,
    "Esophagus": 5,
    "Liver": 6,
    "Stomach": 7,
    "Aorta": 8,
    "Inferior Vena Cava": 9,  # Postcava
    "Pancreas": 10,
    "Right Adrenal Gland": 11,
    "Left Adrenal Gland": 12,
    "Duodenum": 13,
    "Bladder": 14,
    "Prostate/Uterus": 15
}
amos22_color_to_id = {
    1: 9,  # "Spleen"
    2: 8,  # "Right kidney"
    3: 19,  # "Left kidney"
    4: 15,  # "Gallbladder"
    5: 16,  # "Esophagus"
    6: 7,  # "Liver"
    7: 17,  # "Stomach"
    8: 11,  # "Aorta"
    9: 12,  # "Inferior vena cava" (Postcava)
    10: -1,  # "Pancreas"
    11: -1,  # "Right adrenal gland"
    12: -1,  # "Left adrenal gland"
    13: -1,  # "Duodenum"
    14: -1,  # "Bladder"
    15: 20,  # "Prostate/uterus"
}
amos22_id_to_color = {
    9: 1,   # "Spleen"
    8: 2,   # "Right kidney"
    19: 3,  # "Left kidney"
    15: 4,  # "Gallbladder"
    16: 5,  # "Esophagus"
    7: 6,   # "Liver"
    17: 7,  # "Stomach"
    11: 8,  # "Aorta"
    12: 9,  # "Inferior vena cava" (Postcava)
    20: 15, # "Prostate/uterus"
}

endoscapes_color_to_label = {
    0: "Background",
    1: "Cystic Plate",
    2: "Calot Triangle",
    3: "Cystic Artery",
    4: "Cystic Duct",
    5: "Gallbladder",
    6: "Tool"
}
# See https://github.com/Project-MONAI/VISTA/blob/main/vista3d/data/jsons/label_dict.json
label_to_vista_id = {
    "Liver": 1,
    "Spleen": 3,
    "Pancreas": 4,
    "Right Kidney": 5,
    "Aorta": 6,
    "Inferior Vena Cava": 7,
    "Right Adrenal Gland": 8,
    "Left Adrenal Gland": 9,
    "Gallbladder": 10,
    "Esophagus": 11,
    "Stomach": 12,
    "Duodenum": 13,
    "Left Kidney": 14,
    "Bladder": 15,
    "Prostate/Uterus": 118,  # Only Prostate is supported in VISTA3D
    "Prostate": 118,
    "HepaticVessel": 25,
    "HepaticTumor": 26,
}

