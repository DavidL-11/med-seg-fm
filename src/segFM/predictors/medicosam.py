from medico_sam.util import get_medico_sam_model
from medico_sam.evaluation.inference import _run_inference_with_iterative_prompting_for_image
import torch, os

from segFM.DataLoaders.flare22 import Flare22

dataset = Flare22()
image = dataset[0]["img"]
gt = dataset[0]["gt"]
image_name = dataset[0]["name"]

device = device = torch.device("cuda")

model = get_medico_sam_model(
    model_type="vit_b",
    device=device,
    checkpoint_path="/home/david/Documents/segFM/vit_b.pt",
    use_sam3d=False,
    use_sam_med2d=True)

embedding_path = os.path.join("temp_embeddings", f"{os.path.splitext(image_name)[0]}.zarr")

_run_inference_with_iterative_prompting_for_image(
    model, image, gt, start_with_box_prompt=True,
    dilation=5, batch_size=1, embedding_path=embedding_path,
    n_iterations=1, prediction_paths=None, use_masks=False
)