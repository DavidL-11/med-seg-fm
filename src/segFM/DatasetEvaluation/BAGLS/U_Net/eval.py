import numpy as np
import os

from segFM.predictors.bagls_unet import BAGLS_U_Net_Predictor
from segFM.DataLoaders.bagls import BAGLS_Images
from segFM.logger import logger
from segFM import utils

if __name__ == "__main__":
    N_IMAGES = 500

    predictor = BAGLS_U_Net_Predictor(model_path=os.path.join(os.path.dirname(__file__), "bagls_unet_nobn.pth"))

    dataset = BAGLS_Images(
        type="test",
        transform=None,
        mode="point",
        prompt_finder="random",
        n_pos=1,
        n_neg=0,
        bbsize=50,
        noisy=False,
    )

    df = predictor.evaluate_model(
        dataset=dataset,
        n_images=N_IMAGES,
        plot_results=False,
    )

    print(df)

    logger.info(
        f" DSC - Average: {np.mean(df['DSC']):.4f}, Median: {np.median(df['DSC']):.4f}"
    )
    logger.info(
        f" IOU - Average: {np.mean(df['IoU']):.4f}, Median: {np.median(df['IoU']):.4f}"
    )
    logger.info(
        f" NSD - Average: {np.mean(df['NSD']):.4f}, Median: {np.median(df['NSD']):.4f}"
    )

    # Save the CSV file in parent directory
    directory = os.path.dirname(os.path.dirname(__file__))
    output_path = os.path.join(directory, "BAGLS.csv")

    if os.path.exists(output_path):
        # Append the new data to the existing csv
        df.to_csv(output_path, mode="a", header=False, index=False)
    else:
        # Create a new csv file with the data
        df.to_csv(output_path, index=False)
