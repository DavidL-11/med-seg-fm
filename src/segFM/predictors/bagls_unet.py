import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import torch.nn as nn
import os
import pandas as pd

from segFM.utils import dice_score, intersection_over_union
from segFM import utils


class UNet(torch.nn.Module):
    """
    U-Net model for image segmentation. Implemented as described in the paper.

    https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.encoder1 = DoubleConv3x3(in_channels, 64)
        self.pool1 = MaxPool2x2()

        self.encoder2 = DoubleConv3x3(64, 128)
        self.pool2 = MaxPool2x2()

        self.encoder3 = DoubleConv3x3(128, 256)
        self.pool3 = MaxPool2x2()

        self.encoder4 = DoubleConv3x3(256, 512)
        self.pool4 = MaxPool2x2()

        self.bottleneck = DoubleConv3x3(512, 1024)

        self.upconv4 = UpConv2x2(1024, 512)
        self.decoder4 = DoubleConv3x3(1024, 512)

        self.upconv3 = UpConv2x2(512, 256)
        self.decoder3 = DoubleConv3x3(512, 256)

        self.upconv2 = UpConv2x2(256, 128)
        self.decoder2 = DoubleConv3x3(256, 128)

        self.upconv1 = UpConv2x2(128, 64)
        self.decoder1 = DoubleConv3x3(128, 64)

        self.final_conv = Conv1x1(64, out_channels)

    def forward(self, x):
        step1 = self.encoder1(x)  # Double conv 3x3 + ReLU
        step1_pool = self.pool1(step1)  # Maxpool 2x2

        step2 = self.encoder2(step1_pool)  # Double conv 3x3 + ReLU
        step2_pool = self.pool2(step2)  # Maxpool 2x2

        step3 = self.encoder3(step2_pool)  # Double conv 3x3 + ReLU
        step3_pool = self.pool3(step3)  # Maxpool 2x2

        step4 = self.encoder4(step3_pool)  # Double conv 3x3 + ReLU
        step4_pool = self.pool4(step4)  # Maxpool 2x2

        step5_bottom = self.bottleneck(step4_pool)  # Double conv 3x3 + ReLU

        step6_input1 = self.upconv4(step5_bottom)  # Upconv 2x2
        step6_input2 = torch.cat((step6_input1, step4), dim=1)  # Copy and crop
        step6_output = self.decoder4(step6_input2)  # Double conv 3x3 + ReLU

        step7_input1 = self.upconv3(step6_output)  # Upconv 2x2
        step7_input2 = torch.cat((step7_input1, step3), dim=1)  # Copy and crop
        step7_output = self.decoder3(step7_input2)  # Double conv 3x3 + ReLU

        step8_input1 = self.upconv2(step7_output)  # Upconv 2x2
        step8_input2 = torch.cat((step8_input1, step2), dim=1)  # Copy and crop
        step8_output = self.decoder2(step8_input2)  # Double conv 3x3 + ReLU

        step9_input1 = self.upconv1(step8_output)  # Upconv 2x2
        step9_input2 = torch.cat((step9_input1, step1), dim=1)  # Copy and crop
        step9_output = self.decoder1(step9_input2)  # Double conv 3x3 + ReLU

        step10 = self.final_conv(step9_output)  # Conv 1x1

        return step10
    

class DoubleConv3x3(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv3x3, self).__init__()
        self.conv = torch.nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            #nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            #nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class MaxPool2x2(torch.nn.Module):
    def __init__(self):
        super(MaxPool2x2, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.pool(x)


class UpConv2x2(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv2x2, self).__init__()
        self.upconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )

    def forward(self, x):
        return self.upconv(x)


class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class BAGLS_U_Net_Predictor:
    """
    Loads the previously trained UNet model and uses it to predict the segmentation of a given image.
    """

    def __init__(self, model_path, device="cuda"):
        self.device = device
        self.model = UNet(in_channels=1, out_channels=1)
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        self.model.eval()


    def preprocess_image(self, image):
        """
        Preprocess the RGB input image for the model.
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Convert to grayscale
        image = cv2.resize(image, (256, 512))

        image = image.astype(np.float32) / 255.0  # Normalize to [0, 1]

        image = np.expand_dims(image, axis=0)  # Add channel dimension
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        image = torch.tensor(image, dtype=torch.float32)
        return image.to(self.device)

    def postprocess_mask(self, mask, shape):
        mask = torch.sigmoid(mask)
        mask = mask.data.cpu().numpy()

        # Resize back to original shape
        mask = np.squeeze(mask)  # Remove batch and channel dimensions
        mask = cv2.resize(mask, (shape[1], shape[0]))  # Resize to original shape

        # Binarize the output mask
        mask = (mask > 0.5).astype(np.uint8) * 255  # Convert to binary mask
        return mask

    def predict(self, image):
        """
        Predict the segmentation of a given image.

        Parameters:
            image (numpy.ndarray): The input image to be segmented.

        Returns:
            numpy.ndarray: The predicted segmentation mask.
        """
        shape = image.shape
        image = self.preprocess_image(image)

        with torch.no_grad():
            image = image.to(self.device)
            output = self.model(image)

        return self.postprocess_mask(output, shape)
    
    def evaluate_model(self, dataset, n_images, plot_results=False):
        """
        Evaluates the model on a random set of images.

        Parameters:
            n_images (int): Number of images to evaluate.
            plot_results (bool): Whether to plot the results.

        Returns:
            DataFrame: A DataFrame containing the evaluation results.
        """
        dsc = []
        iou = []
        nsd = []
        names = []
        objects = []

        # Get n images from the dataset
        data_list = dataset.get_n_nonduplicate_images(n_images)

        for i, data in enumerate(data_list):
            if i % max(n_images // 10, 1) == 0:
                print(f"Evaluating image {i + 1}/{n_images}")

            image = data["img"]
            ground_truth = data["gt"]

            # Predict the segmentation
            segmask = self.predict(image)

            # Calculate metrics
            _dsc, _iou, _nsd = utils.compute_metrics(segmask, ground_truth)

            dsc.append(_dsc)
            iou.append(_iou)
            nsd.append(_nsd)
            names.append(data["name"])
            objects.append(dataset.color_to_label[255])

        # Create a DataFrame to store the results
        results = pd.DataFrame({
            "Image": names,
            "Object": objects,
            "DSC": dsc,
            "IoU": iou,
            "NSD": nsd
        })

        # Insert "Model" column with the model name
        results.insert(0, "Model", "U-Net")
        # Insert "Dataset" column with the dataset name
        results.insert(1, "Dataset", dataset.name)
        # Insert "n_pos" and "n_neg" columns with the dataset's prompt settings
        results.insert(2, "n_pos", 0)  # no positive prompts for U-Net
        results.insert(3, "n_neg", 0)  # no negative prompts for U-Net
        results.insert(4, "bbsize", 0)  # no bounding box size for U-Net
        results.insert(5, "Mode", "None")  # U-Net uses no prompts
        results.insert(6, "Prompt Finder", "None")

        return results

if __name__ == "__main__":
    # Load the model
    predictor = BAGLS_U_Net_Predictor(model_path=os.path.join(os.path.dirname(__file__), "bagls_unet.pth"))

    # Load an image
    rand = np.random.randint(0, 3500)
    image = cv2.imread(f"Datasets/BAGLS/test/{rand}.png")
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ground_truth = cv2.imread(
        f"Datasets/BAGLS/test/{rand}_seg.png", cv2.IMREAD_GRAYSCALE
    )

    # Predict the segmentation
    seg = predictor.predict(image)

    dsc = dice_score(ground_truth, seg)
    iou = intersection_over_union(ground_truth, seg)
    print(f"Dice Score: {dsc:.3f}")
    print(f"IoU: {iou:.3f}")

    # Display the original image and the predicted segmentation
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap="gray")
    plt.title(f"Original Image #{rand}")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(seg, cmap="gray")
    plt.title(f"Predicted Segmentation - DSC: {dsc:.3f}")
    plt.axis("off")
    plt.show()
    plt.imshow(0.5 * image_gray + 0.5 * seg, cmap="gray")
    plt.title(f"Overlay - DSC: {dsc:.3f}")
    plt.axis("off")
    plt.show()

