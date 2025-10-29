from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from torch.optim import lr_scheduler
from collections import defaultdict
import torch.nn.functional as F
import torch.optim as optim
import torch
import time
import copy
import numpy as np
import cv2

from segFM.predictors.bagls_unet import UNet


TRAIN_SET_SIZE = 1120
VAL_SET_SIZE = 112
NUM_EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 1e-4

class BAGLSDataset(Dataset):
    def __init__(self, num_samples, transform=None, type="training"):
        self.img_dir = f"Datasets/BAGLS/{type}/"
        self.mask_dir = f"Datasets/BAGLS/{type}/"
        self.transform = transform
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img_name = f"{self.img_dir}{idx}.png"
        mask_name = f"{self.mask_dir}{idx}_seg.png"

        image = datasets.folder.default_loader(img_name)
        mask = datasets.folder.default_loader(mask_name)

        # Resize them to 256 width and 512 height and convert to grayscale
        image = image.resize((256, 512)).convert("L")
        mask = mask.resize((256, 512)).convert("L")

        # Convert image and mask to tensors
        image = transforms.ToTensor()(image)
        mask = transforms.ToTensor()(mask)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


class UNetTrainer:
    def __init__(self, in_channels, out_channels, train_set_size, val_set_size, batch_size, learning_rate, num_epochs):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.train_set_size = train_set_size
        self.val_set_size = val_set_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.loss_vals = {
            k: {
                "BCE": [],
                "DSC": [],
                "Combined": [],
            } for k in ["train", "val"]
        }

    def get_data_loaders(self):
        train_set = BAGLSDataset(self.train_set_size, type="training")
        val_set = BAGLSDataset(self.val_set_size, type="test")

        dataloaders = {
            "train": DataLoader(
                train_set, batch_size=self.batch_size, shuffle=True, num_workers=0
            ),
            "val": DataLoader(
                val_set, batch_size=self.batch_size, shuffle=True, num_workers=0
            ),
        }

        return dataloaders

    def dice_loss(self, pred, target, smooth=1.0):
        pred = pred.contiguous()
        target = target.contiguous()

        intersection = (pred * target).sum(dim=2).sum(dim=2)

        loss = 1 - (
            (2.0 * intersection + smooth)
            / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)
        )

        return loss.mean()

    def calc_loss(self, pred, target, metrics, phase, bce_weight=0.5):
        bce = F.binary_cross_entropy_with_logits(pred, target)

        pred = F.sigmoid(pred)
        dice = self.dice_loss(pred, target)

        # Combine BCE and Dice loss to a single loss
        loss = bce * bce_weight + dice * (1 - bce_weight)

        bce_np = bce.data.cpu().numpy()
        dice_np = dice.data.cpu().numpy()
        loss_np = loss.data.cpu().numpy()

        self.loss_vals[phase]["BCE"].append(bce_np.item())
        self.loss_vals[phase]["DSC"].append(dice_np.item())
        self.loss_vals[phase]["Combined"].append(loss_np.item())

        metrics["bce"] += bce_np * target.size(0) #Multiply by batch size to get the total loss for the batch
        metrics["dice"] += dice_np * target.size(0)
        metrics["loss"] += loss_np * target.size(0)

        return loss

    def print_metrics(self, metrics, epoch_samples, phase):
        outputs = []
        for k in metrics.keys():
            outputs.append(f"{k}: {metrics[k] / epoch_samples:4f}")

        print(f"{phase}: {", ".join(outputs)}")

    def train_and_save_model(self):
        # Use the GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize the UNet and move it to the device
        model = UNet(self.in_channels, self.out_channels).to(device)

        # Define the optimizer and the learning rate
        optimizer_ft = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=self.learning_rate
        )

        # Define the learning rate scheduler
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)

        # Train the model
        model = self._train_model(
            model, device, optimizer_ft, exp_lr_scheduler, num_epochs=self.num_epochs
        )

        # Save it to a file
        NAME = "bagls_unet.pth"
        torch.save(model.state_dict(), NAME)
        print(f"Model saved as {NAME}")

    def _train_model(self, model, device, optimizer, scheduler, num_epochs):
        # Load the training and validation datasets
        dataloaders = self.get_data_loaders()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_loss = 1e10

        # Training loop
        for epoch in range(num_epochs):
            print("Epoch {}/{}".format(epoch, num_epochs - 1))
            print("-" * 10)

            since = time.time()

            # Each epoch has a training and validation phase
            for phase in ["train", "val"]:
                if phase == "train":
                    scheduler.step()
                    for param_group in optimizer.param_groups:
                        print("LR", param_group["lr"])

                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                metrics = defaultdict(float)
                epoch_samples = 0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs)

                        # Loss function is binary cross entropy + dice loss
                        loss = self.calc_loss(outputs, labels, metrics, phase)

                        # backward + optimize only if in training phase
                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    # statistics
                    epoch_samples += inputs.size(0)

                self.print_metrics(metrics, epoch_samples, phase)
                epoch_loss = metrics["loss"] / epoch_samples

                # deep copy the model
                if phase == "val" and epoch_loss < best_loss:
                    print("saving best model")
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())

            time_per_epoch = time.time() - since
            print("Time per epoch: {:.0f}m {:.0f}s".format(time_per_epoch // 60, time_per_epoch % 60))
            print(
                "Approximate time remaining: {:.0f}m {:.0f}s".format(
                    (num_epochs - epoch - 1) * time_per_epoch // 60,
                    (num_epochs - epoch - 1) * time_per_epoch % 60,
                )
            )

        print("Best val loss: {:4f}".format(best_loss))

        # load best model weights
        model.load_state_dict(best_model_wts)

        self.plot_loss_vals_per_epoch()

        return model
    
    def plot_loss_vals_per_epoch(self):
        # Construct a dataframe from the loss values
        import pandas as pd
        loss_data = {
            "train": {
                "BCE": self.loss_vals["train"]["BCE"],
                "DSC": self.loss_vals["train"]["DSC"],
                "Combined": self.loss_vals["train"]["Combined"],
            },
            "val": {
                "BCE": self.loss_vals["val"]["BCE"],
                "DSC": self.loss_vals["val"]["DSC"],
                "Combined": self.loss_vals["val"]["Combined"],
            },
        }
        df = pd.DataFrame(loss_data)
        df.to_csv("loss_values.csv", index=False)
        import matplotlib.pyplot as plt

        # Smooth the loss values
        for phase in self.loss_vals.keys():
            self.loss_vals[phase]["BCE"] = np.convolve(
                self.loss_vals[phase]["BCE"], np.ones(15) / 15, mode="valid"
            )
            self.loss_vals[phase]["DSC"] = np.convolve(
                self.loss_vals[phase]["DSC"], np.ones(15) / 15, mode="valid"
            )
            self.loss_vals[phase]["Combined"] = np.convolve(
                self.loss_vals[phase]["Combined"], np.ones(15) / 15, mode="valid"
            )

        # Calculate the factor by which to scale the x-axis
        for phase in self.loss_vals.keys():
            n_images = TRAIN_SET_SIZE if phase == "train" else VAL_SET_SIZE
            x_factor = np.ceil(n_images / BATCH_SIZE)
            # Scale the x-axis by the number of batches
            x_values = np.arange(len(self.loss_vals[phase]["BCE"])) / x_factor
            
            plt.figure(figsize=(10, 5))
            plt.plot(x_values, self.loss_vals[phase]["BCE"], label="BCE Loss")
            plt.plot(x_values, self.loss_vals[phase]["DSC"], label="Dice Loss")
            plt.plot(x_values, self.loss_vals[phase]["Combined"], label="Combined Loss")
            plt.title(f"{phase.capitalize()} Loss per Epoch")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.show()

if __name__ == "__main__":
    trainer = UNetTrainer(
        in_channels=1,
        out_channels=1,
        train_set_size=TRAIN_SET_SIZE,
        val_set_size=VAL_SET_SIZE,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        num_epochs=NUM_EPOCHS,
    )
    
    trainer.train_and_save_model()
