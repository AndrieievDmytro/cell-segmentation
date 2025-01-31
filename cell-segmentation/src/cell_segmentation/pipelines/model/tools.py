import torch
import numpy as np
import os
from torch.utils.data import Dataset
from datetime import datetime
import json
from pathlib import Path
import tifffile as tiff 

# Custom Dataset for Preprocessed Images and Masks

class SegmentationDataset(Dataset):

    def __init__(self, image_dir, mask_dir):
            self.image_dir = Path(image_dir)
            self.mask_dir = Path(mask_dir)

            # Recursively find all image and mask files
            self.image_paths = []
            self.mask_paths = []

            for image_path in self.image_dir.rglob("*.npy"):  # Recursively find .npy files
                mask_path = str(image_path).replace(str(self.image_dir), str(self.mask_dir)).replace(".npy", ".tif")  # Match .tif extension
                if os.path.exists(mask_path):  # Check if corresponding mask exists
                    self.image_paths.append(image_path)
                    self.mask_paths.append(Path(mask_path))  # Store as Path object

            print(f"Found {len(self.image_paths)} image-mask pairs.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]

        # Load preprocessed numpy image
        image = np.load(image_path)

        # Load .tif mask
        mask = tiff.imread(mask_path)

        # Convert mask from 255 to 1 if necessary
        mask = (mask == 255).astype(np.uint8)

        # Convert to PyTorch tensors
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Add channel dimension (C, H, W)
        mask = torch.tensor(mask, dtype=torch.long)  # Class indices for CrossEntropyLoss

        return image, mask


def compute_metrics(pred, target):
    smooth = 1e-6
    pred = (pred > 0.5).float()  # Binarize predictions
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    dice = (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return iou.item(), dice.item()

def save_model_with_metadata(model, parameters, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(save_dir, f"unet_model_{timestamp}.pth")
    torch.save(model.state_dict(), model_path)

    parameters_path = os.path.join(save_dir, f"training_parameters_{timestamp}.json")
    with open(parameters_path, "w") as f:
        json.dump(parameters, f, indent=4)

    print(f"Model saved at: {model_path}")
    print(f"Training parameters saved at: {parameters_path}")
    return model_path, parameters_path