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
        mask = tiff.imread(mask_path)

        # Ensure correct shape for image
        image = torch.tensor(image, dtype=torch.float32)
        if image.ndim == 2:  
            image = image.unsqueeze(0)  # Ensure shape is (C, H, W)

        # Mask remains unchanged (1-8 class values are correct)
        mask = torch.tensor(mask, dtype=torch.long)

        return image, mask


# def compute_metrics(pred, target, num_classes=9):
#     smooth = 1e-6

#     # Convert predictions to class indices (argmax)
#     pred = torch.argmax(pred, dim=1)

#     iou_per_class = []
#     dice_per_class = []

#     for class_idx in range(1, num_classes): 
#         pred_class = (pred == class_idx).float()
#         target_class = (target == class_idx).float()

#         intersection = (pred_class * target_class).sum()
#         union = pred_class.sum() + target_class.sum() - intersection
#         iou = (intersection + smooth) / (union + smooth)
#         dice = (2 * intersection + smooth) / (pred_class.sum() + target_class.sum() + smooth)

#         iou_per_class.append(iou.item())
#         dice_per_class.append(dice.item())

#     return iou_per_class, dice_per_class


# Save Model & Training Metadata
def save_model_with_metadata(model, optimizer, scheduler, parameters, epoch, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(save_dir, f"unet_model_{timestamp}.pth")

    # Save full checkpoint (model + optimizer + scheduler + epoch)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),  #  Save scheduler state
        "parameters": parameters
    }
    torch.save(checkpoint, model_path)

    # Save training parameters separately
    parameters_path = os.path.join(save_dir, f"training_parameters_{timestamp}.json")
    with open(parameters_path, "w") as f:
        json.dump(parameters, f, indent=4)

    print(f"Model saved at: {model_path}")
    print(f"Training parameters saved at: {parameters_path}")

    return model_path, parameters_path
