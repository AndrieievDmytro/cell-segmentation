import torch
import numpy as np
import os
from torch.utils.data import Dataset
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
            exist = os.path.exists(mask_path) 
            if exist :  # Check if corresponding mask exists
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