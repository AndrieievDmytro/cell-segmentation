import torch
import torch.nn as nn
import torch.optim as optim
from .u_net import UNet
from torch.utils.data import DataLoader
from .tools import SegmentationDataset, save_model_with_metadata
from pathlib import Path

def train_model(train_normalized_data, train_mask, train_params):

    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    # Initialize training parameters
    image_size          = train_params["image_size"]
    batch_size          = train_params["batch_size"]
    epochs              = train_params["epochs"]
    learning_rate       = train_params["learning_rate"]
    num_classes         = train_params["num_classes"]
    train_images_dir    = Path(train_normalized_data) / "train"
    train_masks_dir     = Path(train_mask) / "train"
    val_images_dir      = Path(train_normalized_data) / "val"
    val_masks_dir       = Path(train_mask) / "val"
    model_folder        = train_params["model_folder"]
    
    print(f"Training parameters: Image size: {image_size}, Batch size: {batch_size}, "
            f"Epochs: {epochs}, Learning rate: {learning_rate}, Number of classes: {num_classes}, "
                f"Training image path: {train_images_dir}, Training mask path: {train_masks_dir}, "
                    f"Validation image path: {val_images_dir}, Validation mask path: {val_masks_dir}, "
                        f"Output model folder: {model_folder}.")

    # Initialize the model
    model = UNet(num_classes=num_classes)
    model = model.to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # Multi-class segmentation
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create Dataset Instances
    train_dataset = SegmentationDataset(train_images_dir, train_masks_dir)
    val_dataset = SegmentationDataset(val_images_dir, val_masks_dir)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for images, masks in train_loader:  # FIX: Removed `_,`
            images, masks = images.to(device), masks.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)  # FIX: `masks.long()` not needed (already `long` in dataset)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Save the trained model and training parameters
    save_dir = model_folder
    model_path, _ = save_model_with_metadata(model, train_params, save_dir)

    return model_path