import torch
import torch.nn as nn
import torch.optim as optim
from .u_net import UNet
from torch.utils.data import DataLoader
from .tools import SegmentationDataset, save_model_with_metadata
from pathlib import Path
from torch.amp import autocast, GradScaler
from .deeplab_v3plus import DeepLabV3Plus

def train_model(train_normalized_data, train_mask, train_params):

    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.backends.cudnn.benchmark = True

    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    # Initialize training parameters
    image_size                = train_params["image_size"]
    batch_size                = train_params["batch_size"]
    epochs                    = train_params["epochs"]
    learning_rate             = train_params["learning_rate"]
    num_classes               = train_params["num_classes"]
    train_images_dir          = Path(train_normalized_data) / "train"
    train_masks_dir           = Path(train_mask) / "train"
    val_images_dir            = Path(train_normalized_data) / "val"
    val_masks_dir             = Path(train_mask) / "val"
    model_folder_deeplab      = train_params["model_folder_deeplab"]
    backbone                  = train_params["backbone"]
    
    print(f"Training parameters:\n"
            f"  Image size: {image_size}\n"
            f"  Batch size: {batch_size}\n"
            f"  Epochs: {epochs}\n"
            f"  Learning rate: {learning_rate}\n"
            f"  Number of classes: {num_classes}\n"
            f"  Training image path: {train_images_dir}\n"
            f"  Training mask path: {train_masks_dir}\n"
            f"  Validation image path: {val_images_dir}\n"
            f"  Validation mask path: {val_masks_dir}\n"
            f"  Output model folder: {model_folder_deeplab}")


    # Initialize the model
    # model = UNet(num_classes=num_classes)
    model = DeepLabV3Plus(num_classes=num_classes, backbone=backbone)
    model = model.to(device)

    # Define the loss function and optimizer
    # criterion = nn.CrossEntropyLoss(ignore_index=0)  # Multi-class segmentation
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Load checkpoint if available
    latest_checkpoint_path = Path(model_folder_deeplab) / "latest_checkpoint.pth"
    if latest_checkpoint_path.exists():
        checkpoint = torch.load(latest_checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"]
        best_val_loss = checkpoint["best_val_loss"]
        print(f"Resuming training from Epoch {start_epoch}, Best Val Loss: {best_val_loss:.4f}")
    else:
        start_epoch = 0
        best_val_loss = float("inf")

    # Create Dataset Instances
    train_dataset = SegmentationDataset(train_images_dir, train_masks_dir)
    val_dataset = SegmentationDataset(val_images_dir, val_masks_dir)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,  
        pin_memory=True,  
        prefetch_factor=2
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True, 
        prefetch_factor=2
    )

    scaler = GradScaler()

    for epoch in range(start_epoch, epochs):  # Resume from the last saved epoch
        model.train()
        train_loss = 0
        num_train_batches = len(train_loader)  # Total number of training batches

        for images, masks in train_loader:
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)

            optimizer.zero_grad()

            with autocast(device_type=device.type):  # FP16 computations
                outputs = model(images)
                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        train_loss = train_loss / num_train_batches if num_train_batches > 0 else 0    # Average over number of batches

        # Validation loop
        model.eval()
        val_loss = 0
        num_val_batches = len(val_loader) 

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
                
                with autocast(device_type=device.type):  # Enable AMP in inference for efficiency
                    outputs = model(images)
                    loss = criterion(outputs, masks)
            
                val_loss += loss.item()
            
        val_loss = val_loss / num_val_batches if num_val_batches > 0 else 0

        scheduler.step()
        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save the trained model and training parameters
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_dir = model_folder_deeplab
            model_path, _ = save_model_with_metadata(model, optimizer, scheduler, train_params, epoch + 1, save_dir)
            print(f"New best model saved with Val Loss: {best_val_loss:.4f}")
        
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss
        }
        torch.save(checkpoint, f"{model_folder_deeplab}/latest_checkpoint.pth")  # Save full checkpoint

    
    return model_path