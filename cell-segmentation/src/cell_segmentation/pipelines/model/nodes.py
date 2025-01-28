import torch
import torch.nn as nn
import torch.optim as optim
import os
from datetime import datetime
import json
from .u_net import UNet

def train_model(train_data, val_data, training_parameters):
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load preprocessed data (already prepared DataLoader)
    train_loader = train_data
    val_loader = val_data

    # Initialize the model
    model = UNet(
        input_shape=(1, training_parameters["image_size"], training_parameters["image_size"]),
        num_classes=training_parameters["num_classes"]
    )
    model = model.to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # Multi-class segmentation
    optimizer = optim.Adam(model.parameters(), lr=training_parameters["learning_rate"])

    # Training loop
    for epoch in range(training_parameters["epochs"]):
        model.train()
        train_loss = 0

        for images, masks, _ in train_loader:  # Assuming your DataLoader returns images, masks, and labels
            images, masks = images.to(device), masks.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks.long())  # Convert masks to long for CrossEntropyLoss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, masks, _ in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks.long())
                val_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{training_parameters['epochs']}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Save the trained model and training parameters
    save_dir = training_parameters.get("save_dir", "E:\\diploma_proj_latest\\cell-segmentation\\.data\\04_model")
    model_path, parameters_path = save_model_with_metadata(model, training_parameters, save_dir)

    # Return the path to the saved model
    return model_path


def save_model_with_metadata(model, parameters, save_dir):
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Get the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save the model with the timestamp in the filename
    model_path = os.path.join(save_dir, f"unet_model_{timestamp}.pth")
    torch.save(model.state_dict(), model_path)

    # Save the training parameters for future reference
    parameters_path = os.path.join(save_dir, f"training_parameters_{timestamp}.json")
    with open(parameters_path, "w") as f:
        json.dump(parameters, f, indent=4)

    print(f"Model saved at: {model_path}")
    print(f"Training parameters saved at: {parameters_path}")
    return model_path, parameters_path
