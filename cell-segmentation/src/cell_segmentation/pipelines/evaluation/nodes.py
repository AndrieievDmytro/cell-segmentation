from .metrics import (
    dice_loss,
    iou_score,
    f1_score,
    mean_pixel_accuracy,
)
import torch
import os
import json
from datetime import datetime


def evaluate_model(model, test_data, evaluation_parameters):
    """Evaluate the model on the test dataset."""
    # Prepare device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Unpack test data
    x_test, y_test = test_data  # Assuming test_data is a tuple (images, masks)
    x_test, y_test = x_test.to(device), y_test.to(device)

    # Evaluate the model
    with torch.no_grad():
        y_pred = model(x_test)
        y_pred = torch.sigmoid(y_pred)  # For binary segmentation
        y_pred_binary = (y_pred > 0.5).float()

        # Calculate metrics
        results = {
            "Dice Coefficient": dice_loss(y_test, y_pred_binary).item(),
            "IoU": iou_score(y_test, y_pred_binary).item(),
            "F1-Score": f1_score(y_test, y_pred_binary).item(),
            "Mean Pixel Accuracy": mean_pixel_accuracy(y_test, y_pred_binary).item(),
        }

    # Save evaluation metrics
    save_dir = evaluation_parameters.get("save_dir", "evaluation_results")
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(save_dir, f"evaluation_metrics_{timestamp}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Evaluation metrics saved at: {results_path}")
    return results
