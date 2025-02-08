from .metrics import dice_coefficient, iou, pixel_accuracy, precision_recall
import torch
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # Progress bar
# from torchvision.utils import save_image


def evaluate_model(model, dataloader, device, save_dir="segmentation_results", metrics_file="metrics.json"):
    model.eval()  # Set model to evaluation mode
    os.makedirs(save_dir, exist_ok=True)  # Create directory if not exists

    dice_scores, iou_scores, pixel_accuracies, precisions, recalls = [], [], [], [], []

    with torch.no_grad():
        for idx, (images, masks) in enumerate(tqdm(dataloader, desc="Validating")):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)  # Convert logits to class labels

            # Compute metrics
            dice_scores.append(dice_coefficient(outputs, masks))
            iou_scores.append(iou(outputs, masks))
            pixel_accuracies.append(pixel_accuracy(outputs, masks))
            prec, rec = precision_recall(outputs, masks)
            precisions.append(prec)
            recalls.append(rec)

            # Convert tensors to NumPy for visualization
            img_np = images.cpu().squeeze().numpy()
            mask_np = masks.cpu().squeeze().numpy()
            pred_np = preds.cpu().squeeze().numpy()

            # Save individual images
            save_path_img = os.path.join(save_dir, f"image_{idx}.png")
            save_path_mask = os.path.join(save_dir, f"mask_{idx}.png")
            save_path_pred = os.path.join(save_dir, f"pred_{idx}.png")
            save_path_overlay = os.path.join(save_dir, f"overlay_{idx}.png")

            plt.imsave(save_path_img, img_np, cmap="gray")
            plt.imsave(save_path_mask, mask_np, cmap="gray")
            plt.imsave(save_path_pred, pred_np, cmap="gray")

            # Overlay prediction on input image
            fig, ax = plt.subplots(1, 3, figsize=(12, 4))
            ax[0].imshow(img_np, cmap="gray")
            ax[0].set_title("Input Image")
            ax[1].imshow(mask_np, cmap="gray")
            ax[1].set_title("Ground Truth Mask")
            ax[2].imshow(img_np, cmap="gray")
            ax[2].imshow(pred_np, cmap="jet", alpha=0.5)  # Overlay with transparency
            ax[2].set_title("Overlay Prediction")
            plt.savefig(save_path_overlay)
            plt.close()

    # Compute average metrics
    metrics = {
        "Dice Coefficient": sum(dice_scores) / len(dice_scores),
        "IoU": sum(iou_scores) / len(iou_scores),
        "Pixel Accuracy": sum(pixel_accuracies) / len(pixel_accuracies),
        "Precision": sum(precisions) / len(precisions),
        "Recall": sum(recalls) / len(recalls),
    }

    # Save metrics to a JSON file
    metrics_path = os.path.join(save_dir, metrics_file)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"\nMetrics saved at: {metrics_path}")
