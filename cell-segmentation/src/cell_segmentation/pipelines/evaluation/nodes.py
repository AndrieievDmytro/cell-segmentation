from .metrics import dice_coefficient, iou, pixel_accuracy, precision_recall
import torch
import os
import json
import matplotlib.pyplot as plt
from tqdm import tqdm  
from .tools import SegmentationDataset
from torch.utils.data import DataLoader
from pathlib import Path
from ..model.u_net import UNet
from ..model.deeplab_v3plus import DeepLabV3Plus


def evaluate_model(evaluation_parameters):
    test_mask_path = evaluation_parameters["mask_output_folder_test"]
    test_normalized_path = evaluation_parameters["norm_output_folder_test"]
    batch_size = evaluation_parameters["batch_size"]
    model_name = evaluation_parameters["model_name"]

    test_mask_data = Path(test_mask_path)
    test_normilized_data = Path(test_normalized_path)

    if model_name == "u_net":
        save_dir = evaluation_parameters["unet_images_save_dir"]
        metrics_file = evaluation_parameters["unet_metrics_file_path"]
        model_path = evaluation_parameters["model_file_unet"]
        model = UNet()
    elif model_name == "deeplab":
        save_dir = evaluation_parameters["deeplab_images_save_dir"]
        metrics_file = evaluation_parameters["deeplab_metrics_file_path"]
        model_path = evaluation_parameters["model_file_deeplab"]
        model = DeepLabV3Plus(num_classes=9, backbone="resnet50")
    else:
        raise ValueError(f"Invalid model name: {model_name}. Expected 'u_net' or 'deeplab'.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    print(f"Using {'GPU: ' + torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")

    test_dataset = SegmentationDataset(test_normilized_data, test_mask_data)

    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,  
        pin_memory=True,  
        prefetch_factor=2
    )

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=False)
    model.to(device)
    model.eval()

    os.makedirs(save_dir, exist_ok=True)

    dice_scores, iou_scores, pixel_accuracies, precisions, recalls = [], [], [], [], []

    with torch.no_grad():
        for idx, (images, masks) in enumerate(tqdm(test_loader, desc="Validating")):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)

            preds = torch.argmax(outputs, dim=1)  # [batch_size, height, width]

            preds = preds.view(preds.shape[0], 512, 512)  

            dice_scores.append(dice_coefficient(preds, masks))
            iou_scores.append(iou(preds, masks))
            pixel_accuracies.append(pixel_accuracy(preds, masks))
            prec, rec = precision_recall(preds, masks)
            precisions.append(prec)
            recalls.append(rec)

            for i in range(images.size(0)):  
                img_np = images[i].cpu().squeeze().numpy()  # Convert tensor to NumPy
                mask_np = masks[i].cpu().squeeze().numpy()
                pred_np = preds[i].cpu().squeeze().numpy()

                pred_np = (pred_np - pred_np.min()) / (pred_np.max() - pred_np.min() + 1e-8)
                pred_np = (pred_np > 0.7).astype(float)

                img_id = f"{idx}_{i}"

                # Create a single figure with 3 subplots
                fig, ax = plt.subplots(1, 3, figsize=(12, 4))

                # Display the original image
                ax[0].imshow(img_np, cmap="gray")
                ax[0].set_title("Input Image")
                ax[0].axis("off")

                # Display the ground truth mask
                ax[1].imshow(mask_np, cmap="gray")
                ax[1].set_title("Ground Truth")
                ax[1].axis("off")

                # Display overlay: input image + predicted mask (with transparency)
                ax[2].imshow(img_np, cmap="gray")  # Base image
                ax[2].imshow(pred_np, cmap="Reds", alpha=0.5)  # Overlay prediction with transparency
                ax[2].set_title("Prediction Overlay")
                ax[2].axis("off")
                # Save the combined image
                save_path = os.path.join(save_dir, f"combined_{img_id}.png")

                plt.savefig(save_path, bbox_inches="tight", dpi=300)
                plt.close()

    num_samples = max(1, len(dice_scores))
    metrics = {
        "Dice Coefficient": sum(dice_scores) / num_samples,
        "IoU": sum(iou_scores) / num_samples,
        "Pixel Accuracy": sum(pixel_accuracies) / num_samples,
        "Precision": sum(precisions) / num_samples,
        "Recall": sum(recalls) / num_samples,
    }

    metrics_path = os.path.join(save_dir, metrics_file)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"\nMetrics saved at: {metrics_path}")
    
    return metrics

