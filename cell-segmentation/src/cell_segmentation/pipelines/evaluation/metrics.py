import torch

def dice_coefficient(preds, targets, smooth=1e-6):
    assert preds.ndim == 3, f"Unexpected preds shape: {preds.shape}"

    if targets.ndim == 4 and targets.shape[1] > 1:
        targets = torch.argmax(targets, dim=1)
    targets = targets.long()

    assert preds.shape == targets.shape, f"Shape mismatch: preds {preds.shape}, targets {targets.shape}"

    intersection = (preds == targets).float().sum(dim=(1, 2))
    union = preds.sum(dim=(1, 2)) + targets.sum(dim=(1, 2))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.mean().item()

def iou(preds, targets, smooth=1e-6):
    assert preds.ndim == 3, f"Unexpected preds shape: {preds.shape}"

    targets = targets.long()
    
    intersection = (preds == targets).float().sum(dim=(1, 2))
    union = ((preds > 0) | (targets > 0)).float().sum(dim=(1, 2))
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()

def pixel_accuracy(preds, targets):
    assert preds.ndim == 3, f"Unexpected preds shape: {preds.shape}"

    correct = (preds == targets).float().sum()
    total = targets.numel()
    return (correct / total).item()

def precision_recall(preds, targets, smooth=1e-6):
    assert preds.ndim == 3, f"Unexpected preds shape: {preds.shape}"

    targets = targets.long()
    
    TP = (preds * targets).sum(dim=(1, 2))
    FP = (preds * (1 - targets)).sum(dim=(1, 2))
    FN = ((1 - preds) * targets).sum(dim=(1, 2))

    precision = (TP + smooth) / (TP + FP + smooth + 1e-8)
    recall = (TP + smooth) / (TP + FN + smooth + 1e-8)
    recall = recall.clamp(0, 1)  # Ensure recall is between 0 and 1

    return precision.mean().item(), recall.mean().item()
