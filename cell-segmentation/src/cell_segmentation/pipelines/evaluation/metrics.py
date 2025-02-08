import torch
import torch.nn.functional as F

def dice_coefficient(preds, targets, smooth=1e-6):
    preds = torch.argmax(F.softmax(preds, dim=1), dim=1)  # Convert logits to class labels
    targets = targets.long()  # Ensure target is long type
    intersection = (preds * targets).sum(dim=(1, 2))  # Sum over H and W
    union = preds.sum(dim=(1, 2)) + targets.sum(dim=(1, 2))  # Sum for both masks
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean().item()

def iou(preds, targets, smooth=1e-6):
    preds = torch.argmax(F.softmax(preds, dim=1), dim=1)
    targets = targets.long()
    intersection = (preds & targets).sum(dim=(1, 2))
    union = (preds | targets).sum(dim=(1, 2))
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()

def pixel_accuracy(preds, targets):
    preds = torch.argmax(F.softmax(preds, dim=1), dim=1)
    correct = (preds == targets).float().sum()
    total = targets.numel()
    return correct / total

def precision_recall(preds, targets, smooth=1e-6):
    preds = torch.argmax(F.softmax(preds, dim=1), dim=1)
    TP = (preds * targets).sum(dim=(1, 2))  # True Positives
    FP = (preds * (1 - targets)).sum(dim=(1, 2))  # False Positives
    FN = ((1 - preds) * targets).sum(dim=(1, 2))  # False Negatives
    precision = (TP + smooth) / (TP + FP + smooth)
    recall = (TP + smooth) / (TP + FN + smooth)
    return precision.mean().item(), recall.mean().item()
