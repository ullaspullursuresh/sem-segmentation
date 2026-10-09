import torch

def calculate_iou(preds, targets, smooth=1e-6):
    preds = (preds > 0.5).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()

def calculate_dice(preds, targets, smooth=1e-6):
    preds = (preds > 0.5).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    
    dice = (2. * intersection + smooth) / (preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + smooth)
    return dice.mean().item()