from statistics import mean

import numpy as np
import torch


def dice_loss(pred, target_mask, eps=1e-6):
    axes = (2, 3)
    numerator = 2 * torch.sum(pred * target_mask, axes)
    denominator = torch.sum(torch.square(pred) + torch.square(target_mask), axes)
    return 1 - torch.mean(torch.div(numerator, denominator))


def iou(preds, labels, C, EMPTY=1., ignore=None, per_image=False):
    """
    Array of IoU for each (non ignored) class
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        iou = []
        for i in range(C):
            if i != ignore:  # The ignored label is sometimes among predicted classes (ENet - CityScapes)
                intersection = ((label == i) & (pred == i)).sum()
                union = ((label == i) | ((pred == i) & (label != ignore))).sum()
                if not union:
                    iou.append(EMPTY)
                else:
                    iou.append(float(intersection) / float(union))
        ious.append(iou)
    ious = [mean(iou) for iou in zip(*ious)]  # mean accross images if per_image
    return 100 * np.array(ious)
