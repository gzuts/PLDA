import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def accuracy_check(loader, model, device):
    with torch.no_grad():
        total, num_samples = 0, 0
        for images, labels in loader:
            labels, images = labels.to(device), images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += (predicted == labels).sum().item()
            num_samples += labels.size(0)
    return total / num_samples


def confidence_update(model, confidence, batchX, batchY, batch_index):
    with torch.no_grad():
        batch_outputs, features = model(batchX)
        temp_un_conf = F.softmax(batch_outputs, dim=1)
        confidence[batch_index, :] = (
            temp_un_conf * batchY
        )  # un_confidence stores the weight of each example
        # weight[batch_index] = 1.0/confidence[batch_index, :].sum(dim=1)
        base_value = confidence.sum(dim=1).unsqueeze(1).repeat(1, confidence.shape[1])
        confidence = confidence / base_value
    return confidence
