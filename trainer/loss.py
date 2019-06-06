import torch
import torch.nn as nn

def cross_entropy_loss(device, weights = None):
    return nn.CrossEntropyLoss(torch.Tensor(weights).to(device))