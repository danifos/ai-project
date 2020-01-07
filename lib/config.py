import numpy as np
import torch

DEVICE = torch.device('cpu')

def to(**kwargs):
    device = kwargs.pop('device', DEVICE)
    DEVICE = device
