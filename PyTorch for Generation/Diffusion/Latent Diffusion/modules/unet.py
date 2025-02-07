import torch
import torch.nn as nn

class UNET(nn.Module):
    def __init__(self, config):
        super(UNET, self).__init__()