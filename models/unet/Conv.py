import torch
import torch.nn as nn 
import numpy as np

#====================================================
class Conv(nn.Module):
    
    #====================================================
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
        super(Conv, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    #====================================================
    def forward(self, x: torch.Tensor):
        return self.layer(x)
