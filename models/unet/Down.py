import torch
import torch.nn as nn 
import numpy as np
from .Conv import Conv

#====================================================
class Down(nn.Module):
    
    #====================================================
    def __init__(self, in_channels: int, out_channels: int, pool_kernel_size: int = 2, conv_kernel_size: int = 3, conv_padding: int = 1):
        super(Down, self).__init__()
        self.layer = nn.Sequential(
            nn.MaxPool2d(kernel_size=pool_kernel_size),
            Conv(in_channels=in_channels, out_channels=out_channels, kernel_size=conv_kernel_size, padding=conv_padding),
            Conv(in_channels=out_channels, out_channels=out_channels, kernel_size=conv_kernel_size, padding=conv_padding)
        )

    #====================================================
    def forward(self, x: torch.Tensor):
        return self.layer(x)
