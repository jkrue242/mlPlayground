import torch
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np
from .Conv import Conv

#====================================================
class Up(nn.Module):
    
    #====================================================
    def __init__(self, in_channels: int, out_channels: int, upsample_kernel_size: int = 2, upsample_stride: int = 2, conv_kernel_size: int = 3, conv_padding: int = 1):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=upsample_kernel_size, stride=upsample_stride)
        self.conv = nn.Sequential(
            Conv(in_channels=2*out_channels, out_channels=out_channels, kernel_size=conv_kernel_size, padding=conv_padding),
            Conv(in_channels=out_channels, out_channels=out_channels, kernel_size=conv_kernel_size, padding=conv_padding)
        )

    #====================================================
    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        x1 = self.up(x1)
        dY = x2.size()[2] - x1.size()[2]
        dX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [dX // 2, dX - dX // 2, dY // 2, dY - dY // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x