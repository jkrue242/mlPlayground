import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np 
from .Conv import Conv
from .Down import Down 
from .Up import Up

#====================================================
class UNet(nn.Module):

    #====================================================
    def __init__(self, channels: int, classes: int):
        super(UNet, self).__init__()

        # unet 
        self.input = nn.Sequential(
            Conv(channels, 64),
            Conv(64, 64)
        )

        # down 
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        # up
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)

        self.output = nn.Conv2d(64, classes, kernel_size=1)

    #====================================================
    def forward(self, x: torch.Tensor):
        x1 = self.input(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # up blocks have skip connections
        x6 = self.up1(x5, x4)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8, x1)

        y = self.output(x9)
        return y