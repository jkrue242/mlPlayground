import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np 

#====================================================
class Decoder(nn.Module):

    #====================================================
    def __init__(self, embedding_dimension: int = 384, patch_size: int = 14, out_channels: int = 1, decoder_dimension: int = 256):
        super(Decoder, self).__init__()
        self.proj = nn.Linear(embedding_dimension, decoder_dimension)
        self.upsample = nn.Sequential(

            # basically same as unet decoder
            nn.ConvTranspose2d(decoder_dimension, 128, kernel_size=patch_size, stride=patch_size),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, out_channels, kernel_size=1)
        )

    #====================================================
    def forward(self, x: torch.Tensor, target_h: int, target_w: int):
        B, N, C = x.shape

        x = self.proj(x) # embedding projection 
        num_patches_h = int(N**0.5)
        num_patches_w = num_patches_h

        x = x.transpose(1, 2).reshape(B, -1, num_patches_h, num_patches_w)
        x = self.upsample(x)

        if x.shape[2] != target_h or x.shape[3] != target_w:
            x = F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=False)

        return x