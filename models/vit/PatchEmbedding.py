import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np

#====================================================
class PatchEmbedding(nn.Module):
    
    #====================================================
    def __init__(self, image_size: int = 640, patch_size: int = 14, channels: int = 3, embedding_dimension: int = 384):
        super(PatchEmbedding, self).__init__()
        self.patches = (image_size // patch_size) * (image_size // patch_size)

        # take patches of the image via convolution. note these do not overlap
        self.conv = nn.Conv2d(in_channels=channels, out_channels=embedding_dimension, kernel_size=patch_size, stride=patch_size)
        self.normalize = nn.LayerNorm(embedding_dimension)

    #====================================================
    def forward(self, x: torch.Tensor):
        # expect the input in the form of BCHW
        B, C, H, W = x.shape 
        x_patch = self.conv(x)
        x_flat = x_patch.flatten(2).transpose(1, 2)
        return self.normalize(x_flat)