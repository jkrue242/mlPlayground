import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np
from .Decoder import Decoder
from .MultiHeadSelfAttention import MultiHeadSelfAttention
from .PatchEmbedding import PatchEmbedding
from .PositionalEncoding import PositionalEmbedding
from .Transformer import Transformer

#====================================================
class VisionTransformer(nn.Module):

    #====================================================
    def __init__(self, in_channels: int = 3, out_channels: int = 1, image_size: int = 640, transformer_depth: int = 6, embedding_dimension: int = 384, patch_size: int = 14):
        super(VisionTransformer, self).__init__()
        
        # patches for positional encoding
        num_patches = (image_size // patch_size) ** 2
        
        self.patch_embedding = PatchEmbedding(image_size=image_size, patch_size=patch_size, channels=in_channels, embedding_dimension=embedding_dimension)
        self.positional_embedding = PositionalEmbedding(num_patches=num_patches, embedding_dimension=embedding_dimension)
        
        self.transformer = nn.Sequential(
            *[Transformer(embedding_dimension=embedding_dimension) for _ in range(transformer_depth)]
        )
        self.decoder = Decoder(embedding_dimension=embedding_dimension, patch_size=patch_size, out_channels=out_channels)
        self.norm = nn.LayerNorm(embedding_dimension)

    #====================================================
    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape 
        x = self.patch_embedding(x)
        x = self.positional_embedding(x)
        x = self.transformer(x)
        x = self.norm(x)
        x = self.decoder(x, H, W)
        return x 