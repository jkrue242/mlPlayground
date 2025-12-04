import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np 
from .MultiHeadSelfAttention import MultiHeadSelfAttention


#====================================================
class Transformer(nn.Module):

    #====================================================
    def __init__(self, embedding_dimension: int = 384, n_heads: int = 6, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super(Transformer, self).__init__()

        self.norm1 = nn.LayerNorm(embedding_dimension)
        self.attention = MultiHeadSelfAttention(embedding_dimension=embedding_dimension, n_heads=n_heads, dropout=dropout)

        self.norm2 = nn.LayerNorm(embedding_dimension)
        hidden_dimension = int(embedding_dimension * mlp_ratio)

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dimension, hidden_dimension),
            nn.GELU(), # gaussian linear error unit
            nn.Dropout(dropout),
            nn.Linear(hidden_dimension, embedding_dimension),
            nn.Dropout(dropout)
        )

    #====================================================
    def forward(self, x: torch.Tensor):
        x_residual = x
        
        # attention with original
        x_norm = self.norm1(x)
        attention_out = self.attention(x_norm)
        
        # skip connection
        x = x_residual + attention_out
        x_residual = x
        
        # mlp with the attended x
        x_norm = self.norm2(x)
        mlp_out = self.mlp(x_norm)
        
        # skip connection
        x = x_residual + mlp_out
        
        return x