import torch 
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np 

#====================================================
class MultiHeadSelfAttention(nn.Module):

    #====================================================
    def __init__(self, embedding_dimension: int = 384, n_heads: int = 6, dropout: float = 0.1):
        super(MultiHeadSelfAttention, self).__init__()
        assert embedding_dimension % n_heads == 0, "Embedding dimension needs to be a multiple of the number of attention heads"

        self.n_heads = n_heads
        self.head_dimension = embedding_dimension // self.n_heads
        self.scale = self.head_dimension ** -0.5
        self.qkv = nn.Linear(embedding_dimension, embedding_dimension*3, bias=False)
        self.proj = nn.Linear(embedding_dimension, embedding_dimension)
        self.dropout = nn.Dropout(dropout)

    #====================================================
    def forward(self, x: torch.Tensor):
        B, N, C = x.shape

        # linear projection splits into query key value vectors
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dimension).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # attention score via dot product 
        attention = (q @ k.transpose(-2, -1)) * self.scale # (B, num_heads, N, N)
        attention = attention.softmax(dim=-1)
        attention = self.dropout(attention)

        # weighted combination of value vectors applies attention to the values
        x = (attention @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x 