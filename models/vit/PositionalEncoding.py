import torch 
import torch.nn as nn 
import torch.nn.functional as F 


#====================================================
class PositionalEmbedding(nn.Module):

    #====================================================
    def __init__(self, num_patches: int = 14, embedding_dimension: int = 384):
        super(PositionalEmbedding, self).__init__()
        self.embedding = nn.Parameter(torch.zeros(1, num_patches, embedding_dimension)) # learnable parameter
        self.dropout = nn.Dropout(0.1)
        nn.init.trunc_normal_(self.embedding, std=0.02)

    #====================================================
    def forward(self, x:torch.Tensor):
        x = x + self.embedding
        x = self.dropout(x)
        return x