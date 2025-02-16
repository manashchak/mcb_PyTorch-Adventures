import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):

    """
    Sin/Cosine (non-learnable) encodings proposed in Attention is All You Need

    Args:
        max_len: Maximum number of tokens possible in a sequence
        embed_dim: Embedding dimension of each token
    """

    def __init__(self, max_len, embed_dim, requires_grad=False):
        super(PositionalEncoding, self).__init__()

        self.max_len = max_len
        self.embed_dim = embed_dim
        self.requires_grad = requires_grad

        self.encodings = self._build_positional_encodings()

    def _build_positional_encodings(self):

        encoding = torch.zeros(self.max_len, self.embed_dim, dtype=torch.float)
        postion_idx = torch.arange(0, self.max_len, dtype=torch.float).reshape(-1,1)
        embed_dim_skip_idx = torch.arange(0, self.embed_dim, step=2, dtype=torch.float)
        
        encoding[:, 0::2] = torch.sin(postion_idx / (10000 ** (embed_dim_skip_idx / self.embed_dim)))
        encoding[:, 1::2] = torch.cos(postion_idx / (10000 ** (embed_dim_skip_idx / self.embed_dim)))

        encoding = nn.Parameter(encoding, requires_grad=self.requires_grad)

        return encoding
    
class ClassConditionalEmbeddings(nn.Module):
    
    """
    Class conditional generation (where each class is identified as a index) 
    """
    pass