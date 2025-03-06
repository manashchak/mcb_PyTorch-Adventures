import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):

    """
    Sin/Cosine (non-learnable) encodings proposed in Attention is All You Need

    Args:
        max_len: Maximum number of tokens possible in a sequence
        embed_dim: Embedding dimension of each token
    """

    def __init__(self, max_len, time_embed_start_dim, time_embed_proj_dim, requires_grad=False):
        super(PositionalEncoding, self).__init__()

        self.max_len = max_len
        self.time_embed_start_dim = time_embed_start_dim
        self.scaled_time_embed_dim = time_embed_proj_dim
        self.requires_grad = requires_grad

        self.encodings = self._build_positional_encodings()

        self.time_mlp = nn.Sequential(

            nn.Linear(time_embed_start_dim, time_embed_proj_dim),
            nn.SiLU(),
            nn.Linear(time_embed_proj_dim, time_embed_proj_dim),
            nn.SiLU()
            
        )

    def _build_positional_encodings(self):

        encoding = torch.zeros(self.max_len, self.time_embed_start_dim, dtype=torch.float)
        postion_idx = torch.arange(0, self.max_len, dtype=torch.float).reshape(-1,1)
        embed_dim_skip_idx = torch.arange(0, self.time_embed_start_dim, step=2, dtype=torch.float)
        
        encoding[:, 0::2] = torch.sin(postion_idx / (10000 ** (embed_dim_skip_idx / self.time_embed_start_dim)))
        encoding[:, 1::2] = torch.cos(postion_idx / (10000 ** (embed_dim_skip_idx / self.time_embed_start_dim)))

        sincos_emb = nn.Parameter(encoding, requires_grad=self.requires_grad)
        
        encoding = nn.Embedding(self.max_len, self.time_embed_start_dim)
        encoding.weight = sincos_emb
        
        return encoding

    def forward(self, timestep_idx):

        time_embeddings = self.encodings(timestep_idx)

        time_embeddings = self.time_mlp(time_embeddings)

        return time_embeddings
    
class ClassConditionalEmbeddings(nn.Module):
    
    """
    Class conditional generation (where each class is identified as a index) 
    """
    
    def __init__(self, num_classes, embed_dim):

        self.num_classes = num_classes
        self.embed_dim = embed_dim

        self.class_embeddings = nn.Embedding(num_classes, embed_dim)
        self.unconditional_embedding = nn.Parameter(torch.randn(embed_dim))

        self.proj = nn.Sequential(
            
            nn.Linear(embed_dim, embed_dim), 
            nn.SiLU(), 
            nn.Linear(embed_dim, embed_dim), 
            nn.SiLU()

        )

    def forward(self, x):

        embeddings = self.class_embeddings(x)
        embeddings = self.proj(embeddings)
        
        return embeddings