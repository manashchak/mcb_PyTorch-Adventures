import torch 
import torch.nn as nn

def sincos_embeddings(num_tokens, embed_dim, requires_grad=False):

    """
    Sin/Cosine (non-learnable) encodings proposed in Attention is All You Need

    Args:
        num_tokens: Sequence length of image patches
        embed_dim: Embedding dimension of each token
    """

    ### Create Tensors for Position and Embedding Idx ### 
    encoding = torch.zeros(num_tokens, embed_dim, dtype=torch.float)
    position_idx = torch.arange(0, num_tokens, dtype=torch.float).reshape(-1,1)
    embed_dim_skip_idx = torch.arange(0, embed_dim, step=2, dtype=torch.float)
    
    ### Attention is All You Need Pos Embed Formula ###
    encoding[:, 0::2] = torch.sin(position_idx / (10000 ** (embed_dim_skip_idx / embed_dim)))
    encoding[:, 1::2] = torch.cos(position_idx / (10000 ** (embed_dim_skip_idx / embed_dim)))
    
    ### Add Batch Dimension ###
    encoding = encoding.unsqueeze(0)

    ### Convert to Parameter ###
    encoding = nn.Parameter(encoding, requires_grad=requires_grad)

    return encoding

def random_masking(x, mask_ratio=0.75):

    """
    Random Masking Procedure as outlined in MAE Paper. 75% of image tokens
    are randomly removed for each image. We return both the masked image, 
    the mask used, and the indexes needed to restore the masked image

    Args:
        x: Input image embeddings (Batch x Num Tokens x Embed Dim)
        mask_ratio: Proportion of image tokens to mask

    """

    batch_size, seq_len, embed_dim = x.shape

    ### Number of Tokens to Keep After Masking ###
    num_tokens_to_keep = int(seq_len * (1-mask_ratio))
    
    ### Generate Noise for Sampling ###
    noise = torch.rand((batch_size, seq_len), device=x.device)
    
    ### Argsort Noise, and keep only the first num_tokens_to_keep ###
    sorted_idx = torch.argsort(noise, dim=1)
    restore_idx = torch.argsort(sorted_idx, dim=1)
    selected_idx = sorted_idx[:, :num_tokens_to_keep]
    
    ### Add Embedding Dimension to Selected Idx (repeating) ###
    ### Documentation shows x and index must have same number of dimensions ###
    ### https://pytorch.org/docs/main/generated/torch.gather.html
    selected_idx_repeat = selected_idx.unsqueeze(-1).repeat(1,1,embed_dim)

    ### Gather the selected indexes from X (removing all unselected image embeddings) ###
    x_masked = torch.gather(x, dim=1, index=selected_idx_repeat)
    
    ### Create Mask (indicating which indexes were selected to be kept or removed) ###
    mask = torch.ones([batch_size, seq_len], device=x.device)
    mask[:, :num_tokens_to_keep] = 0

    ### Use the restore idx to indicate the true indexes selected to be masked ###
    mask = torch.gather(mask, dim=1, index=restore_idx)

    return x_masked, mask, restore_idx
    
def patchify(images, image_size=224, num_channels=3, patch_size=16):

    """
    Helper function to cut images into patches.

    images: (B x 3 x 224 x 224)
    output: (B x 196 x 768)
    """

    batch_size = images.shape[0]

    ### Compute the Patched Grid Dimension (Num patches along the H and W) ###
    num_patches = image_size // patch_size

    ### Cut Images into Patches (B x C x 14 x 16 x 14 x 16) ###
    patched = images.reshape(batch_size, 
                             num_channels, 
                             num_patches, 
                             patch_size, 
                             num_patches,
                             patch_size)
    
    ### Permute Dimensions (B x 14 x 14 x 16 x 16 x 3) ###
    patched = patched.permute(0,2,4,3,5,1)
    
    ### Merge Dimensions Together (B x 196 x 768) ###
    patched = patched.reshape(batch_size, num_patches**2, num_channels * patch_size**2)
    
    return patched

        