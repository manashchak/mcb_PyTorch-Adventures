import torch
import torch.nn as nn
from VisionTransformer import PatchEmbed, EncoderBlock
from utils import sincos_embeddings, random_masking
from dataclasses import dataclass

@dataclass
class MAEConfig:

    ### Input Image Config ###
    img_size: int = 224
    patch_size: int = 16
    in_chans: int = 3

    ### MAE Encoder Config ###
    encoder_embed_dim: int = 768
    encoder_depth: int = 12
    encoder_num_heads: int = 12
    encoder_attn_p: float = 0.0
    encoder_mlp_p: float = 0.0
    encoder_proj_p: float = 0.0
    encoder_mlp_ratio: int = 4

    ### MAE Decoder Config ###
    decoder_embed_dim: int = 512
    decoder_depth: int =  8
    decoder_num_heads: int = 16
    decoder_attn_p: float = 0.0
    decoder_mlp_p: float = 0.0
    decoder_proj_p: float = 0.0
    decoder_mlp_ratio: int = 4
    
    ### MAE Settings ###
    fused_attention: bool = True
    learnable_positional_encodings: bool = True


class VITMAEEncoder(nn.Module):
    def __init__(self, config):
        
        super(VITMAEEncoder, self).__init__()

        self.config = config

        ### Define Encoder ###
        self.patch_embed = PatchEmbed(img_size=config.img_size, 
                                      patch_size=config.patch_size, 
                                      in_chans=config.in_chans, 
                                      embed_dim=config.encoder_embed_dim)
        
        ### CLS Token and SinCos Positional Embeddings ###
        self.enc_cls_token = nn.Parameter(torch.zeros(1,1,config.encoder_embed_dim))
        self.enc_pos_embed = sincos_embeddings(num_tokens=self.patch_embed.num_patches+1,
                                               embed_dim=config.encoder_embed_dim,  
                                               requires_grad=config.learnable_positional_encodings)
        
        self.encoder_blocks = nn.ModuleList(
            [
                EncoderBlock(embed_dim=config.encoder_embed_dim,
                             num_heads=config.encoder_num_heads, 
                             mlp_ratio=config.encoder_mlp_ratio, 
                             proj_p=config.encoder_proj_p, 
                             attn_p=config.encoder_attn_p, 
                             mlp_p=config.encoder_mlp_p, 
                             fused_attention=config.fused_attention)

                for _ in range(config.encoder_depth)
            ]
        )

        self.encoder_layer_norm = nn.LayerNorm(config.encoder_embed_dim)
    

    def forward(self, x, mask_ratio=0.0):
        
        batch_size, channels, height, width = x.shape 

        ### Patch Embedding ###
        x = self.patch_embed(x)
        
        ### Add Position Embedding without CLS token ###
        x = x + self.enc_pos_embed[:, 1:, :]
        
        ### Random Masking ###
        if mask_ratio > 0.0:

            x, mask, restore_idx = random_masking(x, mask_ratio=mask_ratio)

        ### Add Position Information and Concatenate on CLS Token ###
        cls_token  = self.enc_cls_token + self.enc_pos_embed[:, :1, :]

        ### Expand to Batch Dimension on CLS Token and Concat ###
        cls_token = cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        ### Pass through Transformer Blocks ###
        for block in self.encoder_blocks:
            x = block(x)

        ### Normalize ###
        x = self.encoder_layer_norm(x)

        return x, mask, restore_idx

class VITMAEDecoder(nn.Module):
    def __init__(self, config, num_patches):
        super(VITMAEDecoder, self).__init__()

        self.config = config 
        self.num_patched = num_patches

        ### Linear Layer to Project from Encoder to Decoder Embed Dims ###
        self.encoder2decoder_embbedding_proj = nn.Linear(config.encoder_embed_dim, 
                                                         config.decoder_embed_dim)

        ### Mask Token PlaceHolder (Same as RoBERTa Implementation) ###
        self.mask_token = nn.Parameter(torch.zeros(1,1,config.decoder_embed_dim))
        
        ### CLS Token and SinCos Positional Embeddings ###
        self.dec_cls_token = nn.Parameter(torch.zeros(1,1,config.encoder_embed_dim))
        self.dec_pos_embed = sincos_embeddings(num_tokens=num_patches+1,
                                               embed_dim=config.encoder_embed_dim,  
                                               requires_grad=config.learnable_positional_encodings)

        self.decoder_blocks = nn.ModuleList(
            [
                EncoderBlock(embed_dim=config.decoder_embed_dim,
                             num_heads=config.decoder_num_heads, 
                             mlp_ratio=config.decoder_mlp_ratio, 
                             proj_p=config.decoder_proj_p, 
                             attn_p=config.decoder_attn_p, 
                             mlp_p=config.decoder_mlp_p, 
                             fused_attention=config.fused_attention)

                for _ in range(config.encoder_depth)
            ]
        )

        self.decoder_layer_norm = nn.LayerNorm(config.encoder_embed_dim)

    def forward(self, x, restore_idx):

        x = self.encoder2decoder_embbedding_proj(x)
        


if __name__ == "__main__":

    rand = torch.randn(4,3,224,224)
    mae_config = MAEConfig()
    model = VITMAEEncoder(mae_config)
    out, mask, restore_idx = model(rand, mask_ratio=0.75)
    print(out.shape)
    print(mask.shape)
    print(restore_idx.shape)