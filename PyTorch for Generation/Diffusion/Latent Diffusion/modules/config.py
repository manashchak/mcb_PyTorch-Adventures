from dataclasses import dataclass
from typing import Tuple

@dataclass
class LDMConfig:

    ######################################
    ### VARIATIONAL AUTOENCODER CONFIG ###
    ######################################
    
    ### Input/Latent/Output Channels ###
    img_size: int = 256
    in_channels: int = 3
    out_channels: int = 3
    latent_channels: int = 4

    ### Encoder Config ###
    encoder_residual_layers_per_block: int = 2
    encoder_attention_layers: int = 1
    encoder_attention_residual_connections: bool = True

    ### Decoder Config ###
    decoder_residual_layers_per_block: int = 3
    decoder_attention_layers: int = 1
    decoder_attention_residual_connections: bool = True

    ### Block Config ###
    vae_channels_per_block: Tuple = (128, 256, 512, 512)
    factor: int = 2
    kernel_size: int = 3

    ### Quantization ###
    quantize = False
    codebook_size = 16384

    ###################
    ### UNET CONFIG ###
    ###################

    ### UNET Parts Config ###
    down_block_types: Tuple = ("Down", "AttnDown", "AttnDown", "AttnDown")
    mid_block_types: str = "AttnMid"
    up_block_types: Tuple = ("Up", "AttnUp", "AttnUp", "AttnUp")
    unet_channels_per_block: Tuple = (320, 640, 1280, 1280)
    unet_residual_layers_per_block: int = 2
    downsample_factor: int = 2
    downsample_kernel_size: int = 3

    ### Time Embeddings Config ###
    time_embed_start_dim: int = 320
    time_embed_proj_dim: int = 1280

    ### Attention Config ###
    transformer_blocks_per_layer: int = 1
    transformer_dim_mult: int = 4
    attention_bias: bool = False
    attention_head_dim: int = 8

    ### Text Embeddings Config ###
    text_embed_dim: int = 768

    ######################
    ### GENERAL CONFIG ###
    ######################
    groupnorm_groups: int = 2
    norm_eps: float = 1e-6
    dropout: float = 0.0

