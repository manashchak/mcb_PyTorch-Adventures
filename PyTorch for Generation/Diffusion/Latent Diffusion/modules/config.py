from dataclasses import dataclass
from typing import Tuple

@dataclass
class LDMConfig:

    ######################################
    ### VARIATIONAL AUTOENCODER CONFIG ###
    ######################################
    
    ### Input/Latent/Output Channels ###
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
    channels_per_block: Tuple = (128, 256, 512, 512)
    factor: int = 2
    kernel_size: int = 3
    dropout: float = 0.0

    ### Quantization ###
    quantize = False
    codebook_size = 16384

    ###################
    ### UNET CONFIG ###
    ###################


    ######################
    ### GENERAL CONFIG ###
    ######################

    ### GroupNorm Config ###
    groupnorm_groups: int = 32
    norm_eps: float = 1e-6
        

