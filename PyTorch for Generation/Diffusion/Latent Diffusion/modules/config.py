from dataclasses import dataclass
from typing import Tuple

@dataclass
class LDMConfig:

    ### VARIATIONAL AUTOENCODER CONFIG ###
    in_channels: int = 3
    out_channels: int = 3
    downsample_factor: int = 8
    latent_channels: int = 4
    groupnorm_groups: int = 32
    encoder_layers_per_block: int = 2
    decoder_layers_per_block: int = 3
    channels_per_block: Tuple = (128, 256, 512, 512)
    dropout: float = 0.0


