from dataclasses import dataclass
from typing import Tuple

@dataclass
class LDMConfig:

    ### VARIATIONAL AUTOENCODER CONFIG ###
    in_channels: int = 3
    out_channels: int = 3
    downsample_factor: int = 8
    latent_channels: int = 4
    norm_num_groups: int = 32
    layers_per_block: int = 2
    channels_per_block: Tuple = (128, 256, 512, 512) # num_downs = len(channels_out) - 1


