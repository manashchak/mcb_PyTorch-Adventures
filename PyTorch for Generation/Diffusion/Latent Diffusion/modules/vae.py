import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import EncoderBlock, AttentionResidualBlock, DecoderBlock
from config import LDMConfig

class VAEEncoder(nn.Module):
    def __init__(self, 
                 in_channels = 3, 
                 out_channels = 4, 
                 channels_per_block = (128, 256, 512, 512),
                 layers_per_block = 2,
                 dropout_p=0.0, 
                 groupnorm_groups=32):
        
        super(VAEEncoder, self).__init__()

        self.in_channels = in_channels
        self.latent_channels = out_channels
        self.layers_per_block = layers_per_block
        self.channels_per_block = channels_per_block

        self.conv_in = nn.Conv2d(in_channels=in_channels, 
                                 out_channels=self.channels_per_block[0],
                                 kernel_size=3, 
                                 stride=1,
                                 padding=1)
        
        self.encoder_blocks = nn.ModuleList()

        output_channels = self.channels_per_block[0]
        for i, channels in enumerate(self.channels_per_block):

            in_channels = output_channels
            output_channels = self.channels_per_block[i]
            is_final_block = (i==len(self.channels_per_block)-1)
            
            self.encoder_blocks.append(
                EncoderBlock(
                    in_channels=in_channels, 
                    out_channels=output_channels, 
                    dropout_p=dropout_p,
                    groupnorm_groups=groupnorm_groups, 
                    num_residual_blocks=self.layers_per_block,
                    time_embed_proj=False,
                    add_downsample=not is_final_block, 
                )
            )

        ### AttentionResidualBlock (No change in img size) ###
        self.attn_block = AttentionResidualBlock(
            in_channels=self.channels_per_block[-1],
            dropout_p=dropout_p, 
            num_layers=1, 
            groupnorm_groups=groupnorm_groups,
            time_embed_proj=False, 
            attention_head_dim=self.channels_per_block[-1], 
            attention_residual_connection=True
        )

        ### Final Output Layers ###
        self.out_norm = nn.GroupNorm(num_channels=self.channels_per_block[-1],
                                     num_groups=groupnorm_groups, 
                                     eps=1e-6)
        
        self.conv_out = nn.Conv2d(self.channels_per_block[-1],
                                  self.latent_channels*2, # We want 4 latent channels (so 4 for mean and 4 for std)
                                  kernel_size=3,
                                  padding="same")
    
    def forward(self, x):

        x = self.conv_in(x)

        for block in self.encoder_blocks:
            x = block(x)

        x = self.attn_block(x)

        x = self.out_norm(x)
        x = F.silu(x)
        x = self.conv_out(x)

        return x
    
class VAEDecoder(nn.Module):
    def __init__(self, 
                 in_channels = 4, 
                 out_channels = 3, 
                 channels_per_block = (128, 256, 512, 512),
                 layers_per_block = 2,
                 dropout_p=0.0, 
                 groupnorm_groups=32):
        
        super(VAEDecoder, self).__init__()

        self.latent_channels = in_channels
        self.out_channels = out_channels
        self.layers_per_block = layers_per_block
        self.channels_per_block = channels_per_block[::-1]

        self.conv_in = nn.Conv2d(in_channels=in_channels, 
                                 out_channels=self.channels_per_block[0],
                                 kernel_size=3, 
                                 stride=1,
                                 padding=1)
        
        ### AttentionResidualBlock (No change in img size) ###
        self.attn_block = AttentionResidualBlock(
            in_channels=self.channels_per_block[0],
            dropout_p=dropout_p, 
            num_layers=1, 
            groupnorm_groups=groupnorm_groups,
            time_embed_proj=False, 
            attention_head_dim=self.channels_per_block[0], 
            attention_residual_connection=True
        )

        self.encoder_blocks = nn.ModuleList()

        output_channels = self.channels_per_block[0]
        for i, channels in enumerate(self.channels_per_block):

            in_channels = output_channels
            output_channels = self.channels_per_block[i]
            is_final_block = (i==len(self.channels_per_block)-1)
            
            self.encoder_blocks.append(
                DecoderBlock(
                    in_channels=in_channels, 
                    out_channels=output_channels, 
                    dropout_p=dropout_p,
                    groupnorm_groups=groupnorm_groups, 
                    num_residual_blocks=self.layers_per_block,
                    time_embed_proj=False,
                    add_upsample=not is_final_block, 
                )
            )

        ### Final Output Layers ###
        self.out_norm = nn.GroupNorm(num_channels=self.channels_per_block[-1],
                                     num_groups=groupnorm_groups, 
                                     eps=1e-6)
        
        self.conv_out = nn.Conv2d(self.channels_per_block[-1],
                                  self.out_channels,
                                  kernel_size=3,
                                  padding="same")
    
    def forward(self, x):

        x = self.conv_in(x)

        x = self.attn_block(x)

        for block in self.encoder_blocks:
            x = block(x)

        x = self.out_norm(x)
        x = F.silu(x)
        x = self.conv_out(x)

        return x

class EncoderDecoder(nn.Module):
    def __init__(self, config):

        super(EncoderDecoder, self).__init__()

        self.config = config

        self.encoder = VAEEncoder(in_channels=config.in_channels, 
                                  out_channels=config.latent_channels, 
                                  channels_per_block=config.channels_per_block,
                                  layers_per_block=config.layers_per_block, 
                                  dropout_p=config.dropout,
                                  groupnorm_groups=config.groupnorm_groups)
        
        self.decoder = VAEDecoder(in_channels=config.latent_channels, 
                                  out_channels=config.out_channels, 
                                  channels_per_block=config.channels_per_block,
                                  layers_per_block=config.layers_per_block, 
                                  dropout_p=config.dropout,
                                  groupnorm_groups=config.groupnorm_groups)

    def forward_enc(self, x):
        return self.encoder(x)
    
    def forward_dec(self, x):
        return self.decoder(x)
    
class VAE(EncoderDecoder):
    
    def __init__(self, config):
        super(VAE, self).__init__()

    def kl_loss(self):
        pass
    
class VQVAE(EncoderDecoder):
    pass


    

if __name__ == "__main__":

    config = LDMConfig()
    model = VAE(config)

    print(model)
