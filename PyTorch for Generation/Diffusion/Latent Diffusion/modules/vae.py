import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import ResidualBlock2D, EncoderBlock2D, DecoderBlock2D
from transformer import Attention
from config import LDMConfig

class VAEAttentionResidualBlock(nn.Module):

    """
    In the implementation of autoencoder_kl (https://github.com/huggingface/diffusers/blob/v0.32.2/src/diffusers/models/autoencoders/autoencoder_kl.py)
    
    After all the ResidualBlocks of the Encoder, and at the start of the Decoder there is a ResidualBlock+Self-Attention that they call the UNetMidBlock2D. 
    This class is exactly that, where each Block starts with 1 Residual Block, and then we toggle between Attention and Residual Blocks.

    Args:
        - in_channels: Number of input channels to our Block
        - dropout_p: What dropout probability do you want to use?
        - num_layers: How many iterations of Attention/ResidualBlocks do you want?
        - groupnorm_groups: How many groups in the GroupNormalization
        - norm_eps: eps for GroupNorm
        - attention_head_dim: Embed Dim for each head of attention
        - attention_residual_connections: Do you want a residual connection in Attention?
    
    """

    def __init__(self, 
                 in_channels, 
                 dropout_p = 0.0, 
                 num_layers = 1,
                 groupnorm_groups = 32,
                 norm_eps=1e-6,
                 attention_head_dim=1,
                 attention_residual_connection=True):
        
        super(VAEAttentionResidualBlock, self).__init__()
        
        self.resnets = nn.ModuleList()
        self.attentions = nn.ModuleList()

        ### There is Always One Residual Block ###
        self.resnets.append(
            ResidualBlock2D(
                 in_channels=in_channels, 
                 out_channels=in_channels, 
                 dropout_p=dropout_p, 
                 groupnorm_groups=groupnorm_groups,
                 time_embed_proj=False,
                 norm_eps=norm_eps
            )
        )

        ### For Every Layer, Create an Attention + Residual Block Stack ###
        for _ in range(num_layers):

            self.attentions.append(
                Attention(
                    embedding_dimension=in_channels,
                    head_dim=attention_head_dim,
                    attn_dropout=dropout_p,
                    attention_residual_connection=attention_residual_connection,
                    return_shape="2D"
                )
            )

            self.resnets.append(
                ResidualBlock2D(
                    in_channels=in_channels, 
                    out_channels=in_channels, 
                    dropout_p=dropout_p, 
                    groupnorm_groups=groupnorm_groups,
                    time_embed_proj=False, 
                    norm_eps=norm_eps
                )
            )

    def forward(self, 
                x, 
                time_embed=None):

        x = self.resnets[0](x, time_embed=time_embed)

        for attn, res in zip(self.attentions, self.resnets[1:]):
            x = attn(x)
            x = res(x, time_embed)
        
        return x


class VAEEncoder(nn.Module):

    """
    Encoder for the Variational AutoEncoder

    Args:
        - in_channels: Number of input channels in our images
        - out_channels: The latent dimension output of our encoder
        - double_z: If we are doing VAE, we need Mean/Std channels, else we just need our output
        - channels_per_block: How many starting channels in every block?
        - residual_layers_per_block: How many ResidualBlocks in every EncoderBlock
        - num_attention_layers: Number of Self-Attention layers stacked at end of encoder
        - attention_residual_connections: Do you want to use attention residual connections
        - dropout_p: What dropout probability do you want to use?
        - groupnorm_groups: How many groups in your groupnorm
        - norm_eps: Groupnorm eps
        - downsample_factor: Every block downsamples by what proportion?
        - downsample_kernel_size: What kernel size for downsampling?

    """

    def __init__(self, 
                 in_channels = 3, 
                 out_channels = 4, 
                 double_z = True,
                 channels_per_block = (128, 256, 512, 512), # Downsample Factor: 2^(len(channels_per_block) - 1)
                 residual_layers_per_block = 2,
                 num_attention_layers=1, 
                 attention_residual_connections=True,
                 dropout_p=0.0, 
                 groupnorm_groups=32,
                 norm_eps=1e-6,
                 downsample_factor=2, 
                 downsample_kernel_size=3):
        
        super(VAEEncoder, self).__init__()

        self.in_channels = in_channels
        self.latent_channels = out_channels
        self.residual_layers_per_block = residual_layers_per_block
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
                EncoderBlock2D(
                    in_channels=in_channels, 
                    out_channels=output_channels, 
                    dropout_p=dropout_p,
                    groupnorm_groups=groupnorm_groups,
                    norm_eps=norm_eps, 
                    num_residual_blocks=self.residual_layers_per_block,
                    time_embed_proj=False,
                    add_downsample=not is_final_block, 
                    downsample_factor=downsample_factor, 
                    downsample_kernel_size=downsample_kernel_size
                )
            )

        ### AttentionResidualBlock (No change in img size) ###
        self.attn_block = VAEAttentionResidualBlock(
            in_channels=self.channels_per_block[-1],
            dropout_p=dropout_p, 
            num_layers=num_attention_layers, 
            groupnorm_groups=groupnorm_groups,
            norm_eps=norm_eps,
            attention_head_dim=self.channels_per_block[-1], 
            attention_residual_connection=attention_residual_connections
        )

        ### Final Output Layers ###
        self.out_norm = nn.GroupNorm(num_channels=self.channels_per_block[-1],
                                     num_groups=groupnorm_groups, 
                                     eps=1e-6)
        
        conv_out_channels = 2 * self.latent_channels if double_z else self.latent_channels
        self.conv_out = nn.Conv2d(self.channels_per_block[-1],
                                  conv_out_channels, # We want 4 latent channels (so 4 for mean and 4 for std)
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

    """
    Decoder for the Variational AutoEncoder

    Args:
        - in_channels: Number of input channels in our images
        - out_channels: The latent dimension output of our encoder
        - channels_per_block: How many starting channels in every block? (Need to Reverse)
        - residual_layers_per_block: How many ResidualBlocks in every EncoderBlock
        - num_attention_layers: Number of Self-Attention layers stacked at end of encoder
        - attention_residual_connections: Do you want to use attention residual connections
        - dropout_p: What dropout probability do you want to use?
        - groupnorm_groups: How many groups in your groupnorm
        - norm_eps: Groupnorm eps
        - upsample_factor: Every block upsamples by what proportion?
        - upsample_kernel_size: What kernel size for upsampling?

    """

    def __init__(self, 
                 in_channels = 3, 
                 out_channels = 4, 
                 channels_per_block = (128, 256, 512, 512), # Upsample Factor: 2^(len(channels_per_block) - 1)
                 residual_layers_per_block = 2,
                 num_attention_layers=1, 
                 attention_residual_connections=True,
                 dropout_p=0.0, 
                 groupnorm_groups=32,
                 norm_eps=1e-6,
                 upsample_factor=2, 
                 upsample_kernel_size=3):
        
        super(VAEDecoder, self).__init__()

        self.latent_channels = in_channels
        self.out_channels = out_channels
        self.residual_layers_per_block = residual_layers_per_block
        self.channels_per_block = channels_per_block[::-1]

        self.conv_in = nn.Conv2d(in_channels=in_channels, 
                                 out_channels=self.channels_per_block[0],
                                 kernel_size=3, 
                                 stride=1,
                                 padding=1)
        
        ### AttentionResidualBlock (No change in img size) ###
        self.attn_block = VAEAttentionResidualBlock(
            in_channels=self.channels_per_block[0],
            dropout_p=dropout_p, 
            num_layers=num_attention_layers, 
            groupnorm_groups=groupnorm_groups,
            norm_eps=norm_eps,
            attention_head_dim=self.channels_per_block[0], 
            attention_residual_connection=attention_residual_connections
        )

        self.decoder_blocks = nn.ModuleList()

        output_channels = self.channels_per_block[0]
        for i, channels in enumerate(self.channels_per_block):

            in_channels = output_channels
            output_channels = self.channels_per_block[i]
            is_final_block = (i==len(self.channels_per_block)-1)
            
            self.decoder_blocks.append(
                DecoderBlock2D(
                    in_channels=in_channels, 
                    out_channels=output_channels, 
                    dropout_p=dropout_p,
                    groupnorm_groups=groupnorm_groups, 
                    norm_eps=norm_eps,
                    num_residual_blocks=self.residual_layers_per_block,
                    time_embed_proj=False,
                    add_upsample=not is_final_block, 
                    upsample_factor=upsample_factor,
                    upsample_kernel_size=upsample_kernel_size
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

        for block in self.decoder_blocks:
            x = block(x)

        x = self.out_norm(x)
        x = F.silu(x)
        x = self.conv_out(x)

        return x

class EncoderDecoder(nn.Module):

    """
    Putting the Encoder/Decoder together so we can pass it into another class
    to perform the VAE or VQVAE Task
    """
    
    def __init__(self, config):

        super(EncoderDecoder, self).__init__()

        self.config = config

        self.encoder = VAEEncoder(in_channels=config.in_channels, 
                                  out_channels=config.latent_channels, 
                                  double_z=not config.quantize,
                                  channels_per_block=config.vae_channels_per_block,
                                  residual_layers_per_block=config.residual_layers_per_block, 
                                  num_attention_layers=config.attention_layers,
                                  attention_residual_connections=config.attention_residual_connections,
                                  dropout_p=config.dropout,
                                  groupnorm_groups=config.groupnorm_groups,
                                  norm_eps=config.norm_eps, 
                                  downsample_factor=config.vae_up_down_factor, 
                                  downsample_kernel_size=config.vae_up_down_kernel_size)
        
        self.decoder = VAEDecoder(in_channels=config.latent_channels, 
                                  out_channels=config.out_channels, 
                                  channels_per_block=config.vae_channels_per_block,
                                  residual_layers_per_block=config.residual_layers_per_block + 1, 
                                  num_attention_layers=config.attention_layers,
                                  attention_residual_connections=config.attention_residual_connections,
                                  dropout_p=config.dropout,
                                  groupnorm_groups=config.groupnorm_groups,
                                  norm_eps=config.norm_eps, 
                                  upsample_factor=config.vae_up_down_factor, 
                                  upsample_kernel_size=config.vae_up_down_kernel_size)
        
    def forward_enc(self, x):
        return self.encoder(x)
    
    def forward_dec(self, x):
        return self.decoder(x)
    
class VAE(EncoderDecoder):
    
    def __init__(self, config):
        super(VAE, self).__init__(config=config)
    
    def kl_loss(self):
        pass
    
    def forward(self, x):
        
        encoded = self.forward_enc(x)
        print(encoded.shape)

class VQVAE(EncoderDecoder):
    pass


    

if __name__ == "__main__":

    config = LDMConfig()
    model = VAE(config)

    rand = torch.randn(4,3,256,256)
    model(rand)
