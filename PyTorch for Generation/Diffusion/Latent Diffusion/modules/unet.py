import torch
import torch.nn as nn
from layers import ResidualBlock2D, DownSampleBlock2D, UpSampleBlock2D
from transformer import TransformerBlock2D
from config import LDMConfig

class DownBlock2D(nn.Module):
    
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 attention=True,
                 num_attention_heads=1,
                 cross_attention_dim=None,
                 time_embed_dim=1280, 
                 dropout=0.0,  
                 num_layers=1, 
                 transformers_per_layer=1, 
                 transformer_dim_mult=4,
                 attention_bias=False,
                 norm_eps=1e-6, 
                 groupnorm_groups=32, 
                 add_downsample=True,
                 downsample_factor=2,
                 downsample_kernel_size=3):
        
        super(DownBlock2D, self).__init__()
        
        self.add_downsample = add_downsample

        self.resnets = nn.ModuleList([])
        self.attentions = nn.ModuleList([])

        for i in range(num_layers):
            
            in_channels = in_channels if i == 0 else out_channels

            self.resnets.append(

                ResidualBlock2D(
                    in_channels=in_channels, 
                    out_channels=out_channels, 
                    dropout_p=dropout, 
                    groupnorm_groups=groupnorm_groups,
                    time_embed_proj=True, 
                    time_embed_dim=time_embed_dim,
                    norm_eps=norm_eps
                )
            )

            if attention:
                self.attentions.append(

                    TransformerBlock2D(
                        num_attention_heads=num_attention_heads, 
                        attention_head_dim=out_channels // num_attention_heads,
                        in_channels=out_channels, 
                        cross_attn_dim=cross_attention_dim, 
                        num_layers=transformers_per_layer, 
                        dim_mult=transformer_dim_mult,
                        groupnorm_groups=groupnorm_groups, 
                        attn_bias=attention_bias,
                        norm_eps=norm_eps,
                        dropout_p=dropout
                    )
                )
            
            else:

                self.attentions.append(None)

        if self.add_downsample:
            self.downsample = DownSampleBlock2D(in_channels=out_channels, 
                                                kernel_size=downsample_kernel_size, 
                                                downsample_factor=downsample_factor)
            
    def forward(self, x, time_embed, context=None, attention_mask=None):

        skip_connection_outputs = []

        for i, (res, attn) in enumerate(zip(self.resnets, self.attentions)):

            x = res(x, time_embed)

            if attn is not None:
                x = attn(x, context, attention_mask)

            skip_connection_outputs.append(x)

        if self.add_downsample:
            x = self.downsample(x)
            skip_connection_outputs.append(x)

        return x, skip_connection_outputs
    
class MidBlock2D(nn.Module):
    
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 attention=True,
                 num_attention_heads=1,
                 cross_attention_dim=None,
                 time_embed_dim=1280, 
                 dropout=0.0,  
                 num_layers=1, 
                 transformers_per_layer=1, 
                 transformer_dim_mult=4,
                 attention_bias=False,
                 norm_eps=1e-6, 
                 groupnorm_groups=32):
        
        super(MidBlock2D, self).__init__()
        
        ### Always Starts with a ResNetBlock ###
        self.resnets = nn.ModuleList(
            [
                ResidualBlock2D(
                    in_channels=in_channels, 
                    out_channels=out_channels, 
                    dropout_p=dropout, 
                    groupnorm_groups=groupnorm_groups,
                    time_embed_proj=True, 
                    time_embed_dim=time_embed_dim,
                    norm_eps=norm_eps
                )
            ]
        )

        self.attentions = nn.ModuleList([])

        for i in range(num_layers):
            
            in_channels = in_channels if i == 0 else out_channels

            self.resnets.append(

                ResidualBlock2D(
                    in_channels=in_channels, 
                    out_channels=out_channels, 
                    dropout_p=dropout, 
                    groupnorm_groups=groupnorm_groups,
                    time_embed_proj=True, 
                    time_embed_dim=time_embed_dim,
                    norm_eps=norm_eps
                )
            )

            if attention:

                self.attentions.append(

                    TransformerBlock2D(
                        num_attention_heads=num_attention_heads, 
                        attention_head_dim=out_channels // num_attention_heads,
                        in_channels=out_channels, 
                        cross_attn_dim=cross_attention_dim, 
                        num_layers=transformers_per_layer, 
                        dim_mult=transformer_dim_mult,
                        groupnorm_groups=groupnorm_groups, 
                        attn_bias=attention_bias,
                        norm_eps=norm_eps,
                        dropout_p=dropout
                    )
                )
            
            else:

                self.attentions.append(None)
            
    def forward(self, x, time_embed, context=None, attention_mask=None):
        
        x = self.resnest[0](x, time_embed)

        for i, (res, attn) in enumerate(zip(self.resnets, self.attentions)):

            x = res(x, time_embed)

            if attn is not None:
                x = attn(x, context, attention_mask)

        return x

class UpBlock2D(nn.Module):
    
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 attention=True,
                 num_attention_heads=1,
                 cross_attention_dim=None,
                 time_embed_dim=1280, 
                 dropout=0.0,  
                 num_layers=1, 
                 transformers_per_layer=1, 
                 transformer_dim_mult=4,
                 attention_bias=False,
                 norm_eps=1e-6, 
                 groupnorm_groups=32, 
                 add_upsample=True,
                 upsample_factor=2,
                 upsample_kernel_size=3):
        
        super(DownBlock2D, self).__init__()
        
        self.add_upsample = add_upsample

        self.resnets = nn.ModuleList([])
        self.attentions = nn.ModuleList([])

        for i in range(num_layers):
            
            in_channels = in_channels if i == 0 else out_channels

            self.resnets.append(

                ResidualBlock2D(
                    in_channels=in_channels, 
                    out_channels=out_channels, 
                    dropout_p=dropout, 
                    groupnorm_groups=groupnorm_groups,
                    time_embed_proj=True, 
                    time_embed_dim=time_embed_dim,
                    norm_eps=norm_eps
                )
            )

            if attention:
                self.attentions.append(

                    TransformerBlock2D(
                        num_attention_heads=num_attention_heads, 
                        attention_head_dim=out_channels // num_attention_heads,
                        in_channels=out_channels, 
                        cross_attn_dim=cross_attention_dim, 
                        num_layers=transformers_per_layer, 
                        dim_mult=transformer_dim_mult,
                        groupnorm_groups=groupnorm_groups, 
                        attn_bias=attention_bias,
                        norm_eps=norm_eps,
                        dropout_p=dropout
                    )
                )
            
            else:

                self.attentions.append(None)

        if self.add_upsample:
            self.upsample = UpSampleBlock2D(in_channels=out_channels, 
                                            kernel_size=upsample_kernel_size, 
                                            stride=upsample_factor)
            
    def forward(self, x, time_embed, context=None, attention_mask=None):

        skip_connection_outputs = []

        for i, (res, attn) in enumerate(zip(self.resnets, self.attentions)):

            x = res(x, time_embed)

            if attn is not None:
                x = attn(x, context, attention_mask)

            skip_connection_outputs.append(x)

        if self.add_downsample:
            x = self.downsample(x)
            skip_connection_outputs.append(x)

        return x, skip_connection_outputs


class UNet2DModel(nn.Module):
    def __init__(self, config):
        super(UNet2DModel, self).__init__()

        self.config = config

        self.conv_in = nn.Conv2d(in_channels=config.latent_channels,
                                 out_channels=config.unet_channels_per_block[0],
                                 kernel_size=3, 
                                 padding="same") 

        ### Encoder ###
        self.down_blocks = nn.ModuleList()
        output_channels = config.unet_channels_per_block[0]
        for i, block_type in enumerate(config.down_block_types):
            input_channels = output_channels
            output_channels = config.unet_channels_per_block[i]
            is_final_block = (i == len(config.unet_channels_per_block)-1)

            ### Check if This Block Has Attention ###
            use_attention = True if block_type == "AttnDown" else False

            block = DownBlock2D(in_channels=input_channels, 
                                out_channels=output_channels, 
                                attention=use_attention,
                                num_attention_heads=output_channels//config.attention_head_dim,
                                cross_attention_dim=config.text_embed_dim,
                                time_embed_dim=config.time_embed_proj_dim, 
                                dropout=config.dropout,  
                                num_layers=config.unet_residual_layers_per_block, 
                                transformers_per_layer=config.transformer_blocks_per_layer, 
                                transformer_dim_mult=config.transformer_dim_mult,
                                attention_bias=config.attention_bias,
                                norm_eps=config.norm_eps, 
                                groupnorm_groups=config.groupnorm_groups, 
                                add_downsample=not is_final_block,
                                downsample_factor=config.downsample_factor,
                                downsample_kernel_size=config.downsample_kernel_size)

            self.down_blocks.append(block)

    def forward(self, x, time_embed, context=None, attention_mask=None):
                
        x = self.conv_in(x)

        for block in self.down_blocks:
            x, skip_connection = block(x, time_embed)
            
        pass






if __name__ == "__main__":

    config = LDMConfig(text_embed_dim=None)
    model = UNet2DModel(config)
    print(model)

    rand = torch.randn(4,4,32,32)
    time = torch.randn(4,1280)
    
    model(rand, time)
        