import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 dropout_p = 0.0, 
                 groupnorm_groups = 32,
                 time_embed_proj = False, 
                 time_embed_dim = 128,
                 norm_eps=1e-6):
        
        super(ResidualBlock, self).__init__()


        ### Input Convolutions ###
        self.norm1 = nn.GroupNorm(num_groups=groupnorm_groups, num_channels=in_channels, eps=norm_eps, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding="same")

        ### Second Set of Convolutions ###
        self.norm2 = nn.GroupNorm(num_groups=groupnorm_groups, num_channels=out_channels, eps=norm_eps, affine=True)
        self.dropout = nn.Dropout(dropout_p)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding="same")

        ### Time Embedding Mapping ###
        self.time_expand = None
        if time_embed_proj:
            self.time_expand = nn.Linear(time_embed_dim, out_channels)
        
        ### Residual Connection Upchannels ###
        self.identity_conv = nn.Identity()
        if in_channels != out_channels:
            self.identity_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x, time_embed=None):

        residual_connection = x

        x = self.norm1(x)
        x = F.silu(x)
        x = self.conv1(x)

        if time_embed is not None:
            if self.time_expand is None:
                raise Exception("Passing in Time Embedding into ResidualBlock without time_embed_proj = True")
            else:

                ### Project Time Embedding from (B x time_embed_dim) -> (B x out_channels) ###
                time_embed = self.time_expand(time_embed)

                ### Reshape (B x out_channels) -> (B x out_channels x 1 x 1) ###
                time_embed = time_embed.reshape((*time_embed.shape, 1, 1))

                ### Add Time Information to Images (B x out_channel x h x w) + (B x out_channels x 1 x 1) ###
                x = x + time_embed
        
        x = self.norm2(x)
        x = F.silu(x)
        x = self.dropout(x)
        x = self.conv2(x)

        ### Residual Connection ###
        x = x + self.identity_conv(residual_connection)

        return x

class DownsampleBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 downsample_factor=2,
                 kernel_size=3):        
        super(DownsampleBlock, self).__init__()

        self.downsample_conv = nn.Conv2d(in_channels=in_channels, 
                                         out_channels=in_channels, 
                                         kernel_size=kernel_size, 
                                         stride=downsample_factor,
                                         padding=1)
        
    def forward(self, x):
        return self.downsample_conv(x)

class EncoderBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 dropout_p = 0.0,
                 norm_eps = 1e-6, 
                 groupnorm_groups = 32, 
                 num_residual_blocks = 2, 
                 time_embed_proj = False, 
                 time_embed_dim = 128, 
                 add_downsample = True,
                 downsample_factor = 2, 
                 downsample_kernel_size = 3):
        
        super(EncoderBlock, self).__init__()
        
        self.blocks = nn.ModuleList()

        for i in range(num_residual_blocks):       
            conv_in_channels = in_channels if i == 0 else out_channels
            self.blocks.append(
                ResidualBlock(in_channels=conv_in_channels, 
                              out_channels=out_channels,
                              groupnorm_groups=groupnorm_groups,
                              dropout_p=dropout_p, 
                              time_embed_proj=time_embed_proj, 
                              time_embed_dim=time_embed_dim,
                              norm_eps=norm_eps
                        )
                )
            
        self.downsample = nn.Identity()
        if add_downsample:
            self.downsample = DownsampleBlock(in_channels=out_channels, 
                                              downsample_factor=downsample_factor, 
                                              kernel_size=downsample_kernel_size)

    def forward(self, x, time_embed=None):

        for block in self.blocks:
            x = block(x, time_embed)
        
        print(x.shape)
        x = self.downsample(x)

        return x
    
if __name__ == "__main__":

    model = EncoderBlock(64,64)
    rand = torch.randn(4,64,256,256)
    print(model(rand).shape)

