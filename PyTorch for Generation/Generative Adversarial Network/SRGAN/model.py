import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(

            nn.Conv2d(in_channels=in_channels, 
                      out_channels=in_channels, 
                      kernel_size=3, 
                      stride=1, 
                      padding=1, 
                      bias=False),

            nn.BatchNorm2d(in_channels),

            nn.PReLU(),

            nn.Conv2d(in_channels=in_channels, 
                      out_channels=in_channels, 
                      kernel_size=3, 
                      stride=1, 
                      padding=1, 
                      bias=False),

            nn.BatchNorm2d(in_channels)
        
        )

    def forward(self, x):

        x = x + self.block(x)

        return x

class PixelShuffle(nn.Module):
    """
    Rearranges pixels from the channel dimension to the spatial dimensions. 
    """
    def __init__(self, upscale_factor):
        super(PixelShuffle, self).__init__()
        self.r = upscale_factor

    def forward(self, x):
        
        batch, channels, height, width = x.shape

        if channels % (self.r * self.r) != 0:
            raise ValueError(f"Input channels ({channels}) must be divisible by square of upscale factor ({self.r * self.r})")

        ### Reshape [B x C x H x W] -> [B x C//r**2 x r x r x H x W] ###
        x = x.reshape(batch, channels//(self.r*self.r), self.r, self.r, height, width)

        ### Permute Dimensions [B x C//r**2 x r x r x H x W] -> [B x C//r**2 x H x r x W x r] ###
        x = x.permute(0,1,4,2,5,3)

        ### Flatten Height and Width: [B x C//r**2 x H x r x W x r] -> [B x C//r**2 x H*r x W*r] ###
        x = x.reshape(batch, channels//(self.r*self.r), height*self.r, width*self.r)

        return x

class UpsampleBlock(nn.Module):
    def __init__(self, channels, upsample_factor):
        super(UpsampleBlock, self).__init__()
        
        self.upsample_block = nn.Sequential(
            nn.Conv2d(in_channels=channels, 
                      out_channels=channels*upsample_factor*upsample_factor,
                      kernel_size=3,
                      stride=1, 
                      padding=1),
            PixelShuffle(upsample_factor),
            nn.PReLU()
        )

    def forward(self, x):
        return self.upsample_block(x)
    
class SRNet(nn.Module):
    def __init__(self, 
                 in_channels=3,
                 channels=64, 
                 num_residual_blocks=16, 
                 upsample=4):
        
        super(SRNet, self).__init__()

        ### Projection Conv ###
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, 
                      out_channels=channels, 
                      kernel_size=9, 
                      stride=1, 
                      padding=4),
            nn.PReLU()
        )

        ### Residual Blocks ###
        residual_blocks = nn.ModuleList(
            ResidualBlock(channels) for _ in range(num_residual_blocks)
        )
        self.residual_blocks = nn.Sequential(*residual_blocks)

        ### Post Residual Conv ###
        self.post_res_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, 
                      out_channels=channels,  
                      kernel_size=3, 
                      stride=1, 
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(channels)
        )

        ### Upsample Block ###
        num_upsamples = int(math.log2(upsample))
        upsample_blocks = nn.ModuleList(
            UpsampleBlock(channels=channels, 
                          upsample_factor=2)
            
            for _ in range(num_upsamples)
        )
        self.upsample_blocks = nn.Sequential(*upsample_blocks)

        ### Final Convolution ###
        self.final_conv = nn.Conv2d(in_channels=channels, 
                                    out_channels=in_channels, 
                                    kernel_size=9, 
                                    stride=1, 
                                    padding=4)
        

    def forward(self, x):
        
        ### First Projection ###
        x = self.in_conv(x)

        ### Store Copy for Residual ###
        residual = x

        ### Pass Through Residual Blocks ###
        x = self.residual_blocks(x)

        ### Pass Through Post Residual Block ###
        x = self.post_res_conv(x)

        ### Add Pre Residual Block Output ###
        x = x + residual

        ### Go Through Upsampling Blocks ###
        x = self.upsample_blocks(x)

        ### Final Convolution ###
        x = self.final_conv(x) 

        return F.tanh(x)

class Discriminator(nn.Module):
    def __init__(self, 
                 in_channels=3,
                 channels=[64,64,128,128,256,256,512,512]):
        super(Discriminator, self).__init__()

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, 
                      out_channels=channels[0], 
                      kernel_size=3,
                      stride=1,
                      padding=1), 
            nn.LeakyReLU(0.2)
        )

        feature_extractor = []
        for idx in range(len(channels) - 1):
            
            ### Get Channels for this block ###
            in_channels = channels[idx]
            out_channels = channels[idx + 1]

            ### Stride of 2 on even blocks ###
            stride = 2 if idx%2==0 else 1

            feature_extractor.extend(
                self._block(in_channels=in_channels, 
                            out_channels=out_channels, 
                            stride=stride)
            )
        self.feature_extractor = nn.Sequential(*feature_extractor)
        
        self.head = nn.Sequential(
            nn.Linear(512*6*6, 1024), 
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1)
        )

    def _block(self, in_channels, out_channels, stride):
        
        block = [
            nn.Conv2d(in_channels=in_channels, 
                      out_channels=out_channels, 
                      kernel_size=3, 
                      stride=stride, 
                      padding=1, 
                      bias=False),
            nn.BatchNorm2d(out_channels), 
            nn.LeakyReLU(0.2)
        ]

        return block
    
    def forward(self, x):

        x = self.in_conv(x)
        x = self.feature_extractor(x).flatten(1)
        x = self.head(x)
        
        return x
    

if __name__ == "__main__":

    model = Discriminator()
    rand = torch.randn(4,3,96,96)
    model(rand)