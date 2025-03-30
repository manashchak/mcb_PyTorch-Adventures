import torch
import torch.nn as nn
from dataclasses import dataclass

class ResidualBlock(nn.Module):
    """
    Standard Residual Block as defined in ResnetBlock in CycleGAN, 
    ResidualBlocks dont change output channels or resolution

    Args:
        in_channels: Number of input channels (and output channels) for this block
        padding_type: What type of padding strategy do you want to use?
    """
    def __init__(self, 
                 in_channels,
                 padding_type="reflect"):
        
        super(ResidualBlock, self).__init__()

        self.residual_block = nn.Sequential(

            ### First Block ###
            nn.Conv2d(
                in_channels=in_channels, 
                out_channels=in_channels, 
                kernel_size=3, 
                padding="same",
                padding_mode=padding_type
            ),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),

            ### Second Block ###
            nn.Conv2d(
                in_channels=in_channels, 
                out_channels=in_channels, 
                kernel_size=3,
                padding="same",
                padding_mode="reflect"
            ), 
            nn.InstanceNorm2d(in_channels)

        )

    def forward(self, x):
        return x + self.residual_block(x)

class ResNetGenerator(nn.Module):


    """
    CycleGAN Resnet Generator as described in implementation:
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py#L316

    Args:
        in_channels: Number of input channels in image
        base_channels: First projection layers number of channels
        num_residual_blocks: How many blocks do you want between encoder/decoder?
        num_downsample_blocks: How many downsample (and upsample) blocks do you want?
        padding_type: What padding strategy do you want to use?
    """

    def __init__(self, 
                 in_channels=3, 
                 base_channels=64, 
                 num_residual_blocks=9, 
                 num_downsample_blocks=2,
                 padding_type="reflect"):
        
        super(ResNetGenerator, self).__init__()

        model = []

        ### Pre-Downsampling Layers (Dont change resolution, only channels) ###
        model += [
            nn.Conv2d(
                in_channels=in_channels, 
                out_channels=base_channels, 
                kernel_size=7, 
                stride=1, 
                padding="same", # used 3 here in the implementation, same effect 
                padding_mode="reflect"
            ),
            nn.InstanceNorm2d(base_channels), 
            nn.ReLU(inplace=True)
        ]

        ### Downsampling Blocks ###
        ### Each block doubles the number of channels and downs by factor of 2 ###
        for i in range(num_downsample_blocks):
            downblock_in_chans = int(base_channels * 2 ** i)
            downblock_out_chans = int(base_channels * 2 ** (i + 1))

            model += [
                nn.Conv2d(
                    in_channels=downblock_in_chans, 
                    out_channels=downblock_out_chans, 
                    kernel_size=3, 
                    stride=2, 
                    padding=1
                ), 
                nn.InstanceNorm2d(downblock_out_chans),
                nn.ReLU(inplace=True)
            ]

        ### Stack of Residual Blocks ###
        model += [
            ResidualBlock(in_channels=downblock_out_chans, 
                          padding_type=padding_type)

            for _ in range(num_residual_blocks)
        ]

        ### Upsample Back to Original Image Shape (Reverse our downsample block) ###
        for i in range(num_downsample_blocks):
            upblock_in_chans = int(downblock_out_chans * 2 ** -i)
            upblock_out_chans = int(downblock_out_chans * 2 ** -(i+1))

            model += [
                nn.ConvTranspose2d(
                    in_channels=upblock_in_chans, 
                    out_channels=upblock_out_chans, 
                    kernel_size=3, 
                    stride=2, 
                    padding=1, 
                    output_padding=1
                ), 
                nn.InstanceNorm2d(upblock_out_chans),
                nn.ReLU(inplace=True)
            ]

        ### Final Convolution to Map to Correct Number of Channels ###
        model.append(
            nn.Conv2d(
                in_channels=base_channels, 
                out_channels=in_channels, 
                kernel_size=7, 
                padding="same", 
                padding_mode="reflect"
            )
        )

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
    
class PatchGANDiscriminator(nn.Module):

    """
    PatchGAN discriminator as defined in Image to Image Translation w/ Conditional Adversarial Networks
    https://arxiv.org/pdf/1611.07004 
    """

    def __init__(self, 
                 input_channels=3, 
                 base_channels=64, 
                 depth=3,
                 kernel_size=4, 
                 padding=1,
                 leaky_relu_slope=0.2):
        
        super(PatchGANDiscriminator, self).__init__()

        current_filters = base_channels
        layers = nn.ModuleList([])

        ### Projection from input_channels to base_channels ###
        layers.append(nn.Conv2d(input_channels, current_filters, kernel_size=kernel_size, stride=2, padding=padding))
        layers.append(nn.LeakyReLU(leaky_relu_slope))

        ### Loop For All the Next Layes ###
        for i in range(depth):

            ### Apply a stride of 2 on all convoutions except the last ###
            stride = 2 if i != depth-1 else 1 
            out_channels = current_filters * 2
            layers.append(nn.Conv2d(current_filters, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2))
            
            ### Update Current Filters ###
            current_filters = out_channels

        # Output will have a single channel
        layers.append(nn.Conv2d(current_filters, 1, kernel_size=kernel_size, stride=1, padding=padding))
        
        self.model = nn.Sequential(*layers)

    def forward(self, input):
        return self.model(input)

@dataclass
class ModelConfig:
    in_channels: int = 3
    generator_base_channels: int = 64
    generator_num_residual_blocks: int = 9
    generator_num_downsample_blocks: int = 2
    generator_padding_type: str = "reflect"
    discriminator_base_channels: int = 64
    discriminator_depth: int = 3
    discriminator_kernel_size: int = 4
    discriminator_padding: int = 1
    discriminator_leaky_relu_slope: float = 0.2

class CycleGAN(nn.Module):

    """
    Wrapper putting everything together for CycleGAN so we can do a single
    model = accelerator.prepare(model) in training script

    We create: 
        generator_AB -> Translate from Domain A to Domain B
        generator_BA -> Translate from Domain B to Domain A
        discriminator_A -> Identifies if image is in domain A
        discriminator_B -> Identifies if image is in domain B 
    """
    
    def __init__(self, config):

        self.config = config

        self.generator_AB = ResNetGenerator(
            in_channels=config.in_channels, 
            base_channels=config.generator_base_channels, 
            num_residual_blocks=config.generator_num_residual_blocks,
            num_downsample_blocks=config.generator_num_downsample_blocks, 
            padding_type=config.generator_padding_type
        )

        self.generator_BA = ResNetGenerator(
            in_channels=config.in_channels, 
            base_channels=config.generator_base_channels, 
            num_residual_blocks=config.generator_num_residual_blocks,
            num_downsample_blocks=config.generator_num_downsample_blocks, 
            padding_type=config.generator_padding_type
        )

        self.discriminator_A = PatchGANDiscriminator(
            input_channels=config.input_channels, 
            base_channels=config.discriminator_base_channels, 
            depth=config.discriminator_depth, 
            kernel_size=config.discriminator_kernel_size, 
            padding=config.discriminator_padding, 
            leaky_relu_slope=config.discriminator_leaky_relu_slope
        )

        self.discriminator_B = PatchGANDiscriminator(
            input_channels=config.input_channels, 
            base_channels=config.discriminator_base_channels, 
            depth=config.discriminator_depth, 
            kernel_size=config.discriminator_kernel_size, 
            padding=config.discriminator_padding, 
            leaky_relu_slope=config.discriminator_leaky_relu_slope
        )


if __name__ == "__main__":

    rand = torch.rand(4,3,128,128)
    model = ResNetGenerator()
    out = model(rand)
    print(out.shape)

    # print(model)