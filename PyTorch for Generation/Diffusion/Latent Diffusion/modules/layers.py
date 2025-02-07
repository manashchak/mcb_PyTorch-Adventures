import torch
import torch.nn as nn
import torch.nn.functional as F

    
class UpSampleBlock2D(nn.Module):

    """
    Upsampling Block that takes 

    (B x C x H x W) -> (B x C x H*2 x W*2)

    Args:
        - in_channels: Input channels of images (no change in channels)
        - kernel_size: Kernel size in learnable convolution
        - upsample_factor: By what factor do you want to upsample image by?
    """

    def __init__(self, 
                 in_channels, 
                 kernel_size=3, 
                 upsample_factor=2):
        
        super(UpSampleBlock2D, self).__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.factor = upsample_factor

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=upsample_factor),
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding='same')
        )

    def forward(self, x):
        
        batch, channels, height, width = x.shape

        upsampled = self.upsample(x)

        assert upsampled.shape[2:] == (height*self.factor, width*self.factor)

        return upsampled
    
class DownSampleBlock2D(nn.Module):

    """
    Downsampling Block that takes 

    (B x C x H x W) -> (B x C x H/2 x W/2)

    Args:
        - in_channels: Input channels of images (no change in channels)
        - kernel_size: Kernel size in learnable convolution
        - kernel_size: What stride do you want to use to downsample?

    """

    def __init__(self, 
                 in_channels, 
                 kernel_size=3,
                 downsample_factor=2):    
           
        super(DownSampleBlock2D, self).__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.factor = downsample_factor

        self.downsample_conv = nn.Conv2d(in_channels=in_channels, 
                                         out_channels=in_channels, 
                                         kernel_size=kernel_size, 
                                         stride=downsample_factor,
                                         padding=1)
        
    def forward(self, x):
        
        batch, channels, height, width = x.shape

        downsampled = self.downsample_conv(x)

        assert downsampled.shape[2:] == (height/self.factor, width/self.factor)
        
        return downsampled
    
class ResidualBlock2D(nn.Module):

    """
    Core Residual Block of a Stack of Convolutions and a Residual Connection

    Args:
        - in_channels: Input channels to the block
        - out_channels: Output channels from the block
        - dropout_p: Dropout probability you want to use
        - groupnorm_groups: How many groups do you want in Groupnorm
        - time_embed_proj: Do you want to inject time embeddings?
        - time_embed_dim: Time embedding dimension
        - norm_eps: Groupnorm eps
    """

    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 dropout_p = 0.0, 
                 groupnorm_groups = 32,
                 time_embed_proj = False, 
                 time_embed_dim = 1280,
                 norm_eps=1e-6):
        
        super(ResidualBlock2D, self).__init__()


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

class EncoderBlock2D(nn.Module):

    """
    The Encoder block is a stack of Residual Blocks with an optional
    downsampling layer to reduce the image size

    Args:
        - in_channels: The number of input channels to the Encoder
        - out_channels: Number of output channels of the Encoder
        - dropout_p: The dropout probability in the Residual Blocks
        - norm_eps: Groupnorm eps
        - num_residual_blocks: Number of Residual Blocks in the Encoder
        - time_embed_proj: Do you want to enable time embeddings?
        - time_embed_dim: Time embedding dimension
        - add_downsample: Do you want to downsample the image?
        - downsample_factor: By what factor do you want to downsample
        - downsample_kernel_size: Kernel size for downsampling convolution
    """

    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 dropout_p = 0.0,
                 norm_eps = 1e-6, 
                 groupnorm_groups = 32, 
                 num_residual_blocks = 2, 
                 time_embed_proj = False, 
                 time_embed_dim = 1280, 
                 add_downsample = True,
                 downsample_factor = 2, 
                 downsample_kernel_size = 3):
        
        super(EncoderBlock2D, self).__init__()
        
        self.blocks = nn.ModuleList()

        for i in range(num_residual_blocks):       
            conv_in_channels = in_channels if i == 0 else out_channels
            self.blocks.append(
                ResidualBlock2D(in_channels=conv_in_channels, 
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
            self.downsample = DownSampleBlock2D(in_channels=out_channels, 
                                                downsample_factor=downsample_factor, 
                                                kernel_size=downsample_kernel_size)

    def forward(self, x, time_embed=None):

        for block in self.blocks:
            x = block(x, time_embed)
        
        x = self.downsample(x)

        return x


class DecoderBlock2D(nn.Module):

    """
    The Decoder block is a stack of Residual Blocks with an optional
    upsampling layer to reduce the image size

    Args:
        - in_channels: The number of input channels to the Encoder
        - out_channels: Number of output channels of the Encoder
        - dropout_p: The dropout probability in the Residual Blocks
        - norm_eps: Groupnorm eps
        - num_residual_blocks: Number of Residual Blocks in the Encoder
        - time_embed_proj: Do you want to enable time embeddings?
        - time_embed_dim: Time embedding dimension
        - add_upsample: Do you want to upsample the image?
        - upsample_factor: By what factor do you want to upsample
        - upsample_kernel_size: Kernel size for upsampling convolution
    """
     

    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 dropout_p = 0.0,
                 norm_eps = 1e-6, 
                 groupnorm_groups = 32, 
                 num_residual_blocks = 2, 
                 time_embed_proj = False, 
                 time_embed_dim = 128, 
                 add_upsample = True,
                 upsample_factor = 2, 
                 upsample_kernel_size = 3):
        
        super(DecoderBlock2D, self).__init__()
        
        self.blocks = nn.ModuleList()

        for i in range(num_residual_blocks):       
            conv_in_channels = in_channels if i == 0 else out_channels
            self.blocks.append(
                ResidualBlock2D(in_channels=conv_in_channels, 
                                out_channels=out_channels,
                                groupnorm_groups=groupnorm_groups,
                                dropout_p=dropout_p, 
                                time_embed_proj=time_embed_proj, 
                                time_embed_dim=time_embed_dim,
                                norm_eps=norm_eps
                        )
                )
            
        self.upsample = nn.Identity()
        if add_upsample:
            self.upsample = UpSampleBlock2D(in_channels=out_channels, 
                                            upsample_factor=upsample_factor, 
                                            kernel_size=upsample_kernel_size)

    def forward(self, x, time_embed=None):

        for block in self.blocks:
            x = block(x, time_embed)
        
        x = self.upsample(x)

        return x

class Attention(nn.Module):

    """

    Implementation of Self and Cross Attention in One Module

    By default, self-attention will be computed on src (our images). If tgt is provided, then we are doing cross
    attention. In cross attention, an attention_mask can be used (padding mask for our embedded text), and 
    src is our text and tgt is the images.

    Self-Attention:
        - Compute Self Attention on the src Tensor
            - One new step to include though is reshaping our src (image)
                from (B x C x H x W) -> (B x H*W x C) before doing attention
    
    Cross Attention
        - src: Our text Context (B x L x E)
        - tgt: What we want to weight against our src and output
            - One new step to include though is reshaping our tgt (image)
                from (B x C x H x W) -> (B x H*W x C) before doing attention
        - attention_mask: Padding mask for the text embeddings

    Args:
        - embedding_dimension: Number of channels in Image (the channels in BCHW act as our embedding)
        - head_dim: What embedding dimension do you want to use per head?
        - attn_dropout: What dropout probability do you want to use in attention
        - groupnorm_groups: Number of groups in Groupnormalization
        - attention_residual_connections: Do you want to add the input of attention to the output?

    """

    def __init__(self, 
                 embedding_dimension=768, 
                 head_dim=1, 
                 attn_dropout=0.0,
                 groupnorm_groups=32,
                 attention_residual_connection=True):
        
        super(Attention, self).__init__()

        self.embedding_dimension = embedding_dimension
        self.attn_dropout = attn_dropout
        self.attn_residual = attention_residual_connection
        
        ### Attention Head Dim ###
        self.head_dim = head_dim
        assert embedding_dimension % head_dim == 0
        self.num_heads = embedding_dimension // head_dim

        ### GroupNorm ###
        self.groupnorm = nn.GroupNorm(num_channels=embedding_dimension, num_groups=groupnorm_groups, eps=1e-6)

        ### Attention Projections ###
        self.q_proj = nn.Linear(embedding_dimension, embedding_dimension)
        self.k_proj = nn.Linear(embedding_dimension, embedding_dimension)
        self.v_proj = nn.Linear(embedding_dimension, embedding_dimension)

        ### Post Attention Projection ###
        self.out_proj = nn.Linear(embedding_dimension, embedding_dimension)

    def _img2seq(self, x):
        """
        (B x C x H x W) -> (B x H*W x C)
        """
        batch, channels, height, width = x.shape

        x = x.reshape(batch, channels, height*width).transpose(-1,-2)

        seq_len = height * width

        return x, seq_len
    
    def _seq2img(self, x, img_dim=None):
        """
        (B x H*W x C) -> (B x C x H x W)
        """
        batch, seq_len, channels = x.shape

        ### Assume Square Image if no img_dim is provided ###
        if img_dim is None:
            h = w = int(seq_len**0.5)

        else:
            h, w = img_dim

        x = x.transpose(-1,-2).reshape(batch, channels, h, w)

        return x

    def forward_self_attn(self, images):
        
        ### Reshape from Img Dim to Seq Dim ###
        images, num_patches = self._img2seq(images)

        ### QKV Projection ###
        q = self.q_proj(images).reshape(-1, num_patches, self.num_heads, self.head_dim).transpose(1,2).contiguous()
        k = self.k_proj(images).reshape(-1, num_patches, self.num_heads, self.head_dim).transpose(1,2).contiguous()
        v = self.v_proj(images).reshape(-1, num_patches, self.num_heads, self.head_dim).transpose(1,2).contiguous()

        attention_out = F.scaled_dot_product_attention(q,k,v, 
                                                       dropout_p=self.attn_dropout if self.training else 0.0)
        
        ### Reshape back to (B, num_patches, head_dim) and Project with Linear ###
        attention_out = attention_out.transpose(1,2).flatten(2)
        attention_out = self.out_proj(attention_out)
        
        return attention_out


    def forward_cross_attn(self,
                           images, 
                           context,
                           attention_mask=None):
        
        ### Reshape from Img Dim to Seq Dim ###
        images, num_patches = self._img2seq(images)

        ### Query Projection on Images ###
        q = self.q_proj(images).reshape(-1, num_patches, self.num_heads, self.head_dim).transpose(1,2).contiguous()

        ### Key/Value Projections on Text ###
        batch, context_len, embed_dim = context.shape
        k = self.k_proj(context).reshape(batch, context_len, self.num_heads, self.head_dim).transpose(1,2).contiguous()
        v = self.v_proj(context).reshape(batch, context_len, self.num_heads, self.head_dim).transpose(1,2).contiguous()

        ### This is our text attention mask ###
        if attention_mask is not None:
            attention_mask = attention_mask.bool()
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1).repeat(1,1,num_patches,1)

        attention_out = F.scaled_dot_product_attention(q,k,v,
                                                       attn_mask=attention_mask, 
                                                       dropout_p=self.attn_dropout if self.training else 0.0)
        
        ### Reshape back to (B, num_patches, head_dim) and Project with Linear ###
        attention_out = attention_out.transpose(1,2).flatten(2)
        attention_out = self.out_proj(attention_out)

        return attention_out

    def forward(self, 
                x, 
                context=None, 
                attention_mask=None):

        residual = x 

        if context is None:
            attention_out = self.forward_self_attn(x)
        else:
            attention_out = self.forward_cross_attn(x, context, attention_mask)
        
        ### Return Back to Image Dimensions (B x H*W x C) -> (B x C x H x W) ###
        output = self._seq2img(attention_out)

        if self.attn_residual:
            output = output + residual

        return output










