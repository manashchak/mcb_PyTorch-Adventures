import torch
import torch.nn as nn
import torch.nn.functional as F

def img2seq(x):
    """
    (B x C x H x W) -> (B x H*W x C)
    """
    batch, channels, height, width = x.shape

    x = x.reshape(batch, channels, height*width).transpose(-1,-2)

    seq_len = height * width

    return x, seq_len

def seq2img(x, img_dim=None):
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
        - cross_attn_dim: The embedding dimension of the text context (None for self-attention)
        - head_dim: What embedding dimension do you want to use per head?
        - attn_dropout: What dropout probability do you want to use in attention
        - groupnorm_groups: Number of groups in GroupNormalization (None if we dont need it)
        - attention_residual_connections: Do you want to add the input of attention to the output?

    """

    def __init__(self, 
                 embedding_dimension=768, 
                 cross_attn_dim=None,
                 head_dim=1, 
                 attn_dropout=0.0,
                 groupnorm_groups=None,
                 attention_residual_connection=True,
                 bias=True,
                 return_shape="1D"):
        
        super(Attention, self).__init__()

        self.embedding_dimension = embedding_dimension
        self.cross_attn_dim = cross_attn_dim
        self.attn_dropout = attn_dropout
        self.attn_residual = attention_residual_connection
        self.groupnorm_groups = groupnorm_groups

        if return_shape not in ["1D", "2D"]:
            raise Exception("Attention can output '1D' or '2D'")
        self.return_shape = return_shape

        ### Attention Head Dim ###
        self.head_dim = head_dim
        assert embedding_dimension % head_dim == 0
        self.num_heads = embedding_dimension // head_dim

        ### GroupNorm ###
        if self.groupnorm_groups is not None:
            self.groupnorm = nn.GroupNorm(num_channels=embedding_dimension, 
                                        num_groups=groupnorm_groups, eps=1e-6)

        ### Attention Projections ###
        kv_input_dim = embedding_dimension if cross_attn_dim is None else cross_attn_dim
        self.q_proj = nn.Linear(embedding_dimension, embedding_dimension, bias=bias)
        self.k_proj = nn.Linear(kv_input_dim, embedding_dimension, bias=bias)
        self.v_proj = nn.Linear(kv_input_dim, embedding_dimension, bias=bias)

        ### Post Attention Projection ###
        self.out_proj = nn.Linear(embedding_dimension, embedding_dimension)

    def _check_for_reshape(self, images):

        ### Reshape from Img Dim to Seq Dim if in shape (B,C,H,W) ###
        if len(images.shape) == 4:
            images, num_patches = img2seq(images)
        else:
            num_patches = images.shape[1]

        return images, num_patches

    def forward_self_attn(self, images):
        
        batch_size = images.shape[0]

        images, num_patches = self._check_for_reshape(images)

        ### QKV Projection ###
        q = self.q_proj(images).reshape(batch_size, num_patches, self.num_heads, self.head_dim).transpose(1,2).contiguous()
        k = self.k_proj(images).reshape(batch_size, num_patches, self.num_heads, self.head_dim).transpose(1,2).contiguous()
        v = self.v_proj(images).reshape(batch_size, num_patches, self.num_heads, self.head_dim).transpose(1,2).contiguous()

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

        images, num_patches = self._check_for_reshape(images)

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

        ### x can be 1D or 2D ###
        residual = x 

        if self.groupnorm_groups is not None:
            x = self.groupnorm(x)
        
        if context is None:
            attention_out = self.forward_self_attn(x)
        else:
            attention_out = self.forward_cross_attn(x, context, attention_mask)
        
        ### attention_out is always 1d ###
        if self.attn_residual:
            
            ### If residual shape doesnt match attention_out
            ### then the residuals must be (B,C,H,W), so we can
            ### reshape based on the output shape we want
            if len(attention_out.shape) != len(residual.shape):
                
                ### If we want a 1D output, flatten the residual before adding
                if self.return_shape == "1D":
                    residual, _ = img2seq(residual)
                
                ### If we want a 2D output, reshape the attention_out before adding
                elif self.return_shape == "2D":
                    attention_out = seq2img(attention_out)
                    
            attention_out = attention_out + residual
        
        else:

            if self.return_shape == "2D":
                attention_out = seq2img(attention_out)
                
        return attention_out


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
            nn.Upsample(scale_factor=upsample_factor, mode="nearest"),
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
                 class_embed_proj=False,
                 class_embed_dim=512,
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

        ### Class Embedding Mapping ###
        self.class_expand = None
        
        if class_embed_proj:
            self.class_expand = nn.Linear(class_embed_dim, out_channels)
        
        ### Residual Connection Upchannels ###
        self.identity_conv = nn.Identity()
        if in_channels != out_channels:
            self.identity_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, 
                x, 
                time_embed=None,
                class_conditioning=None):

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

        if class_conditioning is not None:

            if self.class_expand is None:
                raise Exception("Passing in Class Conditioning to ResidualBlock without class_embed_proj = True")
            else:
                
                ### Project Class Embedding from (B x class_embed_dim) -> (B x out_channels) ###
                class_conditioning = self.class_expand(class_conditioning)

                ### Reshape (B x out_channels) -> (B x out_channels x 1 x 1) ###
                class_conditioning = class_conditioning.reshape((*class_conditioning.shape, 1, 1))

                ### Add Time Information to Images (B x out_channel x h x w) + (B x out_channels x 1 x 1) ###
                x = x + class_conditioning

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
                                time_embed_proj=False, 
                                class_embed_proj=False,
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
                                time_embed_proj=False, 
                                class_embed_proj=False,
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
        - norm_eps: eppassead_dim: Embed Dim for each head of attention
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
                    groupnorm_groups=groupnorm_groups,
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
        
        encoder_out_channels = 2 * config.latent_channels if not config.quantize else config.latent_channels
        self.post_encoder_conv = nn.Conv2d(encoder_out_channels, encoder_out_channels, kernel_size=1, stride=1) \
                                    if config.post_encoder_latent_proj else nn.Identity()
        self.pre_decoder_conv = nn.Conv2d(config.latent_channels, config.latent_channels, kernel_size=1, stride=1) \
                                    if config.pre_decoder_latent_proj else nn.Identity()
        
    def forward_enc(self, x):
        x = self.encoder(x)
        x = self.post_encoder_conv(x)
        return x
    
    def forward_dec(self, x):
        x = self.pre_decoder_conv(x)
        x = self.decoder(x)
        return x

class VQVAE(EncoderDecoder):

    """
    Vector-Quantized Variational AutoEncoder as described in 
    Neural Discrete Representation Learning
    https://arxiv.org/abs/1711.00937
    """

    def __init__(self, config):
        super(VQVAE, self).__init__(config=config)

        self.config = config
        
        ### Ensure Quantization On in VQVAE ###
        self.config.quantize = True
        
        ### Projections To/From VQ Embed Dim ###
        self.conv_quantize_proj = nn.Conv2d(config.latent_channels, 
                                            config.vq_embed_dim, 
                                            kernel_size=1,
                                            stride=1)
        
        self.conv_latent_proj = nn.Conv2d(config.vq_embed_dim,
                                          config.latent_channels, 
                                          kernel_size=1,
                                          stride=1)

        ### Quantization Codebook ###
        self.embedding = nn.Embedding(config.codebook_size, config.vq_embed_dim)
        self.embedding.weight.data.uniform_(-1.0 / config.codebook_size, 1.0 / config.codebook_size)

    def quantize(self, z, compute_loss=False, compute_perplexity=False):

        ### Reshape to (B*H*W x E) ###
        z = z.permute(0,2,3,1)
        z_flattened = z.reshape(-1, self.config.vq_embed_dim)
        
        ### Compute Distance Between Each Embedding and Codevectors ###
        pairwise_dist = torch.cdist(z_flattened, self.embedding.weight)
        
        ### For each of our input vectors find the index of the closest codevector ###
        closest = torch.argmin(pairwise_dist, dim=-1)
   
        ### Index our Embedding Matrix to grab cooresponding codevectors ###
        quantized = self.embedding(closest).reshape(*z.shape)
        
        ### Compute CodeBook and Commitment Loss ###
        if compute_loss:
            codebook_loss = torch.mean((quantized - z.detach())**2)
            commitment_loss = torch.mean((quantized.detach() - z)**2)
            loss = codebook_loss + self.config.commitment_beta * commitment_loss

        ### Compute Codebook Perplexity ###
        if compute_perplexity:

            ### One Hot Encode Index ###
            one_hot_closest = torch.zeros(closest.shape[0], self.config.codebook_size, device=z.device)
            one_hot_closest[list(range(closest.shape[0])), closest] = 1
            util_proportion = torch.mean(one_hot_closest, dim=0)

            ### Compute Perplexity ###
            perplexity = torch.exp(-torch.sum(util_proportion * torch.log(util_proportion + 1e-8)))

        ### Copy Gradients (Straight Through Estimator) ###
        quantized = z + (quantized - z).detach()

        ### Permute Back to Original Image Shape (B,C,H,W) ###
        quantized = quantized.permute(0,3,2,1)

        output = {"quantized": quantized}
        if compute_loss:
            output["codebook_loss"] = codebook_loss
            output["commitment_loss"] = commitment_loss
            output["quantization_loss"] = loss

        if compute_perplexity:
            output["perplexity"] = perplexity

        return output

    def encode(self, x):
        
        x = self.forward_enc(x)

        z = self.conv_quantize_proj(x) 

        return self.quantize(z)

    def decode(self, z):
        
        x = self.conv_latent_proj(z)

        x = self.forward_dec(x)

        return x

    def forward(self, x):
        
        ### Encode ###
        x = self.forward_enc(x)

        ### Project to Quantized Dimension ###
        z = self.conv_quantize_proj(x)

        ### Quantize Embeddings ###
        output = self.quantize(z, 
                               compute_loss=True,
                               compute_perplexity=True)

        ### Project Quantized Back to Latent Dimension ###
        x = self.conv_latent_proj(output["quantized"])

        ### Decode ###
        reconstruction = self.forward_dec(x)
        output["reconstruction"] = reconstruction

        return output