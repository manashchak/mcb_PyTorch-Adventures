import torch
import torch.nn as nn
import torch.nn.functional as F

    
class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, kernel_size=3, upsample_factor=2):
        super(UpSampleBlock, self).__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=upsample_factor),
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding='same')
        )

    def forward(self, x):
        return self.upsample(x)
    
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
        
        x = self.downsample(x)

        return x


class DecoderBlock(nn.Module):
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
        
        super(DecoderBlock, self).__init__()
        
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
            
        self.upsample = nn.Identity()
        if add_upsample:
            self.upsample = UpSampleBlock(in_channels=out_channels, 
                                          upsample_factor=upsample_factor, 
                                          kernel_size=upsample_kernel_size)

    def forward(self, x, time_embed=None):

        for block in self.blocks:
            x = block(x, time_embed)
        
        x = self.upsample(x)

        return x

class Attention(nn.Module):
    """
    Regular Self-Attention but in this case we utilize flash_attention
    incorporated in the F.scaled_dot_product_attention to speed up our training. 
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

        ### Attention Projections ###
        self.groupnorm = nn.GroupNorm(num_channels=embedding_dimension, num_groups=groupnorm_groups, eps=1e-6)
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
    
    def _seq2img(self, x):
        """
        (B x H*W x C) -> (B x C x H x W)
        """
        batch, seq_len, channels = x.shape
        h = w = int(seq_len**0.5)

        x = x.transpose(-1,-2).reshape(batch, channels, h, w)

        return x

    def forward(self, 
                src, 
                tgt=None, 
                attention_mask=None, 
                causal=False):

        """
        By default, self-attention will be computed on src (with optional causal and/or attention mask). If tgt is provided, then
        we are doing cross attention. In cross attention, an attention_mask can be used, but no causal mask can be applied.

        Self-Attention:
            - Compute Self Attention on the src Tensor
                - One new step to include though is reshaping our src (image)
                  from (B x C x H x W) -> (B x H*W x C) before doing attention
        
        Cross Attention
            - src: Our text Context (B x L x E)
            - tgt: What we want to weight against our src and output
                - One new step to include though is reshaping our tgt (image)
                  from (B x C x H x W) -> (B x H*W x C) before doing attention

        """

        ### Grab Shapes ###
        batch = src.shape[0]

        ### If target is not provided, we are doing self attention (with potential causal mask) ###    
        if tgt is None:

            residual = src
            
            ### Reshape from Img Dim to Seq Dim ###
            src, src_len = self._img2seq(src)

            ### QKV Projection ###
            q = self.q_proj(src).reshape(batch, src_len, self.num_heads, self.head_dim).transpose(1,2).contiguous()
            k = self.k_proj(src).reshape(batch, src_len, self.num_heads, self.head_dim).transpose(1,2).contiguous()
            v = self.v_proj(src).reshape(batch, src_len, self.num_heads, self.head_dim).transpose(1,2).contiguous()

            ### Implementing Attention Mask (but it'll never be used for Self-Attention) ####
            if attention_mask is not None:
                attention_mask = attention_mask.bool()
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1).repeat(1,1,src_len,1)

            attention_out = F.scaled_dot_product_attention(q,k,v, 
                                                           attn_mask=attention_mask, 
                                                           dropout_p=self.attn_dropout if self.training else 0.0, 
                                                           is_causal=causal)
        
        ### If target is provided then we are doing cross attention ###
        ### Our query will be the target and we will be crossing it with the encoder source (keys and values) ###
        ### The src_attention_mask will still be the mask here, just repeated to the target size ###

        ### In our case the src is the Text Encodings and tgt is the Image embeddings ###
        else:

            residual = tgt
  
            ### Reshape from Img Dim to Seq Dim ###
            tgt, tgt_len = self._img2seq(tgt)

            batch, src_len, embed_dim = src.shape

            ### Compute Queries on our Image Tensor ###
            q = self.q_proj(tgt).reshape(batch, tgt_len, self.num_heads, self.head_dim).transpose(1,2).contiguous()
            
            ### Compute Keys and Values on our Text Embeddings ###
            k = self.k_proj(src).reshape(batch, src_len, self.num_heads, self.head_dim).transpose(1,2).contiguous()
            v = self.v_proj(src).reshape(batch, src_len, self.num_heads, self.head_dim).transpose(1,2).contiguous()

            ### This is our src attention mask (on the text encoding) ###
            if attention_mask is not None:
                attention_mask = attention_mask.bool()
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1).repeat(1,1,tgt_len,1)

            attention_out = F.scaled_dot_product_attention(q,k,v,
                                                           attn_mask=attention_mask, 
                                                           dropout_p=self.attn_dropout if self.training else 0.0, 
                                                           is_causal=False)

        ### Reshape and Project ###
        attention_out = attention_out.transpose(1,2).flatten(2)
        attention_out = self.out_proj(attention_out)

        ### Return Back to Image Dimensions (B x H*W x C) -> (B x C x H x W) ###
        output = self._seq2img(attention_out)

        if self.attn_residual:
            output = output + residual

        return output

class AttentionResidualBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 dropout_p = 0.0, 
                 num_layers = 1,
                 groupnorm_groups = 32,
                 time_embed_proj = False, 
                 time_embed_dim = 128,
                 norm_eps=1e-6,
                 attention_head_dim=1,
                 attention_residual_connection=True):
        
        super(AttentionResidualBlock, self).__init__()
        
        self.resnets = nn.ModuleList()
        self.attentions = nn.ModuleList()

        ### There is Always One Residual Block ###
        self.resnets.append(
            ResidualBlock(
                 in_channels=in_channels, 
                 out_channels=in_channels, 
                 dropout_p=dropout_p, 
                 groupnorm_groups=groupnorm_groups,
                 time_embed_proj=time_embed_proj, 
                 time_embed_dim=time_embed_dim,
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
                    attention_residual_connection=attention_residual_connection
                )
            )

            self.resnets.append(
                ResidualBlock(
                    in_channels=in_channels, 
                    out_channels=in_channels, 
                    dropout_p=dropout_p, 
                    groupnorm_groups=groupnorm_groups,
                    time_embed_proj=time_embed_proj, 
                    time_embed_dim=time_embed_dim,
                    norm_eps=norm_eps
                )
            )

    def forward(self, 
                x, 
                time_embed=None, 
                text_embed=None, 
                attention_mask=None):

        x = self.resnets[0](x, time_embed=time_embed)

        for attn, res in zip(self.attentions, self.resnets[1:]):
            
            ### If we dont have text, then we are doing Self-Attenion on our Images as src ###
            if text_embed is None:
                x = attn(src=x)
            
            ### If we do have text, then we are doing cross attention with text as src and images as tgt ###
            else:
                x = attn(src=text_embed, tgt=x, attention_mask=attention_mask)

            x = res(x, time_embed)
        
        return x







