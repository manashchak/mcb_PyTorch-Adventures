import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file
from safetensors.torch import load_file

class LoRALinear(nn.Module):

    """
    This is a basic implementation of the paper LoRA
    https://arxiv.org/pdf/2106.0968

    """
    def __init__(self, 
                 layer, 
                 rank=8, 
                 lora_alpha=1, 
                 use_rslora=True, 
                 lora_dropout=0.0,
                 b_grad=True):
        
        super().__init__()
        
        assert isinstance(layer, nn.Linear), "LoRALinear Only Converts nn.Linear Layers"
        self.rank = rank 
        self.lora_alpha = lora_alpha

        ### Store Existing Layer ###
        self.orig_layer = layer
        self.weight = layer.weight
        self.bias = layer.bias

        ### Change Gradient Flag ###
        self.orig_layer.weight.requires_grad = False
        if self.orig_layer.bias is not None:
            self.orig_layer.bias.requires_grad = b_grad

        ### Get In/Out Features (PyTorch Weight Matrix goes Out/In) ###
        out_features, in_features = self.orig_layer.weight.shape

        ### Create our Low Rank Matricies ###
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank), requires_grad=True)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features), requires_grad=True)

        ### Initialize lora_A w/ gaussian, B stays as 0s (as described in paper) ###
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        ### Create LoRA Dropout ###
        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else lambda x: x

        ### Compute Scaling for LoRA ###
        self.scaling = self.lora_alpha / self.rank**0.5 if use_rslora else self.lora_alpha / self.rank

    def __repr__(self):
        return f"LoRALinear(in_features={self.weight.shape[1]}, out_features={self.weight.shape[1]}, rank={self.rank})"
    
    def forward(self, x):

        ### Output W/ Original Weights (No Gradients) ###
        orig_output = self.orig_layer(x)

        ### Low Rank Output (With Gradients) ###
        low_rank_output = ((self.lora_dropout(x) @ self.lora_A) @ self.lora_B) * self.scaling

        ### Sum Outputs ###
        output = orig_output + low_rank_output

        ### Add Bias (if its there) ###
        if self.bias is not None:

            output = output + self.bias

        return output
    
class LoRAEmbedding(nn.Module):

    """
    This is a basic implementation of the paper LoRA
    https://arxiv.org/pdf/2106.0968

    """
    def __init__(self, 
                 layer, 
                 rank=8, 
                 lora_alpha=1, 
                 use_rslora=True, 
                 padding_idx=None):
        
        super().__init__()
        
        assert isinstance(layer, nn.Embedding), "LoRAEmbedding only works with nn.Embedding"

        self.rank = rank 
        self.lora_alpha = lora_alpha
        self.padding_idx = padding_idx

        ### These Are Our Pretrained Parameters ###
        self.orig_embed = layer
        self.weight = layer.weight

        ### Change Gradient Flag ###
        self.orig_embed.weight.requires_grad = False

        ### Get In/Out Features (PyTorch Weight Matrix goes Out/In) ###
        num_embeddings, embedding_dim = self.orig_embed.weight.shape

        ### Create our Low Rank Matricies ###
        self.lora_A = nn.Parameter(torch.zeros(num_embeddings, rank), requires_grad=True)
        self.lora_B = nn.Parameter(torch.zeros(rank, embedding_dim), requires_grad=True)

        ### Different than the paper but matches implementation ###
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        ### Compute Scaling for LoRA ###
        self.scaling = self.lora_alpha / self.rank**0.5 if use_rslora else self.lora_alpha / self.rank
    
    def __repr__(self):
        return f"LoRAEmbedding({self.weight.shape[0]}, {self.weight.shape[1]}, rank={self.rank})"
    
    def forward(self, x):

        ### Output W/ Original Weights (No Gradients) ###
        orig_output = self.orig_embed(x)

        ### Embed with Low Rank A Embedding Matrix ###
        low_rank_A_output = F.embedding(x, self.lora_A)

        ### Project Back to Embed Dim with Low Rank B ###
        low_rank_output = (low_rank_A_output @ self.lora_B) * self.scaling
    
        ### Sum Outputs ###
        output = orig_output + low_rank_output
        
        return output
    
class LoRAConv2d(nn.Module):

    """
    This is a basic implementation of the paper LoRA
    https://arxiv.org/pdf/2106.0968

    """
    def __init__(self, 
                 layer,
                 kernel_size, 
                 stride,
                 padding,
                 rank=8, 
                 lora_alpha=1, 
                 use_rslora=True, 
                 lora_dropout=0.0,
                 b_grad=True):
        
        super().__init__()

        assert isinstance(layer, nn.Conv2d), "LoRAConv2d only works with nn.Conv2d" 

        self.rank = rank 
        self.lora_alpha = lora_alpha
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        ### These Are Our Pretrained Parameters ###
        self.orig_conv = layer
        self.weight = layer.weight
        self.bias = layer.bias

        ### Change Gradient Flag ###
        self.orig_conv.weight.requires_grad = False
        if self.orig_conv.bias is not None:
            self.orig_conv.bias.requires_grad = b_grad

        ### Convolution Weight Shape ###
        out_channels, in_channels, kernel_height, kernel_width = self.orig_conv.weight.shape

        ### Create our Low Rank Matricies (Flatten kernel weights and output rank) ###
        self.lora_A = nn.Parameter(torch.zeros(rank, in_channels, kernel_height, kernel_width), requires_grad=True)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_channels), requires_grad=True)

        ### Initialize lora_A w/ gaussian, B stays as 0s (as described in paper) ###
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        ### Create LoRA Dropout ###
        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else lambda x: x

        ### Compute Scaling for LoRA ###
        self.scaling = self.lora_alpha / self.rank**0.5 if use_rslora else self.lora_alpha / self.rank

    def __repr__(self):
        return f"LoRAConv2D(in_channels={self.weight.shape[1]}, out_channels={self.weight.shape[0]}, rank={self.rank}, kernel_size={self.kernel_size})"
    
    def forward(self, x):

        ### Output W/ Original Weights (No Gradients) ###
        orig_output = self.orig_conv(x)
        
        ### Low Rank Output (With Gradients) ###
        lora_rank_A_output = F.conv2d(input=x, 
                                      weight=self.lora_A, 
                                      bias=None,
                                      stride=self.stride, 
                                      padding=self.padding)
        
        ### Permute to have rank_channels last (B x H x W x rank) ###
        lora_rank_A_output = lora_rank_A_output.permute(0,2,3,1)
        
        ### Multiply by lora_B (B x H x W x out_channels) ###
        low_rank_output = (self.lora_dropout(lora_rank_A_output) @ self.lora_B) * self.scaling

        ### Return Back to Image Shape (B x out_channels x H x W) ###
        low_rank_output = low_rank_output.permute(0,3,1,2)
      
        ### Sum Outputs ###
        output = orig_output + low_rank_output
      
        return output
    
class LoRAModel(nn.Module):

    def __init__(self, 
                 model, 
                 rank=16, 
                 lora_alpha=1.0,
                 use_rslora=True,
                 target_modules=None,
                 exclude_modules=None,
                 lora_dropout=0.0,
                 initializer_range=1.0,
                 b_grad=True):
        
        super().__init__()
        
        self.model = model
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.use_rslora = use_rslora
        self.lora_dropout = lora_dropout
        self.initializer_range = initializer_range
        self.b_grad = b_grad
        self.target_modules = target_modules if isinstance(target_modules, list) else [target_modules]
        self.exclude_modules = exclude_modules if isinstance(exclude_modules, list) else [exclude_modules]

        ### Compute Number of Trainable Parameters Before LoRA ###
        before_params = self._compute_trainable_parameters()

        ### Change Layers to LoRA ###
        self._apply_lora(self.model)

        ### Compute Number of Trainable Parameters After LoRA ###
        after_params = self._compute_trainable_parameters()

        print(f"Initial Parameters : {before_params} || LoRA Parameters : {after_params} || Trainable Proportion : {round(after_params*100/before_params, 2)}%")

    def _apply_lora(self, module):
        
        ### If no target module specified, apply to all linear/embedding layers ###
        if self.target_modules is None:
            convert_to_lora = True

        ### Recursively Go Through Model and Find Linear Layers ###
        for name, child in module.named_children():
            
            ### If our name is in our target modules and the target modules was not None we convert ###            
            if (self.target_modules is not None): 
                if (name in self.target_modules):
                    convert_to_lora = True
                else:
                    convert_to_lora = False
                
            ### If a child of this module is a linear layer, update with our LoRALinear ###
            if isinstance(child, nn.Linear):
                
                if convert_to_lora:

                    ### Create LoRA Layer for This Linear Layer ###
                    lora_layer = LoRALinear(layer=child,
                                            rank=self.rank,
                                            lora_alpha=self.lora_alpha, 
                                            use_rslora=self.use_rslora, 
                                            lora_dropout=self.lora_dropout,
                                            b_grad=self.b_grad)
                    
                    ### Replace the linear layer (identified by its name) in this module with our lora layer ###
                    setattr(module, name, lora_layer)

            ### If its an Embedding Layer then We Can Replace With Our Own LoraEmbedding ###
            elif isinstance(child, nn.Embedding):
                
                if convert_to_lora:

                    lora_layer = LoRAEmbedding(layer=child, 
                                               rank=self.rank, 
                                               lora_alpha=self.lora_alpha, 
                                               use_rslora=self.use_rslora, 
                                               padding_idx=child.padding_idx)
                    
                    ### Replace the embedding layer (identified by its name) in this module with our lora layer ###
                    setattr(module, name, lora_layer)

            elif isinstance(child, nn.Conv2d):

                if convert_to_lora:

                    lora_layer = LoRAConv2d(layer=child,
                                            kernel_size=child.kernel_size, 
                                            stride=child.stride, 
                                            padding=child.padding, 
                                            rank=self.rank, 
                                            lora_alpha=self.lora_alpha, 
                                            use_rslora=self.use_rslora, 
                                            lora_dropout=self.lora_dropout, 
                                            b_grad=self.b_grad)
                    
                    setattr(module, name, lora_layer)

            ### Else, Dig Deeper Into the Module To Search For Linear Layers (as long as module wasnt selected to be exluded) ###
            else:
                
                dig_deeper = True
                if (self.exclude_modules is not None) and (name in self.exclude_modules):
                    dig_deeper = False

                if dig_deeper:                    
                    self._apply_lora(child)

    def save_model(self, path, save_trainable_only=True):
        
        state_dict = {name: param for name, param in self.named_parameters() \
                      if (param.requires_grad and save_trainable_only)}

        save_file(state_dict, path)
        
    def load_model(self, path):
        
        state_dict = load_file(path)
        self.load_state_dict(state_dict=state_dict, strict=False)

    def _compute_trainable_parameters(self):

        total = 0
        for param in self.parameters():
            if param.requires_grad:
                total += param.numel()

        return total
    
    def forward(self, *inputs, **kwargs):
        return self.model(*inputs, **kwargs)
    
