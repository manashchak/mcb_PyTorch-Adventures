"""
LoRA Implementation that follows:
    - Paper: https://arxiv.org/pdf/2106.09685
    - Repo: https://github.com/microsoft/LoRA/

Also, implementation by michaelnny was super helpful!
    - https://github.com/michaelnny/QLoRA-LLM/
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Literal, Union
from safetensors.torch import save_file, load_file

class LoRALayerConfig:

    """
    Default Class for LoRA Layer with all the attributes
    we will need across all our layers

    Args: 
        - r: rank for LoRA
        - lora_alpha: LoRA constant
        - lora_dropout: Dropout Probability on LoRA
        - use_rslora: Scale lora_alpha by root of the rank
        - lora_dtype: What precision type to train our LoRA Matricies
    """

    def __init__(self, 
                 rank=8, 
                 lora_alpha=8, 
                 lora_dropout=0.0, 
                 use_rslora=True,
                 lora_dtype=torch.float32):
        
        self.rank = rank
        self.lora_alpha = lora_alpha 
        self.scaling = self.lora_alpha / self.rank**0.5 if use_rslora else self.lora_alpha / self.rank
        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else lambda x: x
        self.lora_dtype = lora_dtype

class LoRALinear(nn.Linear, LoRALayerConfig):

    """
    LoRA Implementation on a Linear Layer, basically a wrapper on the
    nn.Linear module, with the additional lora_A/lora_B layers
    """

    def __init__(self, 
                 in_features, 
                 out_features,
                 bias=True, 
                 rank=8, 
                 lora_alpha=8, 
                 lora_dropout=0.0,
                 use_rslora=True,
                 lora_dtype=torch.float32,
                 **kwargs):
        
        ### Initialize Inherited Classes ###
        nn.Linear.__init__(self, in_features, out_features, bias=bias, **kwargs)
        LoRALayerConfig.__init__(self,
                                 rank=rank, 
                                 lora_alpha=lora_alpha, 
                                 lora_dropout=lora_dropout, 
                                 use_rslora=use_rslora, 
                                 lora_dtype=lora_dtype)
        
        assert rank > 0, "If Rank is 0, Why are you doing LoRA?"

        ### Disable Gradients on Linear Layer Weights/Biases ###
        self.weight.requires_grad = False
        
        ### Define LoRA Layers (shape is [in_features, out_features] ###
        ### Normally Weight matricies are [out_features, in_features] ###
        ### This just saves us a few transposes ###
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank, dtype=lora_dtype))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features, dtype=lora_dtype))

        ### Initialize lora_A. In the paper they use normal, but in the ###
        ### implementation they use kaiming_uniform_, so lets use that! ###
        nn.init.kaiming_normal_(self.lora_A, a=math.sqrt(5))

    def _load_pretrained_weights(self, state_dict):
        self.weight.data = state_dict["weight"]
        if "bias" in state_dict.keys():
            self.bias.data = state_dict["bias"]

    def _merge_weights(self):

        """
        xW^T + xAB = x(W^T + AB)
        """

        ### Merge Weights ###
        merged_weight = self.weight.data + (self.lora_A @ self.lora_B).T * self.scaling

        ### Store Weights ###
        state_dict = {"weight": merged_weight}
        if self.bias is not None:
            state_dict["bias"] = self.bias

        ### Load Weights to New Linear Layer ###
        merged_linear = nn.Linear(self.in_features, 
                                  self.out_features,
                                  bias=True if self.bias is not None else False)
        
        merged_linear.load_state_dict(state_dict)

        return merged_linear

    def forward(self, x):
        
        ### Pass Through Original Weights ###
        orig_layer_out = F.linear(x, self.weight, bias=self.bias)
        
        ### LoRA Layers ###
        lora_mult = (self.lora_A @ self.lora_B) * self.scaling
        low_rank_out = self.lora_dropout(x) @ lora_mult

        ### Sum Outputs ###
        output = orig_layer_out + low_rank_out
        
        return output
    
class LoRAEmbedding(nn.Embedding, LoRALayerConfig):
    
    """
    LoRA Implementation on an Embedding Layer, basically a wrapper on the
    nn.Embedding module, with the additional lora_A/lora_B layers
    """

    def __init__(self, 
                 num_embeddings, 
                 embedding_dim, 
                 rank=8, 
                 lora_alpha=8, 
                 lora_dropout=0.0,
                 use_rslora=True,
                 lora_dtype=torch.float32,
                 **kwargs):
        
        ### Initialize Inherited Classes ###
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoRALayerConfig.__init__(self,
                                 rank=rank, 
                                 lora_alpha=lora_alpha, 
                                 lora_dropout=lora_dropout, 
                                 use_rslora=use_rslora, 
                                 lora_dtype=lora_dtype)
        
        assert rank > 0, "If Rank is 0, Why are you doing LoRA?"

        ### Disable Gradients on Linear Layer Weights ###
        self.weight.requires_grad = False
        
        ### Define LoRA Layers ###
        self.lora_A = nn.Parameter(torch.zeros(self.num_embeddings, rank, dtype=lora_dtype))
        self.lora_B = nn.Parameter(torch.zeros(rank, self.embedding_dim, dtype=lora_dtype))

        ### Initialize lora_A. In the paper they use normal, but in the ###
        ### implementation they use kaiming_uniform_, so lets use that! ###
        nn.init.kaiming_normal_(self.lora_A, a=math.sqrt(5))

    def _load_pretrained_weights(self, state_dict):
        self.weight.data = state_dict["weight"]

    def _merge_weights(self):
        """
        xW^T + xAB = x(W^T + AB)
        """

        ### Merge Weights ###
        merged_weights = self.weight.data + (self.lora_A @ self.lora_B) * self.scaling

        ### Store in Embedding Layer ###
        state_dict = {"weight": merged_weights}
        merged_emb = nn.Embedding(self.num_embeddings, self.embedding_dim)
        merged_emb.load_state_dict(state_dict)

        return merged_emb


    def forward(self, x):

        ### Pass Through Original Weights ###
        orig_layer_out = F.embedding(input=x, 
                                     weight=self.weight,
                                     padding_idx=self.padding_idx, 
                                     max_norm=self.max_norm, 
                                     norm_type=self.norm_type, 
                                     scale_grad_by_freq=self.scale_grad_by_freq, 
                                     sparse=self.sparse)
        
        ### Pass Through lora_A ###
        low_rank_A_output = F.embedding(input=x, 
                                        weight=self.lora_A, 
                                        padding_idx=self.padding_idx, 
                                        max_norm=self.max_norm, 
                                        norm_type=self.norm_type, 
                                        scale_grad_by_freq=self.scale_grad_by_freq, 
                                        sparse=self.sparse)
        
        ### Project lora_A rank dimension to embedding dimension w/ lora_B ###
        low_rank_output = (low_rank_A_output @ self.lora_B) * self.scaling

        ### Sum Outputs ###
        output = orig_layer_out + low_rank_output

        return output

class LoRAConv2d(nn.Conv2d, LoRALayerConfig):
    
    """
    LoRA Implementation on an Conv2d Layer, basically a wrapper on the
    nn.Conv2d module, with the additional lora_A/lora_B layers

    This is a simplified convolution implementation, take a look at the original
    implementation for more details!
    """

    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 stride=1, 
                 padding=0, 
                 bias=True,
                 rank=8, 
                 lora_alpha=8, 
                 lora_dropout=0.0,
                 use_rslora=True,
                 lora_dtype=torch.float32,
                 **kwargs):
        
        ### Initialize Inherited Classes ###
        nn.Conv2d.__init__(self,
                           in_channels=in_channels, 
                           out_channels=out_channels, 
                           kernel_size=kernel_size, 
                           stride=stride, 
                           padding=padding, 
                           bias=bias, 
                           **kwargs)
        LoRALayerConfig.__init__(self,
                                 rank=rank, 
                                 lora_alpha=lora_alpha, 
                                 lora_dropout=lora_dropout, 
                                 use_rslora=use_rslora, 
                                 lora_dtype=lora_dtype)
        
        assert rank > 0, "If Rank is 0, Why are you doing LoRA?"

        ### Disable Gradients on Linear Layer Weights/Biases ###
        self.weight.requires_grad = False
        
        ### Create Our Low Rank Matricies (Flatten Kernel Weights and Output Rank) ###
        self.lora_A = nn.Parameter(torch.zeros(rank, self.in_channels, *self.kernel_size, dtype=lora_dtype))
        self.lora_B = nn.Parameter(torch.zeros(rank, self.out_channels, dtype=lora_dtype))

        ### Initialize lora_A. In the paper they use normal, but in the ###
        ### implementation they use kaiming_uniform_, so lets use that! ###
        nn.init.kaiming_normal_(self.lora_A, a=math.sqrt(5))

    def _load_pretrained_weights(self, state_dict):
        self.weight.data = state_dict["weight"]
        if "bias" in state_dict.keys():
            self.bias.data = state_dict["bias"]

    def _merge_weights(self):

        """
        xW^T + xAB = x(W^T + AB)
        """
        ### (rank x in_chan x k_h, k_w) -> (rank x in_chan*k_h*k_w) ###    
        lora_A_flatten = self.lora_A.flatten(1)

        ### Matmul with lora_B Transposed (lora_B is rank x out_channels) -> (out_channels x rank) ###
        lora_mult = (self.lora_B.T @ lora_A_flatten) * self.scaling
        
        ### Place Back into Conv Weight Shape: (ou_chan x in_chan*k_h*k_w) -> (out_chan x in_chan x k_h x k_w) ###
        lora_mult = lora_mult.reshape(self.out_channels, self.in_channels, *self.kernel_size)

        ### Merge ###
        merged_weight = self.weight.data + lora_mult

        ### Store Weights ###
        state_dict = {"weight": merged_weight}
        if self.bias is not None:
            state_dict["bias"] = self.bias

        ### Load Weights to New Linear Layer ###
        merged_conv = nn.Conv2d(self.in_channels, 
                                self.out_channels,
                                kernel_size=self.kernel_size, 
                                stride=self.stride, 
                                padding=self.padding,
                                bias=True if self.bias is not None else False)
        
        merged_conv.load_state_dict(state_dict)

        return merged_conv

    def forward(self, x):

        ### Output W/ Original Weights ###
        orig_layer_out = F.conv2d(input=x, 
                                  weight=self.weight, 
                                  bias=self.bias,
                                  stride=self.stride, 
                                  padding=self.padding)
        
        ### Low Rank Outputs ###
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
        output = orig_layer_out + low_rank_output

        return output

@dataclass
class LoraConfig:
    
    rank: int = 8
    target_modules: Optional[Union[list[str], str]] = None
    exclude_modules: Optional[Union[list[str], str]] = None
    lora_alpha: float = 8.0
    lora_dropout: float = 0.0
    bias: Literal["none", "all", "lora_only"] = "none"
    use_rslora: bool = True
    lora_dtype: Literal["fp32", "fp16", "bf16"] = "fp32"

dtype_precision_dict = {"fp32": torch.float32, 
                        "fp16": torch.float16, 
                        "bf16": torch.bfloat16}

class LoraModel(nn.Module):

    def __init__(self, model, config):
        super(LoraModel, self).__init__()

        self.model = model
        self.config = config

        ### Ensure Taraget Modules/Exclude Modules are Lists ###
        if self.config.target_modules is None:
            self.config.target_modules = []
        elif isinstance(self.config.target_modules, str):
            self.config.target_modules = [self.config.target_modules]
        if self.config.exclude_modules is None:
            self.config.exclude_modules = []
        elif isinstance(self.config.exclude_modules, str):
            self.config.exclude_modules = [self.config.exclude_modules]

        ### Disable All Grads in Model ###
        self._disable_all_grads()

        ### Apply LoRA to Target Modules ###
        self._apply_lora(model)

        ### Toggle Bias Gradients ###
        self._toggle_bias_grad()

    def _apply_lora(self, module):

        ### Recursively Go Through Model and Find Layers To Convert ###
        for name, child in module.named_children():
            
            ### Check if Layer is Included to Convert to LoRA ###
            if any([inc in name for inc in self.config.target_modules]):
                
                ### Convert Linear to LoRA ###
                if isinstance(child, nn.Linear):

                    new_layer = LoRALinear(in_features=child.in_features, 
                                           out_features=child.out_features, 
                                           bias=True if child.bias is not None else False,
                                           rank=self.config.rank,
                                           lora_alpha=self.config.lora_alpha, 
                                           lora_dropout=self.config.lora_dropout, 
                                           use_rslora=self.config.use_rslora,
                                           lora_dtype=dtype_precision_dict[self.config.lora_dtype])

                    new_layer._load_pretrained_weights(child.state_dict())
                    setattr(module, name, new_layer)

                elif isinstance(child, nn.Conv2d):

                    new_layer = LoRAConv2d(in_channels=child.in_channels, 
                                           out_channels=child.out_channels, 
                                           kernel_size=child.kernel_size, 
                                           stride=child.stride, 
                                           padding=child.padding, 
                                           bias=True if child.bias is not None else False,
                                           rank=self.config.rank, 
                                           lora_alpha=self.config.lora_alpha, 
                                           lora_dropout=self.config.lora_dropout, 
                                           use_rslora=self.config.use_rslora,
                                           lora_dtype=dtype_precision_dict[self.config.lora_dtype])
                    
                    new_layer._load_pretrained_weights(child.state_dict())
                    setattr(module, name, new_layer)

                elif isinstance(child, nn.Embedding):

                    new_layer = LoRAEmbedding(num_embeddings=child.num_embeddings, 
                                              embedding_dim=child.embedding_dim, 
                                              rank=self.config.rank, 
                                              lora_alpha=self.config.lora_alpha, 
                                              lora_dropout=self.config.lora_dropout, 
                                              use_rslora=self.config.use_rslora,
                                              lora_dtype=dtype_precision_dict[self.config.lora_dtype])
                    
                    new_layer._load_pretrained_weights(child.state_dict())
                    setattr(module, name, new_layer)

            ### If there are more children, Recurse into them ###
            if len(list(child.children())) > 0:
                self._apply_lora(child)

    def _toggle_bias_grad(self):

        for name, param in self.model.named_parameters():
            
            ### Dont want to disable gradients for Excluded Layers ###
            if not any([ex in name for ex in self.config.exclude_modules]):
                if ".bias" in name:
                    if self.config.bias == "none":
                        param.requires_grad = False
                    elif self.config.bias == "all":
                        param.requires_grad = True
                    elif (self.config.bias == "lora_only") and ("lora_" in name):
                        param.requires_grad = True

    def _disable_all_grads(self):
        
        for name, param in self.model.named_parameters():

            ### If not in exclude modules, turn off gradients ###
            if not any([ex in name for ex in self.config.exclude_modules]):
                param.requires_grad = False

    def _compute_trainable_parameters(self):

        total_learnable_params = 0
        for param in self.model.parameters():
            if param.requires_grad:
                total_learnable_params += param.numel()

        return total_learnable_params


if __name__ == "__main__":

    from transformers import AutoModelForImageClassification

    model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")
    lora_config = LoraConfig(exclude_modules="classifier", 
                             target_modules=["projection", "query", "key", "value", "dense"])
    lora_model = LoraModel(model, lora_config)
    
    for name, param in lora_model.named_parameters():
        if "bias" in name:
            print(name, param.requires_grad)