"""
Very close to my LoRA Implementation found at:
https://github.com/priyammaz/PyTorch-Adventures/tree/main/PyTorch%20Tools/LoRA

And also, lots of inspiration and help for the code was taken from:

LoRA Implementation that follows:
    - Paper: https://arxiv.org/pdf/2106.09685
    - Repo: https://github.com/microsoft/LoRA/

QLoRA Implementation:
    - Paper: https://arxiv.org/pdf/2305.14314

Also, implementation by michaelnny was super helpful!
    - https://github.com/michaelnny/QLoRA-LLM/
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Literal, Union
import bitsandbytes as bnb
import bitsandbytes.nn as bnn
from safetensors.torch import save_file

class LoRALayerBase:

    """
    Base Class for LoRA Layer with all the attributes
    we will need across all our layers

    Args: 
        - r: rank for LoRA
        - lora_alpha: LoRA constant
        - lora_dropout: Dropout Probability on LoRA
        - use_rslora: Scale lora_alpha by root of the rank

    """

    def __init__(self, 
                 rank=8, 
                 lora_alpha=8, 
                 lora_dropout=0.0, 
                 use_rslora=True,):
        
        self.rank = rank
        self.lora_alpha = lora_alpha 
        self.scaling = self.lora_alpha / self.rank**0.5 if use_rslora else self.lora_alpha / self.rank
        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else lambda x: x

    def _load_pretrained_weights(self, state_dict):

        """
        Weights are in the following formats:
            - nn.Linear: [Out x In]
            - nn.Conv2d: [out_c x in_c x k_h x k_w] => [out_c x in_c*k_h*k_w]
            - nn.Embedding: [vocab_size x embed_dim]
        """
        self.weight.data = state_dict["weight"].flatten(1)
        if "bias" in state_dict.keys():
            if self.bias is None:
                raise Exception("Loading layer with bias to a layer without bias enabled")
            self.bias.data = state_dict["bias"]
        


class QLoRALinear(bnn.Linear4bit, LoRALayerBase):

    """
    QLoRA Implementation on a Linear Layer, basically a wrapper on the
    bnn.Linear4bit module, with the additional lora_A/lora_B layers
    """

    def __init__(self, 
                 in_features, 
                 out_features,
                 bias=True, 
                 rank=8, 
                 lora_alpha=8, 
                 lora_dropout=0.0,
                 use_rslora=True,
                 **kwargs):
        
        ### Initialize Inherited Classes ###
        bnn.Linear4bit.__init__(self, in_features, out_features, bias=bias, **kwargs)
        LoRALayerBase.__init__(self,
                               rank=rank, 
                               lora_alpha=lora_alpha, 
                               lora_dropout=lora_dropout, 
                               use_rslora=use_rslora)
        
        assert rank > 0, "If Rank is 0, Why are you doing LoRA?"

        ### Disable Gradients on Linear Layer Weights ###
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        
        ### Define LoRA Layers (shape is [in_features, out_features] ###
        ### Normally Weight matricies are [out_features, in_features] ###
        ### This just saves us a few transposes ###
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        ### Initialize lora_A. In the paper they use normal, but in the ###
        ### implementation they use kaiming_uniform_, so lets use that! ###
        nn.init.kaiming_normal_(self.lora_A, a=math.sqrt(5))
        
    def _merge_weights(self):

        """
        xW^T + xAB = x(W^T + AB)
        """

        ### Dequantize Weights ###
        if self.quant_state is not None:
            weight = bnb.functional.dequantize_4bit(self.weight.data.clone(), self.quant_state)
        else:
            weight = self.weight.data.clone()

        ### Merge Weights ###
        merged_weight = weight.data + (self.lora_A @ self.lora_B).T * self.scaling

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
        orig_layer_out = bnn.Linear4bit.forward(self, x)

        ### LoRA Layers ###
        lora_mult = (self.lora_A @ self.lora_B) * self.scaling
        low_rank_out = self.lora_dropout(x) @ lora_mult

        ### Sum Outputs ###
        output = orig_layer_out + low_rank_out
        
        return output
    
class LoRAEmbedding(nn.Embedding, LoRALayerBase):
    
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
                 **kwargs):
        
        ### Initialize Inherited Classes ###
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoRALayerBase.__init__(self,
                               rank=rank, 
                               lora_alpha=lora_alpha, 
                               lora_dropout=lora_dropout, 
                               use_rslora=use_rslora)
        
        assert rank > 0, "If Rank is 0, Why are you doing LoRA?"

        ### Disable Gradients on Linear Layer Weights ###
        self.weight.requires_grad = False
        
        ### Define LoRA Layers ###
        self.lora_A = nn.Parameter(torch.zeros(self.num_embeddings, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, self.embedding_dim))

        ### Initialize lora_A. In the paper they use normal, but in the ###
        ### implementation they use kaiming_uniform_, so lets use that! ###
        nn.init.kaiming_normal_(self.lora_A, a=math.sqrt(5))

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

class QLoRAConv2d(bnn.Linear4bit, LoRALayerBase):
    
    """
    QLoRA Implementation on an Conv2d Layer. Now there is no bnn.Conv2d at the 
    time of making this, so we need to do a bit more work. Convolutions are 
    implemented with linear layers underneath the hood, so we will leverage
    the Im2Col algorithm to do this operation with quantized linear weights. 

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
                 **kwargs):
        
        ### Conv to Linear Conversion ###
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding    

        ### Initialize Inherited Classes ###
        bnn.Linear4bit.__init__(self, 
                                input_features=(self.kernel_size[0] * self.kernel_size[1] * self.in_channels),
                                output_features=self.out_channels, 
                                bias=bias, 
                                **kwargs)
        
        LoRALayerBase.__init__(self,
                               rank=rank, 
                               lora_alpha=lora_alpha, 
                               lora_dropout=lora_dropout, 
                               use_rslora=use_rslora)
        
        assert rank > 0, "If Rank is 0, Why are you doing LoRA?"

        ### Disable Gradients on Linear Layer Weights/Biases ###
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        
        ### Create Our Low Rank Matricies (Flatten Kernel Weights and Output Rank) ###
        self.lora_A = nn.Parameter(torch.zeros(rank, self.in_channels, *self.kernel_size))
        self.lora_B = nn.Parameter(torch.zeros(rank, self.out_channels))

        ### Initialize lora_A. In the paper they use normal, but in the ###
        ### implementation they use kaiming_uniform_, so lets use that! ###
        nn.init.kaiming_normal_(self.lora_A, a=math.sqrt(5))

    def _merge_weights(self):

        """
        xW^T + xAB = x(W^T + AB)
        """

        ### Dequantize Weights ###
        if self.quant_state is not None:
            weight = bnb.functional.dequantize_4bit(self.weight.data.clone(), self.quant_state)
        else:
            weight = self.weight.data.clone()
        
        ### Reshape Linear Weight Back to Convolution Weight Shape ###
        weight = weight.reshape(self.out_channels, self.in_channels, *self.kernel_size)
        
        ### (rank x in_chan x k_h, k_w) -> (rank x in_chan*k_h*k_w) ###    
        lora_A_flatten = self.lora_A.flatten(1)

        ### Matmul with lora_B Transposed (lora_B is rank x out_channels) -> (out_channels x rank) ###
        lora_mult = (self.lora_B.T @ lora_A_flatten) * self.scaling
        
        ### Place Back into Conv Weight Shape: (ou_chan x in_chan*k_h*k_w) -> (out_chan x in_chan x k_h x k_w) ###
        lora_mult = lora_mult.reshape(self.out_channels, self.in_channels, *self.kernel_size)

        ### Merge ###
        merged_weight = weight + lora_mult

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
     
        ### Im2Col Algorithm (B x C x H x W) -> (B x C*H*W x num_patches) ###
        unfolded_x = F.unfold(input=x, 
                              kernel_size=self.kernel_size,
                              stride=self.stride, 
                              padding=self.padding)

        ### Make our (C*H*W) dimension last ###
        unfolded_x = unfolded_x.transpose(1,2)

        ### Project with Linear Layer ###
        proj_unfolded_x = bnn.Linear4bit.forward(self, unfolded_x)

        ### Reshape from (B x C*H*W x oc) -> (B x oc x C*H*W) ###
        proj_unfolded_x = proj_unfolded_x.transpose(1,2)
        
        ### Compute Output Shape (with Convolution Formula) ###
        H_out = (x.shape[2] + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        W_out = (x.shape[3] + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1

        ### Place Back in the Image Format ###
        quant_out = proj_unfolded_x.reshape(-1, self.out_channels, H_out, W_out)

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
        output = quant_out + low_rank_output

        return output
    
@dataclass
class QLoraConfig:
    
    rank: int = 8
    target_modules: Optional[Union[list[str], str]] = None
    exclude_modules: Optional[Union[list[str], str]] = None
    lora_alpha: float = 8.0
    lora_dropout: float = 0.0
    bias: Literal["none", "all", "lora_only"] = "none"
    use_rslora: bool = True

class QLoraModel(nn.Module):

    def __init__(self, model, config):
        super(QLoraModel, self).__init__()

        self.lora_model = model
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

        ### Get Number of Trainable Parameters ###
        orig_trainable_params = self._compute_trainable_parameters()

        ### Disable All Grads in Model ###
        self._disable_all_grads()

        ### Apply LoRA to Target Modules ###
        self._apply_lora(self.lora_model)

        ### Toggle Bias Gradients ###
        self._toggle_bias_grad()

        ### Get LoRA Trainable Parameters ###
        lora_trainable_params = self._compute_trainable_parameters()

        print_string = ""
        print_string += f"Initial Parameters : {orig_trainable_params} || "
        print_string += f"LoRA Parameters : {lora_trainable_params} || "
        print_string += f"Trainable Proportion : {round(lora_trainable_params*100/orig_trainable_params, 2)}%"

        print(print_string)

    def forward(self, *inputs, **kwargs):

        """
        The forward function is the same, so a catchall here
        to pass all of our stuff from the forward methdod into
        our models forward method
        """

        return self.lora_model(*inputs, **kwargs)
    
    def _exclude_module_name_check(self, name):
        return any([ex in name for ex in self.config.exclude_modules])
    
    def _target_module_name_check(self, name):
        return any([tgt in name for tgt in self.config.target_modules])

    def _apply_lora(self, module):

        """
        Method to recursively replace all the layers in a model with LoraLayers
        """

        ### Recursively Go Through Model and Find Layers To Convert ###
        for name, child in module.named_children():
            
            ### Check if Layer is Included to Convert to LoRA ###
            if self._target_module_name_check(name):
                
                ### Convert Linear to LoRA ###
                if isinstance(child, nn.Linear):

                    new_layer = QLoRALinear(in_features=child.in_features, 
                                            out_features=child.out_features, 
                                            bias=True if child.bias is not None else False,
                                            rank=self.config.rank,
                                            lora_alpha=self.config.lora_alpha, 
                                            lora_dropout=self.config.lora_dropout, 
                                            use_rslora=self.config.use_rslora)

                    new_layer._load_pretrained_weights(child.state_dict())
                    setattr(module, name, new_layer)

                elif isinstance(child, nn.Conv2d):

                    new_layer = QLoRAConv2d(in_channels=child.in_channels, 
                                            out_channels=child.out_channels, 
                                            kernel_size=child.kernel_size, 
                                            stride=child.stride, 
                                            padding=child.padding, 
                                            bias=True if child.bias is not None else False,
                                            rank=self.config.rank, 
                                            lora_alpha=self.config.lora_alpha, 
                                            lora_dropout=self.config.lora_dropout, 
                                            use_rslora=self.config.use_rslora)
                    
                    new_layer._load_pretrained_weights(child.state_dict())
                    setattr(module, name, new_layer)

                elif isinstance(child, nn.Embedding):

                    new_layer = LoRAEmbedding(num_embeddings=child.num_embeddings, 
                                             embedding_dim=child.embedding_dim, 
                                             rank=self.config.rank, 
                                             lora_alpha=self.config.lora_alpha, 
                                             lora_dropout=self.config.lora_dropout, 
                                             use_rslora=self.config.use_rslora)
                    
                    new_layer._load_pretrained_weights(child.state_dict())
                    setattr(module, name, new_layer)

            ### If there are more children and its not an exclusion module, Recurse into them ###
            if (len(list(child.children())) > 0) and not any([ex in name for ex in self.config.exclude_modules]):
                self._apply_lora(child)

    def _toggle_bias_grad(self):

        """
        Method to turn off bias gradients depending on:
            - none:  Dont train any biases
            - all: train all biases
            - lora_only: train biases only in lora layers
        """

        for name, param in self.lora_model.named_parameters():
            
            ### Dont want to disable gradients for Excluded Layers ###
            if not self._exclude_module_name_check(name):
                if ".bias" in name:
                    if self.config.bias == "none":
                        param.requires_grad = False
                    elif self.config.bias == "all":
                        param.requires_grad = True
                    elif (self.config.bias == "lora_only") and self._target_module_name_check(name):
                        param.requires_grad = True

    def _disable_all_grads(self):
        
        """
        Helper function to disable all gradients 
        """

        for name, param in self.lora_model.named_parameters():

            ### If not in exclude modules, turn off gradients ###
            if not self._exclude_module_name_check(name):
                param.requires_grad = False

    def _compute_trainable_parameters(self):

        """
        Helper function to compute all parameters with gradients
        """

        total_learnable_params = 0
        for param in self.lora_model.parameters():
            if param.requires_grad:
                total_learnable_params += param.numel()

        return total_learnable_params

    def _merge_weights(self, module):

        """
        Recursively trigger weight merging and replace in model 
        """

        for name, child in module.named_children():

            if isinstance(child, (QLoRALinear, LoRAEmbedding, QLoRAConv2d)):
                 
                 ### Merge the Layer ###
                 merged_layer = child._merge_weights()

                 ### Replace LoRA Layer with Merged ###
                 setattr(module, name, merged_layer)

            else:

                if len(list(child.children())) > 0:
                    self._merge_weights(child)

    def save_model(self, path, merge_weights=False):

        """
        Method to save model safetensors to the given path
            - merge_weights -> True: Merge LoRA weights and save
            - merge_weights -> False: Only save trainable weights
        """

        def _detach_cpu(param):
            return param.detach().cpu()
        
        ### Create New Model with Merged Weights ###
        if merge_weights:
            
            ### Merge Weights ###
            self._merge_weights(self.lora_model)

            ### If Merged, then state_dict will have ALL Weights ###
            ### When merging weights, we can remove "lora_model." from the name ###
            ### because we can just load these weights into the original model ###
            state_dict = {name.replace("lora_model.", ""): _detach_cpu(param) for (name, param) in self.named_parameters()}

        ### Otherwise Save only the parameters we trained, everything else is frozen ###
        ### and can be taken from the original model weights ###
        ### To load these weights, the model needs to be wrapped in LoraModel ###
        else:

            state_dict = {name: _detach_cpu(param) for (name, param) in self.named_parameters() if (param.requires_grad)}

        save_file(state_dict, path)

if __name__ == "__main__":

    from transformers import AutoModelForImageClassification

    model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")
    print(model)

    lora_config = QLoraConfig(exclude_modules="classifier", 
                             target_modules=["query", "key", "value", "dense", "projection"],
                             bias="lora_only")
    lora_model = QLoraModel(model, lora_config).to("cuda")
    print(lora_model.lora_model.vit.encoder.layer[0].attention.output.dense.bias)
    