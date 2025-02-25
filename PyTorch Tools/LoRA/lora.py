import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file, load_file

class LoRALinear(nn.Module):

    def __init__(self, 
                 layer, 
                 rank=8, 
                 lora_alpha=1, 
                 use_rslora=True, 
                 lora_dropout=0.0,
                 b_grad=True, 
                 lora_dtype=torch.float32):
        
        super().__init__()

        assert isinstance(layer, nn.Linear), "LoRALinear Only Converts nn.Linear Layers"
        self.rank = rank 
        self.lora_alpha = lora_alpha
        self.lora_dtype = lora_dtype

        # Store Original Layer
        self.orig_layer = layer
        self.weight = layer.weight
        self.bias = layer.bias

        ### Update Gradient Flag ###
        self.orig_layer.weight.requires_grad = False
        self.orig_layer.bias.requires_grad = b_grad

        # Get In/Out Features
        self.out_features, self.in_features = self.orig_layer.weight.shape

        # Create Low-Rank Matrices
        self.lora_A = nn.Parameter(torch.zeros(self.in_features, rank, dtype=lora_dtype), requires_grad=True)
        self.lora_B = nn.Parameter(torch.zeros(rank, self.out_features, dtype=lora_dtype), requires_grad=True)

        # Initialize lora_A with Gaussian, B stays as 0s
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # Create LoRA Dropout
        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else lambda x: x

        # Compute Scaling for LoRA
        self.scaling = self.lora_alpha / self.rank**0.5 if use_rslora else self.lora_alpha / self.rank

    def __repr__(self):
        return f"LoRALinear(in_features={self.in_features}, out_features={self.out_features}, rank={self.rank})"
    
    def _merge_weights(self):
        """
        xW^T + xAB = x(W^T + AB)
        """
        merged_weight = self.orig_layer.weight.data + (self.lora_A @ self.lora_B).T * self.scaling

        merged_layer = nn.Linear(
            self.orig_layer.weight.shape[1],
            self.orig_layer.weight.shape[0],
            bias=True if self.bias is not None else False
        )

        merged_layer.weight.data = merged_weight

        if self.bias is not None:
            merged_layer.bias.data = self.bias

        return merged_layer

    def forward(self, x):

        # Cast x to LoRA dtype
        x = x.to(self.lora_dtype)

        # Output with original weights (no gradients)
        orig_output = self.orig_layer(x)

        # Low-rank output (with gradients)
        lora_mult = (self.lora_A @ self.lora_B) * self.scaling  # Shape: (in_features, out_features)
        low_rank_output = x @ lora_mult  # Shape: (batch_size, out_features)

        # Sum outputs
        output = orig_output + low_rank_output

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
        self.num_embeddings, self.embedding_dim = self.orig_embed.weight.shape

        ### Create our Low Rank Matricies ###
        self.lora_A = nn.Parameter(torch.zeros(self.num_embeddings, rank), requires_grad=True)
        self.lora_B = nn.Parameter(torch.zeros(rank, self.embedding_dim), requires_grad=True)

        ### Different than the paper but matches implementation ###
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        ### Compute Scaling for LoRA ###
        self.scaling = self.lora_alpha / self.rank**0.5 if use_rslora else self.lora_alpha / self.rank
    
    def __repr__(self):
        return f"LoRAEmbedding({self.num_embeddings}, {self.embedding_dim}, rank={self.rank})"
    
    def _merge_weights(self):
        """
        xW^T + xAB = x(W^T + AB)
        """
        merged_weight = self.orig_embed.weight.data + (self.lora_A @ self.lora_B) * self.scaling

        merged_layer = nn.Embedding(self.num_embeddings, self.embedding_dim)

        merged_layer.weight.data = merged_weight

        return merged_layer
        

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
                 rank=8, 
                 lora_alpha=1, 
                 use_rslora=True, 
                 lora_dropout=0.0,
                 b_grad=True,
                 lora_dtype=torch.float32):
        
        super().__init__()

        assert isinstance(layer, nn.Conv2d), "LoRAConv2d only works with nn.Conv2d" 
        self.rank = rank 
        self.lora_alpha = lora_alpha
        self.kernel_size = layer.kernel_size
        self.stride = layer.stride
        self.padding = layer.padding
        self.lora_dtype = lora_dtype

        ### Store the Layer ###
        self.orig_conv = layer
        self.weight = layer.weight
        self.bias = layer.bias

        ### Change Gradient Flag ###
        self.orig_conv.weight.requires_grad = False
        self.orig_conv.bias.requires_grad = b_grad

        ### Convolution Weight Shape ###
        self.out_channels, self.in_channels, self.kernel_height, self.kernel_width = self.orig_conv.weight.shape

        ### Create our Low Rank Matricies (Flatten kernel weights and output rank) ###
        self.lora_A = nn.Parameter(torch.zeros(rank, self.in_channels, self.kernel_height, self.kernel_width, dtype=lora_dtype), requires_grad=True)
        self.lora_B = nn.Parameter(torch.zeros(rank, self.out_channels, dtype=lora_dtype), requires_grad=True)

        ### Initialize lora_A w/ gaussian, B stays as 0s (as described in paper) ###
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        ### Create LoRA Dropout ###
        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else lambda x: x

        ### Compute Scaling for LoRA ###
        self.scaling = self.lora_alpha / self.rank**0.5 if use_rslora else self.lora_alpha / self.rank

    def __repr__(self):
        return f"LoRAConv2D(in_channels={self.in_channels}, out_channels={self.out_channels}, rank={self.rank}, kernel_size={self.kernel_size})"
    
    def _merge_weights(self):
        """
        xW^T + xAB = x(W^T + AB)
        """
        ### (rank x in_chan x k_h, k_w) -> (rank x in_chan*k_h*k_w) ###    
        lora_A_flatten = self.lora_A.flatten(1)

        ### Matmul with lora_B Transposed (lora_B is rank x out_channels) -> (out_channels x rank) ###
        lora_mult = (self.lora_B.T @ lora_A_flatten) * self.scaling
        
        ### Place Back into Conv Weight Shape: (ou_chan x in_chan*k_h*k_w) -> (out_chan x in_chan x k_h x k_w) ###
        lora_mult = lora_mult.reshape(self.out_channels, self.in_channels, self.kernel_height, self.kernel_width)

        ### Merge ###
        merged_weight = self.orig_conv.weight.data + lora_mult

        merged_layer = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels, 
            kernel_size=(self.kernel_height, self.kernel_width),
            stride=self.stride, 
            padding=self.padding
        )

        merged_layer.weight.data = merged_weight

        if self.bias is not None:
            merged_layer.bias.data = self.bias

        return merged_layer

    def forward(self, x):
        
        ### Cast x to LoRA Dtype ###
        x = x.to(self.lora_dtype)

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
                 freeze_non_lora=True,
                 initializer_range=1.0,
                 b_grad=True,
                 lora_dtype=torch.float32):
        
        super().__init__()
        
        self.model = model
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.use_rslora = use_rslora
        self.lora_dropout = lora_dropout
        self.initializer_range = initializer_range
        self.b_grad = b_grad
        self.lora_dtype = lora_dtype
        self.freeze_non_lora = freeze_non_lora
        self.target_modules = target_modules
        self.exclude_modules = exclude_modules

        if self.target_modules is not None:
            assert isinstance(self.target_modules, list), "Make sure you pass in target modules as a list!"
        else:
            self.target_modules = []

        if self.exclude_modules is not None:
            assert isinstance(self.exclude_modules, list), "Make sure you pass in exclude modules as a list!"
        else:
            self.exclude_modules = []

        ### Compute Number of Trainable Parameters Before LoRA ###
        before_params = self._compute_trainable_parameters()

        ### Change Layers to LoRA ###
        self._apply_lora(self.model)

        ### Compute Number of Trainable Parameters After LoRA ###
        after_params = self._compute_trainable_parameters()

        print(f"Initial Parameters : {before_params} || LoRA Parameters : {after_params} || Trainable Proportion : {round(after_params*100/before_params, 2)}%")

    def _apply_lora(self, module):
    
        ### Recursively Go Through Model and Find Linear Layers ###
        for name, child in module.named_children():
            
            ### If name is in exclude_modules, dont do anything. Dont convert to LoRA and dont change requires_grad ###      
            if name in self.exclude_modules:  
                convert_to_lora = False
                exclude_from_changes = True

            else:

                exclude_from_changes = False
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
                                            b_grad=self.b_grad,
                                            lora_dtype=self.lora_dtype)
                    
                    ### Replace the linear layer (identified by its name) in this module with our lora layer ###
                    setattr(module, name, lora_layer)
                
                ### If this is a module we can change (exlude_from_change flag) and we want to freeze all non-lora params (freeze_non_lora flag) ##
                ### then we can do that here! ###
                elif not exclude_from_changes and self.freeze_non_lora:

                    child.weight.requires_grad = False
                    child.bias.requires_grad = False

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

                elif not exclude_from_changes and self.freeze_non_lora:

                    child.weight.requires_grad = False

            elif isinstance(child, nn.Conv2d):

                if convert_to_lora:

                    lora_layer = LoRAConv2d(layer=child,
                                            rank=self.rank, 
                                            lora_alpha=self.lora_alpha, 
                                            use_rslora=self.use_rslora, 
                                            lora_dropout=self.lora_dropout, 
                                            b_grad=self.b_grad,
                                            lora_dtype=self.lora_dtype)
                    
                    setattr(module, name, lora_layer)

                elif not exclude_from_changes and self.freeze_non_lora:

                    child.weight.requires_grad = False
                    child.bias.requires_grad = False


            ### Else, Dig Deeper Into the Module To Search For Linear Layers (as long as module wasnt selected to be exluded) ###
            else:
                
                dig_deeper = True
                if name in self.exclude_modules:
                    dig_deeper = False

                if dig_deeper:                    
                    self._apply_lora(child)

    def _merge_weights(self, module):

        for name, child in module.named_children():
            
            if isinstance(child, (LoRALinear, LoRAEmbedding, LoRAConv2d)):
                
                ### Compute Merged Layer ###
                merged_layer = child._merge_weights()
                
                ### Replace LoRA Layer with Merged Layer ###
                setattr(module, name, merged_layer)

            ### Continue Recursively Going through Model ###
            else:

                self._merge_weights(child)

    def save_model(self, path, save_trainable_only=True, merge_weights=False):
        
        if not merge_weights:
            state_dict = {name: param for name, param in self.named_parameters() \
                        if (param.requires_grad and save_trainable_only)}

        else:

            self._merge_weights(self.model)

            state_dict = {name: param for name, param in self.named_parameters()}

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
