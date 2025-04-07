"""
Llama4 Implementation. Full code credit goes to the 
Huggingface Team and their code for modeling_llama4.py
as well as the Meta team that published the model!
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class Llama4TextConfig:
    vocab_size: int = 202048
    hidden_size: int = 5120
    intermediate_size: int = 8192
    intermediate_size_mlp: int = 16384
    num_hidden_layers: int = 48
    num_attention_heads: int = 40
    num_key_value_heads: int = 8
    head_dim: int = 128
    max_position_embeddings: int = 4096 * 32
    rms_norm_eps: float = 1e-5
    pad_token_id: int = 200018
    bos_token_id: int = 1
    eos_token_id: int = 2
    rope_theta: float = 500000
    attention_dropout: float = 0.0
    num_experts_per_tok: int = 1
    num_local_experts: int = 16
    use_qk_norm: bool = True
    no_rope_layer_interval: int = 4
    attention_chunk_size: int = 8192
    attn_temperature_tuning: float = 4
    floor_scale: int = 8192
    attn_scale: float = 0.1

@dataclass
class Llama4VisionConfig:
    hidden_size: int = 768
    num_hidden_layers: int = 34
    num_attention_heads: int = 16
    num_channels: int = 3
    intermediate_size: int = 5632
    vision_output_dim: int = 7680
    image_size: int = 448
    patch_size: int = 14
    norm_eps: float = 1e-5
    pixel_shuffle_ratio: float = 0.5
    projector_input_dim: int = 4096
    projector_output_dim: int = 4096
    projector_dropout: float = 0.0
    attention_dropout: float = 0.0
    rope_theta: int = 10000

class Llama4TextExperts(nn.Module):

    def __init__(self, config):
        super(Llama4TextExperts, self).__init__()
        
        self.num_experts = config.num_local_experts
        self.intermediate_size = config.intermediate_size
        self.hidden_size = config.hidden_size
        self.expert_dim = config.intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, 2*self.expert_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.expert_dim, self.hidden_size))
        self.act_fn = nn.SiLU()

        nn.init.normal_(self.gate_up_proj)
        nn.init.normal_(self.down_proj)

    def forward(self, hidden_states):

        ### Hidden States we Pass in Here are already Sorted (according to the logic explained in Llama4TextMoe) ###
        
        ### Go Ahead and Split the Num Experts and Num Tokens Dimension Up (Num Experts x Num Tokens x Embed Dim) ###
        hidden_states = hidden_states.reshape(self.num_experts, -1, self.hidden_size)

        ### Now Multiply All our Tokens (copied for each Expert) by each Expert Embeddings ###
        ### gate_up_proj: (Num Experts x Hidden Size x 2*Expert Dim) ###
        ### gate_up -> (Num Experts x Num Tokens x 2*Expert Dim)
        gate_up = torch.bmm(hidden_states, self.gate_up_proj)
        
        ### The reason our linear layer was 2*Expert Dim output was because we do a gated linear layer 
        ### just like you find in previous Llama implementations
        ### First lets go ahead and chunk on the final dimension so we get two tensors:
        ### gate -> (Num Experts x Num Tokens x Expert Dim)
        ### up -> (Num Experts x Num Tokens x Expert Dim)
        gate, up = gate_up.chunk(2, dim=-1)

        ### Apply the Activation Function to Up (the SiLU activation) and multiply to gate
        ### Effectively giving finer control over what information continues and what doesnt
        ### gated -> (Num Experts x Num Tokens x Expert Dim)
        gated = up * self.act_fn(gate)

        ### Now Finally Projec the Expert Dim back down to the hidden size using Down Proj ###
        next_states = torch.bmm(gated, self.down_proj)

        ### Flatten from (Num Experts x Num Tokens x Embed Dim) -> (Num Experts*Num Tokens x Embed Dim) ###
        next_states = next_states.view(-1,self.hidden_size)

        return next_states

class Llama4TextMLP(nn.Module):

    def __init__(self, config):

        """
        Exactly the same as the Llama4TextExperts, but this is a single MLP layer. In
        Llama4 they pass tokens to both an MOE and a single MLP and add the results  
        """

        super(Llama4TextMLP, self).__init__()

        self.config = config
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.activation_fn = nn.SiLU()

    def forward(self, x):
        gated = self.activation_fn(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(gated)
    
class Llama4TextMoe(nn.Module):

    def __init__(self, config):
        super(Llama4TextMoe, self).__init__()

        self.top_k = config.num_experts_per_tok
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_local_experts
        self.experts = Llama4TextExperts(config)
        self.router = nn.Linear(config.hidden_size, config.num_local_experts, bias=False)
        self.shared_expert = Llama4TextMLP(config)

    def forward(self, hidden_states):

        batch_size, seq_len, embed_dim = hidden_states.shape

        ### Flatten Batch and Seq_Len Dimensions so we have (Num Tokens x Embed Dim) ###
        hidden_states = hidden_states.view(-1, self.hidden_dim)

        ### Get Expert Index for Each Token (Num Tokens x Num Experts) ###
        router_logits = self.router(hidden_states)

        ### Now we want each token to go to its own expert, but to parallize this easily we can do the following: 
        ### Step 1: Lets pretend each expert will get ALL TOKENS ###
        tokens_per_expert = batch_size * seq_len

        ### Step 2: Get the topK experts for each token (from our router logits) 
        ### this gives:
        ###     router_top_value: The logit from our router logits sorted for each token from largest to smallest (upto k of them)
        ###     router_indices: The indexes of which expert had the largest logit to smallest (upto k of them)
        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=1)

        ### Step 3: Create a Matrix of -infinity in the shape of (Num Tokens x Num Experts)
        ### Fill this matrix at the indexes of the topk experts selected by the router for each token with the logits for those k experts
        ### Basically makes sure we have -inf for the non topk tokens and its router logit otherwise!
        ### And go ahead and transpose this matrix so it finally becomes as (Num Experts x Num Tokens)
        router_scores = torch.full_like(router_logits, float("-inf")).scatter_(dim=1, index=router_indices, src=router_top_value).transpose(0,1)
       
        ### Step 4: Because we are passing in "ALL TOKENS" to every expert, lets update our router indicies to be 
        ### indexes from 0 to NUM TOKENS, repeated for EVERY Expert! It will something like:

        ### [0, 1, 2, ..., Num Tokens]
        ### [0, 1, 2, ..., Num Tokens]
        ### [0, 1, 2, ..., Num Tokens]
        ### ...

        ### Repeating the number of rows for the number of experts we have!
        router_indices = torch.arange(tokens_per_expert, device=hidden_states.device).unsqueeze(0).expand(router_scores.size(0), -1)
        
        ### Step 5: Now when we grab our embeddings with these indexes, we have the embedding dimension as well. Our Data is the shape
        ### of (Num Tokens x Embed Dim) Lets go ahead and flatten our router indicies, add a dimension for the embed dim and repeat. 
        ### This means we will end with a (Num_Experts*Num_tokens x Embed Dim) Matrix at the end. 
        router_indices = router_indices.reshape(-1,1).expand(-1, self.hidden_dim)

        ### Step 6: Update our Router Scores to be between 0 and 1. All our non topk scores (that are currently -inf) will become 0!
        router_scores = torch.sigmoid(router_scores.float()).to(hidden_states.dtype)
        
        ### Step 7: Gather All Our Hidden States By our router_indices (repeating all tokens for each Expert!)
        routed_in = torch.gather(
            input=hidden_states, 
            dim=0, 
            index=router_indices
        )

        ### Step 8: We now have repeated all our token embeddings for every expert. But we only want to keep the experts that
        ### were in our TopK. Well, we hav eour router_scores that is 0 for non Topk expert indicies. We can therefore go 
        ### ahead and multiply, as it will 0 out our embeddings assigned to a an expert that was not in its TopK
        ### This also multiplies the embeddings from their topk experts by the weights assigned by the router

        ### router_scores -> (Num Experts x Num Tokens)
        ### routed_in -> (Num Experts*Num Tokens x Embed Dim)
        routed_in = routed_in * router_scores.reshape(-1, 1)

        ### Lets Pass this to our Experts!! ###
        ### routed_out -> (Num_Experts*Num_Tokens x Embed Dim)
        routed_out = self.experts(routed_in)

        ### Similary, pass our Hidden States to the Shared Expert ###
        ### In Llama4 we have both Experts and a shared expert ###
        ### and we add the results together at the end! ###
        ### shared_expert_out -> (Num_Tokens x Embed Dim)
        shared_expert_out = self.shared_expert(hidden_states)
        
        ### Now we Add our Results Together! But we have a bit of a challenge ###
        ### routed_out repeats all tokens for all experts (but we zeroed out the experts that were not topk for each token)
        ### shared_expert_out is just our projection for all the tokens. 
        ### If we want to add our shared_expert_out to our routed_outs, we just need to grab the correct indicies from our 
        ### routed_outs that coorespond to the correct experts for each token, and then add it all together!

        ### REMEMBER: The output of Tokens (when passed to their Non-Topk expert) will just be 0!
        ### We are basically going through every experts output here and adding their outputs each expert
        ### to our shared_expert_out. But because there are lots of zeros in our embeddings from each 
        ### expert (if that was not a topk expert), we are basically accumulating across all the tokens

        ### This is the same as just for looping through all the expert outputs and adding it to our 
        ### shared_expert_out, but this is a oneliner that does that same thing!

        ### Take a look at scatter_add_ if you want more details! 
        ### https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_add_.html#torch.Tensor.scatter_add_
        shared_expert_out.scatter_add_(dim=0, index=router_indices, src=routed_out)

        return shared_expert_out

class Llama4TextRotaryEmbedding(nn.Module):
    
    def __init__(self, config, device=None):
        super(Llama4TextRotaryEmbedding, self).__init__()

        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        
        inv_freq = self._compute_default_rope_parameters(self.config, device=device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _compute_default_rope_parameters(self, config, device):

        base = config.rope_theta
        head_dim = config.head_dim

        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / head_dim))

        return inv_freq 
    
    @torch.no_grad()
    def forward(self, x, position_ids):

        ### x is our data (Batch x Seq Len x ) ###
        ### position_ids is the position index of every token (Batch x Seq Len) ###

        ### Our inv_freq is a vector of length (Hidden Dim/2). Lets add dimensions and repeat batch size number of times 
        ### inv_freq_expanded -> (Batch x Hidden_Dim/2 x 1)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)

        ### Our position ids is just Batch x Seq Len, add a new dimension to make it (Batch x 1 x Seq Len)
        position_ids_expanded = position_ids[:, None, :].float()

        ### Now We can compute our freqs for all positions (by multiplying by the position index) ###
        ### (B x H/2 X 1) @ (B x 1 x L) -> (B x H/2 x L)
        with torch.autocast(device_type=x.device.type, enabled=False):
            freqs = (inv_freq_expanded @ position_ids_expanded)

            ### Now make sure Freqs is (B x L x H/2) by transpose ###
            freqs = freqs.transpose(1,2)

            ### Convert Frequencies to complex (polar) representations (with magnitude 1)###
            freqs_cis = torch.polar(abs=torch.ones_like(freqs), angle=freqs)

        return freqs_cis

def apply_rotary_emb(xq, xk, freqs_cis):

    ### We want to "rotate" every consecutive pair along the embedding dimension ###
    ### by our freqs_cis, Lets first split them up! ###
    ### xq_split -> (B x Seq_len x Num Heads x Embed_dim/2 x 2)
    ### xk_split -> (B x Seq_len x Num Heads x Embed_dim/2 x 2)
    xq_split = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_split = xk.float().reshape(*xk.shape[:-1], -1, 2)

    ### Now rotating with a complex number is the same as mulitplying (phase shift) ###
    ### take a look here for rotation property: https://wumbo.net/concepts/complex-number-system/ ###   
    ### technically multiplying also scales, but we made sure our freqs_cis was magnitude 1 ###
    ### Problem: xq_split and xk_split are currently real numbers. Lets check what we wanted to do. 
    ### Lets go along the embedding dim for one token:
    ### [e0, e1, e2, e3, e4, e5, e6, e7, e8, ...]
    ### We have gone ahead and chunked it like the following:
    ### [e0, e1]
    ### [e2, e3]
    ### [e4, e5]
    ### [e6, e7]
    ### ....

    ### And now we want to rotate each of these 2dim vector by our complex freqs_cis. To do this correctly, 
    ### These 2dim vectors must also be complex so we can convert them to complex numbers like the following:
    ### [e0+j*e1]
    ### [e2+j*e3]
    ### [e4+j*e5]
    ### [e6+j*e7]
    ### ....

    ### xq_split -> (B x Seq_len x Num Heads x Embed_dim/2) # Complex number used the 2 dimension from earlier
    ### xk_split -> (B x Seq_len x Num Heads x Embed_dim/2) # Complex number used the 2 dimension from earlier
    xq_split = torch.view_as_complex(xq_split)
    xk_split = torch.view_as_complex(xk_split)

    ### Now Go Ahead and Multiply with our freqs (adding extra dimension for attention heads) !
    xq_out = xq_split * freqs_cis[:, :, None, :]
    xk_out = xk_split * freqs_cis[:, :, None, :]

    ### Now that we have done our Rotary embeddings with complex numbers, go ahead and return back to real numbers ###
    ### xq_out -> (B x Seq_len x Num Heads x Embed_dim/2 x 2) # Convert to real gave that 2 dimension back
    ### xk_out -> (B x Seq_len x Num Heads x Embed_dim/2 x 2) # Convert to real gave that 2 dimension back
    xq_out = torch.view_as_real(xq_out)
    xk_out = torch.view_as_real(xk_out)

    ### Flatten Embed_dim/2 x 2 dimensions back to Embed_dim
    xq_out = xq_out.flatten(3)
    xk_out =xk_out.flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)

class Llama4TextL2Norm(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def _norm(self, x):

        ### We want to do x / sqrt((x**2).mean()) along the last dimension! ###
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x)

    def extra_repr(self):
        return f"eps={self.eps}"
    
class Llama4TextRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):

        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"
    
class Cache:
    """
    KV Cache Method that is close to the Huggingface DynamicCache
    https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py
    """

    def __init__(self, config):

        ### Counter for Number of Tokens in Cache ###
        self._seen_tokens = 0

        ### Key/Value Cache (List of Tensor, where list is over model layers) ###
        self.key_cache = [torch.tensor([]) for _ in range(config.num_hidden_layers)]
        self.value_cache = [torch.tensor([]) for _ in range(config.num_hidden_layers)]

    def __repr__(self):        
        return f"DyanmicCache(Num_Layers: {len(self.key_cache)} | Cached Tokens: {self.key_cache[0].shape[2]})"
        
    def update(self, key_states, value_states, layer_idx):
        
        ### Only iterate num tokens seen on the first layer ###
        ### key_states (B x H x L x E)
        ### value_states (B x H x L x E)

        if layer_idx == 0:  
            self._seen_tokens += key_states.shape[-2]

        ### Append New key/Value states to key/value cache ###
        self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
        self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]
    
    def get_seq_len(self, layer_idx=0):
        return self.key_cache[layer_idx].shape[-2] if self.key_cache[layer_idx].numel() != 0 else 0

class Llama4TextAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super(Llama4TextAttention, self).__init__()

        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.scaling = self.head_dim **-0.5
        self.attn_scale = config.attn_scale
        self.floor_scale = config.floor_scale
        self.attn_temperature_tuning = config.attn_temperature_tuning
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.use_rope = int((layer_idx+1) % 4 == 0)

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads*self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads*self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads*self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads*self.head_dim, config.hidden_size, bias=False)

        if self.config.use_qk_norm and self.use_rope:
            self.qk_norm = Llama4TextL2Norm()
        else:
            self.qk_norm = None

    def _repeat_kv(self, hidden_states, n_rep):

        ### Add Extra Dimension to Repeat Over ###
        ### (B x H x L x E) -> (B x H x 1 x L x E) -> (B x H x n_rep x L x E) -> (B x H*n_rep x L x E)

        batch, heads, seq_len, embed_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, heads, n_rep, seq_len, embed_dim)
        hidden_states = hidden_states.reshape(batch, heads*n_rep, seq_len, embed_dim)

        return hidden_states

    def forward(self, 
                hidden_states, 
                position_embeddings=None, 
                attention_mask=None, 
                past_key_value=None, # this is a Cache Object
                cache_position=None):
        
        ### Get Shape of Tensor (without Embed dims) ###
        input_shape = hidden_states.shape[:-1]

        ### Create a Shape Tuple (-1 for num heads in the future) ###
        hidden_shape = (*input_shape, -1, self.head_dim)

        ### Split Embed Dim by Heads ###
        query_states = self.q_proj(hidden_states).reshape(hidden_shape)
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1,2)
        
        ### Apply RoPE to Query and Key ###
        if self.use_rope:
            query_states, key_states = apply_rotary_emb(query_states, key_states, position_embeddings.to(query_states.device))

        ### L2 Norm ###
        if self.qk_norm is not None:
            query_states = self.qk_norm(query_states)
            key_states = self.qk_norm(key_states)

        ### Temperature Tuning (on layers not using rope) ###
        ### Not total sure how they came up with this as the scaling factor in the ###
        ### Paper they referenced https://arxiv.org/abs/2501.19399
        ### was just S log(N) where S was some learnable parameter

        ### Here was their reasoning:
        ### "We are applying temperature tuning (https://arxiv.org/abs/2501.19399) to NoPE layers, where
        ### the inference-time temperature tuning function is customized to not affect short context
        ### while working at very long context"

        cache_position = cache_position.unsqueeze(0).expand(hidden_states.shape[0],-1)
        if self.attn_temperature_tuning and not self.use_rope:
            attn_scales = (
                torch.log(torch.floor((cache_position.float() + 1.0) / self.floor_scale) + 1.0) * self.attn_scale + 1.0
            )
        
            attn_scales = attn_scales.view((*input_shape, 1, 1))
            query_states = (query_states * attn_scales).to(query_states.dtype)

        ### Flip the Head and Seq Len Dimensions ###
        ### (B x L x H x D) -> (B x H x L x D)
        query_states = query_states.transpose(1,2)
        key_states = key_states.transpose(1,2)

        ### Update Key Value Cache ###
        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)

        ### Grouped Query Attention ###

        ### Step 1: Repeat Keys/Values Heads to match the Query Heads
        ### Query: (B x Attention Heads x L x E)
        ### Key: (B x KV Heads x L x E)
        ### Value: (B x KV Heads x L x E)
        key_states = self._repeat_kv(key_states, self.num_key_value_groups)
        value_states = self._repeat_kv(value_states, self.num_key_value_groups)

        ### Step 2: Compute Attention Weights ###
        attn_weights = torch.matmul(query_states, key_states.transpose(2,3)) * self.scaling

        ### Step 3: Apply Attention Mask (add large negative numbers to masked positions) ###
        ### Also we crop out attention mask only upto the tokens we need ###
        if attention_mask is not None:

            ### Crop Attetion Mask to only include upto the number of key/value tokens we have to attend to ###
            causal_mask = attention_mask[:, :, :, :key_states.shape[-2]]

            ### Add Causal Mask to Weights ###
            attn_weights = attn_weights + causal_mask

        ### Standard Attention ###
        attn_weights = nn.functional.softmax(attn_weights.float(), dim=-1).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1,2)

        ### Final Projection ###
        attn_output = attn_output.flatten(2)
        attn_output = self.o_proj(attn_output)

        return attn_output

class Llama4TextDecoderLayer(nn.Module):

    def __init__(self, config, layer_idx):
        super(Llama4TextDecoderLayer, self).__init__()

        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.self_attn = Llama4TextAttention(config, layer_idx)
        self.use_chunked_attention = int((layer_idx+1) % 4 != 0)
        self.feed_forward = Llama4TextMoe(config)
        self.input_layernorm = Llama4TextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Llama4TextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, 
                hidden_states, 
                attention_mask, 
                chunk_causal_mask,  
                past_key_value, 
                cache_position, 
                position_embeddings):
        
        residual = hidden_states

        ### Normalize ###
        hidden_states = self.input_layernorm(hidden_states)

        ### Did you pass in chunked attention mask? Then use it if enabled! ###
        if self.use_chunked_attention and chunk_causal_mask is not None:
            attention_mask = chunk_causal_mask

        ### Compute Attention ###
        attention_states = self.self_attn(
                hidden_states=hidden_states, 
                position_embeddings=position_embeddings, 
                attention_mask=attention_mask, 
                past_key_value=past_key_value, # this is a Cache Object
                cache_position=cache_position
        )
        
        ### Residual Connection ###
        hidden_states = residual + attention_states

        ### MOE Layer ###
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states.view(residual.shape)

        return hidden_states

class Llama4TextModel(nn.Module):

    def __init__(self, config):
        super(Llama4TextModel, self).__init__()

        self.config = config

        self.padding_idx = config.pad_token_id
        self.vocab_sie = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)
        self.layers = nn.ModuleList(
            [
                Llama4TextDecoderLayer(config, layer_idx) 
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        self.norm = Llama4TextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Llama4TextRotaryEmbedding(config)

    def forward(self,
                input_ids=None, 
                input_embeds=None,
                attention_mask=None, 
                position_ids=None, 
                past_key_values=None, 
                cache_position=None):
        
        ### Set up Key/Value Cache if it doesnt exist ###
        if past_key_values is None:
            past_key_values = Cache(self.config)

        ### Get Input Embeddings ###
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds

        ### Set up Cache Position ###
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_len()
            cache_position = torch.arange(past_seen_tokens, past_seen_tokens+hidden_states.shape[1], device=hidden_states.device)

        ### Setup Position Ids ###
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        ### Update Causal Mask ###
        causal_mask, chunked_attention_mask = self._update_causal_mask(
            attention_mask, hidden_states, cache_position
        )

        ### Compute Rotary Embeddings ###
        freq_cis = self.rotary_emb(hidden_states, position_ids)

        ### Pass Through Layers ###
        
        for layer in self.layers:
            hidden_states = layer(hidden_states, 
                                  attention_mask, 
                                  chunked_attention_mask if chunked_attention_mask is not None else causal_mask, 
                                  past_key_values, 
                                  cache_position, 
                                  freq_cis)
        
        hidden_states = self.norm(hidden_states)

        return hidden_states, past_key_values
    
    def _prepare_4d_causal_attention_mask_with_cache_position(
            self,
            attention_mask, 
            sequence_length, 
            cache_position, 
            batch_size,
            dtype,
            device
    ):
        
        if attention_mask is not None:
            causal_mask = attention_mask

        else:

            ### Create a Causal Mask ###
            min_dtype = torch.finfo(dtype).min

            causal_mask = torch.full(
                (sequence_length, sequence_length), fill_value=min_dtype, dtype=dtype, device=device
            )

            ### If we have more than one token, we need causal mask, so current token doesnt look forward ###
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            
            ### Check Wherever our Sequence index is longer than the cache positions ### 
            ### If a Query is BEFORE the Key/Value Cache Index, then the Query would have ###
            ### to look into the future which is not allowed and needs to be masked! ###
            causal_mask *= torch.arange(sequence_length, device=device) > cache_position.to(device).reshape(-1,1)

            ### Add Extra Dimension to causal Mask (4 dimensions, with placeholder for head dim) ###
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)

            ### If Attention mask is Not None, then add causal and attention mask together ###
            if attention_mask is not None:
                causal_mask = causal_mask.clone()
                
                ### Only keep upto length in attention mask ###
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(device)
                padding_mask = (padding_mask==0)
                causal_mask[:,:,:,:mask_length] = causal_mask[:,:,:,:mask_length].masked_fill(padding_mask, min_dtype)

            return causal_mask

    def _create_chunked_attention_mask(
            self, 
            attention_chunk_size, 
            start, 
            end, 
            device
    ):
        
        ### Create Blocks in the Chunk Sizes you want ###
        # tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2],
        #         [0, 0, 0, 1, 1, 1, 2, 2, 2],
        #         [0, 0, 0, 1, 1, 1, 2, 2, 2],
        #         [1, 1, 1, 0, 0, 0, 1, 1, 1],
        #         [1, 1, 1, 0, 0, 0, 1, 1, 1],
        #         [1, 1, 1, 0, 0, 0, 1, 1, 1],
        #         [2, 2, 2, 1, 1, 1, 0, 0, 0],
        #         [2, 2, 2, 1, 1, 1, 0, 0, 0],
        #         [2, 2, 2, 1, 1, 1, 0, 0, 0]])

        block_pos = torch.abs(
            (torch.arange(start, end).unsqueeze(0)//attention_chunk_size) -
            (torch.arange(start, end).unsqueeze(1)//attention_chunk_size)
        )
        
        ### Do computation again but without absolute value or division to get token positions###
        # tensor([[ 0,  1,  2,  3,  4,  5,  6,  7,  8],
        #         [-1,  0,  1,  2,  3,  4,  5,  6,  7],
        #         [-2, -1,  0,  1,  2,  3,  4,  5,  6],
        #         [-3, -2, -1,  0,  1,  2,  3,  4,  5],
        #         [-4, -3, -2, -1,  0,  1,  2,  3,  4],
        #         [-5, -4, -3, -2, -1,  0,  1,  2,  3],
        #         [-6, -5, -4, -3, -2, -1,  0,  1,  2],
        #         [-7, -6, -5, -4, -3, -2, -1,  0,  1],
        #         [-8, -7, -6, -5, -4, -3, -2, -1,  0]])

        tok_pos = torch.arange(start, end).unsqueeze(0) - torch.arange(start,end).unsqueeze(1)
        
        ### Wherever our block pos is 0 (our chunks) and our tok_pos is negative (causal mask) We want those ###
        mask = (block_pos==0) & (tok_pos<=0)

        return mask.to(device)
        
    def _update_causal_mask(self, 
                            attention_mask, 
                            input_tensor, 
                            cache_position):

        ### Get Sequence Length and Attention Chunk Size ###        
        ### Honestly the attention chunk size is huge (over 8000) 
        ### we could probably skip this as we are just creating a 
        ### minimal version here but lets just do it anyway! 
        seq_len = input_tensor.shape[1]
        attention_chunk_size = self.config.attention_chunk_size

        ### Get the Starting and Ending Position from Cache ###
        start_cache_pos = cache_position[0]
        
        ### Two Checks: Is our First Cache Position greater than our chunk size:
        cond1 = start_cache_pos >= attention_chunk_size

        ### Is our first cache in the first chunk, but with the seq_len it rolls over to the second chunk ###
        cond2 = (start_cache_pos < attention_chunk_size) & (start_cache_pos + seq_len > attention_chunk_size)

        ### This is just a fancy if/else 
        ### If cond1 is True then key_length = attention_chunk_size + seq_len - 1
        ### elif cond1 is false, then if cond2 is True then key_length = start_cache_pos + seq_len
        ### else key_length = attention_chunk_size
        key_length = torch.where(
            cond1,
            attention_chunk_size + seq_len - 1,
            torch.where(cond2, start_cache_pos + seq_len, attention_chunk_size),
        )

        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask, 
            sequence_length=seq_len, 
            cache_position=cache_position, 
            batch_size=input_tensor.shape[0], 
            dtype=input_tensor.dtype, 
            device=input_tensor.device
        )

        chunked_attention_mask = None
        if seq_len > self.config.attention_chunk_size:
            chunked_attention_mask = self._create_chunked_attention_mask(
                self.config.attention_chunk_size, 
                start=start_cache_pos,
                end=start_cache_pos+key_length, 
                device=input_tensor.device 
            )

            ### Mask only valid wherever attention mask is valid ###
            chunked_attention_mask = chunked_attention_mask & attention_mask

            ### Add dimensions and fill with -inf ###
            chunked_attention_mask = (
                    chunked_attention_mask[None, None, :, :]
                    .to(input_tensor.dtype)
                    .masked_fill(chunked_attention_mask, torch.finfo(input_tensor.dtype).min)
                )
            
        return causal_mask, chunked_attention_mask

class Llama4ForCausalLM(nn.Module):

    def __init__(self, config):
        super(Llama4ForCausalLM, self).__init__()

        self.model = Llama4TextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, 
                input_ids=None, 
                input_embeds=None,
                attention_mask=None, 
                position_ids=None, 
                past_key_values=None):
        
        outputs, past_key_values = self.model(
            input_ids, 
            input_embeds,
            attention_mask=attention_mask, 
            position_ids=position_ids, 
            past_key_values=past_key_values
        )

        logits = self.lm_head(outputs)
 
        return logits, past_key_values

class Llama4VisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = nn.GELU()  
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states
    
class Llama4VisionMLP2(nn.Module):

    """
    Pretty standard MLP Layer found in most things like the Vision Transformer

    I feel like there is a bug in Huggingface regarding the implementation of this, 
    so i am just manually passing in the in_features/out_features
    https://github.com/huggingface/transformers/issues/37321
    """
    def __init__(self, in_features, out_features, config):
        super(Llama4VisionMLP2, self).__init__()

        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.fc1 = nn.Linear(in_features, out_features, bias=False)
        self.fc2 = nn.Linear(out_features, out_features, bias=False)
        self.activation_fn = nn.GELU()
        self.dropout = nn.Dropout(config.projector_dropout)

    def forward(self, hidden_states):

        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.activation_fn(self.fc2(hidden_states))
        return hidden_states
    
class Llama4MultiModalProjector(nn.Module):
    def __init__(self, vision_config, text_config):
        super(Llama4MultiModalProjector, self).__init__()
        self.linear_1 = nn.Linear(
            vision_config.projector_output_dim, 
            text_config.hidden_size, 
            bias=False
        )

    def forward(self, x):
        return self.linear_1(x)
    
class Llama4VisionPixelShuffleMLP(nn.Module):

    def __init__(self, config):
        super(Llama4VisionPixelShuffleMLP, self).__init__()
        self.pixel_shuffle_ratio = config.pixel_shuffle_ratio
        self.inner_dim = int(config.hidden_size//(self.pixel_shuffle_ratio**2))
        self.output_dim = config.projector_output_dim
        self.mlp = Llama4VisionMLP2(self.inner_dim, self.output_dim, config)

    def _pixel_shuffle(self, x):

        ### Data is already in ViT Shape (B x Num Patches x Embed Dim) ###
        ### We want to convert back to (B x sqrt(Num Patches) x sqrt(Num Patches) x Embed Dim) )
        batch_size, num_patches, embed_dim = x.shape
        patch_size = int(num_patches**0.5)

        ### Reshape Tensor ###
        x = x.reshape(batch_size, patch_size, patch_size, -1)

        ### Pixel Shuffle Moves Information (pixels) from the channel dimension to the spatial dimension ###
        ### Or it moves pixels from spatial to channels depending on ratio ###
        ### Lets Do that on the last dimension first ###
        batch_size, height, width, channels = x.shape
        x = x.reshape(batch_size, height, int(width * self.pixel_shuffle_ratio), int(channels/self.pixel_shuffle_ratio))

        ### Lets reshape this to expose the height to our channels and pixel shuffle again ###
        x = x.transpose(1,2)
        x = x.reshape(batch_size, int(width*self.pixel_shuffle_ratio), int(height*self.pixel_shuffle_ratio), int(channels/(self.pixel_shuffle_ratio**2)))
        
        ### Finall reshape back to image shape (B x H x W x C) and put back to sequence shape ###
        x = x.permute(0,2,1,3)
        
        x = x.reshape(batch_size, -1, x.shape[-1])
        
        return x
    
    def forward(self, x):

        ### Apply Pixel Shuffle ###
        shuffled = self._pixel_shuffle(x)

        ### Map Channels (embed_dim) to our target out_features ###
        proj = self.mlp(shuffled)

        return proj

class Llama4VisionRotaryEmbedding(nn.Module):
    def __init__(self, config):
        super(Llama4VisionRotaryEmbedding, self).__init__()

        ### Get Number of Patches in Image and create Arange array ###
        index = config.image_size // config.patch_size
        image_index = torch.arange(index**2, dtype=torch.int32).reshape(index**2, 1)

        ### Add on -2 Index for CLS token at the end ###
        image_index = torch.cat([image_index, torch.tensor([-2]).reshape(-1,1)], dim=0)
        
        ### Remember that our X,Y coodinates get flattened to a vector in the
        ### end so lets just compute those now. The X coordinate repeats every column
        ### and the Y coordinate repeats every row so we can access them like this:
        x_coord = image_index % index
        y_coord = image_index // index

        ### Get the frequency dim (per head) ###
        freq_dim = config.hidden_size // config.num_attention_heads // 2 

        ### Compute Rope Freq just like before ###
        rope_freq = 1.0 / (config.rope_theta ** (torch.arange(0, freq_dim, 2)[:(freq_dim // 2)].float() / freq_dim))
        
        ### Compute Frequencies for X and Y coordinate and repeat along embed dim ###
        ### freqs_x -> NumPatches+1, 1, freq_dim
        ### freqs_y -> NumPatches+1, 1, freq_dim

        ### Also the +1 makes sure we arent multiplying by a 0! We will keep that 0 freq for the cls token ###
        freqs_x = ((x_coord + 1)[..., None] * rope_freq[None, None, :]).repeat_interleave(2, dim=-1)
        freqs_y = ((y_coord + 1)[..., None] * rope_freq[None, None, :]).repeat_interleave(2, dim=-1)
        
        ### Concatenate X and Y Frequencies Together ###
        ### then grab every other index so in the end we have 
        ### half the frequencies on the x index and half from the y index
        ### and together they provide all the positional information of that (x,y) coordinate
        freqs = torch.cat([freqs_x, freqs_y], dim=-1)[..., ::2]

        ### So even through we added our CLS Token index to the end, 
        ### We want its position information at the beginning? 0 Frequency
        ### Why not! Im sure it wouldnt matter either way but lets keep it like
        ### the huggingface implementation 
        freqs = freqs.masked_fill(image_index.reshape(-1,1,1) < 0, 0)

        ### Convert Frequencies to Sin/Cos, stack and convert to complex ###
        self.freq_cis = torch.view_as_complex(torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1))
    
    def forward(self, x):
        return self.freq_cis.to(x.device)

def vision_apply_rotary_emb(
    query, key, freqs_ci):

    query_ = torch.view_as_complex(query.float().reshape(*query.shape[:-1], -1, 2))
    key_ = torch.view_as_complex(key.float().reshape(*key.shape[:-1], -1, 2))

    ### freq_ci -> (Num Patches x 1 x Head Dim), we need to add the batch dimension
    ### because Query is (batc x num_patches x num heads x head dim)
    freqs_ci = freqs_ci.unsqueeze(0) 

    ### Multiply, convert back to real, flatten the real/complex component dimension ###
    query_out = torch.view_as_real(query_ * freqs_ci).flatten(3)
    key_out = torch.view_as_real(key_ * freqs_ci).flatten(3)

    return query_out.type_as(query), key_out.type_as(key)  # but this drops to 8e-3

class Llama4VisionAttention(nn.Module):

    """
    Standard ViT Attention Mechanism!
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_key_value_groups = 1
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.embed_dim, bias=True)

    def forward(
        self,
        hidden_states,
        freqs_ci,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        ## Project QKV and reshape to (B)
        query_states = self.q_proj(hidden_states).view(hidden_shape)
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape)

        ### Apply Rotary Embeddings
        query_states, key_states = vision_apply_rotary_emb(query_states, key_states, freqs_ci=freqs_ci)

        ### Transpose to (B x H x L x E)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        ### Compute Attention ###
        attn_output = F.scaled_dot_product_attention(query_states, key_states, value_states)

        ### Return Output back to (B x Num Patches x Embed Dim) ###
        attn_output = attn_output.reshape(*input_shape, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output

class Llama4VisionEncoderLayer(nn.Module):
    def __init__(self, config):
        super(Llama4VisionEncoderLayer, self).__init__()

        self.hidden_size = config.hidden_size
        self.self_attn = Llama4VisionAttention(config)
        self.mlp = Llama4VisionMLP(config)

        self.input_layernorm = nn.LayerNorm(config.hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size)
    
    def forward(self, hidden_states, freqs_ci):

        reisudal = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        ### Compute Attention ###
        hidden_states = self.self_attn(
            hidden_states, 
            freqs_ci, 
        )

        ### Residual Connection ###
        hidden_states = hidden_states + reisudal

        ### Feed Forward ###
        residual = hidden_states    
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
    
class Llama4VisionEncoder(nn.Module):

    def __init__(self, config):
        super(Llama4VisionEncoder, self).__init__()

        self.config = config
        self.layers = nn.ModuleList(
            [
                Llama4VisionEncoderLayer(config) for _ in range(config.num_hidden_layers)
            ]
        )
    
    def forward(self,
                hidden_states, 
                freqs_ci):
        
        for layer in self.layers:
            hidden_states = layer(
                hidden_states, freqs_ci
            )

        return hidden_states

class Llama4UnfoldConvolution(nn.Module):

    def __init__(self, config):
        super(Llama4UnfoldConvolution, self).__init__()
        
        ### Use nn.Unfold instead of a Conv to do Patch Embedding ###
        ### basically the same thing (as convs use nn.Unfold) ###
        ### because convolutions ARE LINEAR LAYERS ###

        self.unfold = nn.Unfold(kernel_size=config.patch_size, stride=config.patch_size)

        ### Linear Projection to Wanted embed Dim ###
        self.linear = nn.Linear(config.num_channels * config.patch_size * config.patch_size, 
                                config.hidden_size, 
                                bias=False)
        
    def forward(self, x):

        ### Convert to Patches (B x C x Num Patches)###
        x = self.unfold(x)
        
        ### Project to Embed Dim (reshape first to have C last) ###
        x = self.linear(x.permute(0,2,1))

        return x

class Llama4VisionModel(nn.Module):

    def __init__(self, config):
        super(Llama4VisionModel, self).__init__()

        ### Scale Parameter Constant (probably for scale before training) ###
        ### We arent training so this doesnt really matter but lets keep it ###
        ### the same! ###
        self.scale = config.hidden_size**-0.5

        ### Just Like Vision Transformer, Number of Patches + CLS Token ###
        self.num_patches = (config.image_size // config.patch_size) ** 2 + 1

        ### Patch Emebedding ###
        self.patch_embedding = Llama4UnfoldConvolution(config)

        ### CLS Token ###
        self.class_embedding = nn.Parameter(self.scale * torch.randn(config.hidden_size))

        ### Positional Info (Additional to Rotary) ###
        self.positional_embedding_vlm = nn.Parameter(self.scale * torch.randn(self.num_patches, config.hidden_size))

        ### Rotary Embeddings ###
        self.rotary_embed = Llama4VisionRotaryEmbedding(config)

        ### Layer Norm ###
        self.layernorm_pre = nn.LayerNorm(config.hidden_size)
        self.layernorm_post = nn.LayerNorm(config.hidden_size)

        ### Encoders ###
        self.model = Llama4VisionEncoder(config)
        self.vision_adapter = Llama4VisionPixelShuffleMLP(config)

    def forward(self, 
                pixel_values):
        
        batch_size, num_channels, height, width = pixel_values.shape

        ### Patchify Image to Embeddings ###
        patch_embed = self.patch_embedding(pixel_values)
        batch_size, num_patches, embed_dim = patch_embed.shape

        ### Concat on CLS token (repeat for batches though) ##
        ### also as we implemented in rotary, we append to the end ###
        cls_token = self.class_embedding.expand(batch_size, 1, embed_dim)
        hidden_state = torch.cat([patch_embed, cls_token], dim=1)
        num_patches += 1

        ### Add on Positional Embeddings ###
        hidden_state = hidden_state + self.positional_embedding_vlm
        
        ### Layernorm before Encoder ###
        hidden_state = self.layernorm_pre(hidden_state)

        ### Compute frequencies for rotary ###
        freqs_ci = self.rotary_embed(hidden_state)

        ### Pass Through Model ###
        output = self.model(
            hidden_state, 
            freqs_ci
        )

        ### Layernorm after output ###
        output = self.layernorm_post(output)

        ### Cut of CLS token ###
        output = output[:, :-1]

        ### Project to Embeddings ###
        adapted = self.vision_adapter(output)
        
        return adapted
        
class Llama4ForConditionalGeneration(nn.Module):
    def __init__(self, 
                 vision_config, 
                 text_config,
                 boi_token_index=200080,
                 eoi_token_index=200081, 
                 image_token_index=200092):
        super(Llama4ForConditionalGeneration, self).__init__()

        ### Load Vision Model ###
        self.vision_model = Llama4VisionModel(vision_config)
        
        ### Projects from Vision hidden dim to Text hidden Dim ###
        self.multi_modal_projector = Llama4MultiModalProjector(vision_config, text_config)

        ### Load Text Model ###
        self.language_model = Llama4ForCausalLM(text_config)

        self.vocab_size = text_config.vocab_size
        self.pad_token_id = text_config.pad_token_id
        self.boi_token_index = boi_token_index
        self.eoi_token_index = eoi_token_index
        self.image_token_index = image_token_index

    def forward(self,
                input_ids, 
                pixel_values=None,
                attention_mask=None, 
                position_ids=None, 
                past_key_values=None):

        ### Convert Tokens to Embeddings (already has placeholder for image tokens if we are passing in image) ###
        token_embeddings = self.language_model.model.embed_tokens(input_ids)
        
        if pixel_values is not None:

            ### Conver Image to Pixel Values (B x Num Patches x Embed Dim) ###
            image_features = self.vision_model(pixel_values)

            ### Store Current Shape of Text ###
            text_embed_shape = token_embeddings.shape

            ### Flatten to (B*Num Patches x Embed Dim)
            image_flat = image_features.reshape(-1, image_features.shape[-1])

            ### Project Image Features to token embeddings Embed dim ###
            proj_image_flat = self.multi_modal_projector(image_flat)

            ### Find where our image_token_index is ###
            ### The processor in huggingface will automatically insert 
            ### tokens like <IMAGE_TOKEN> into the text. We just need to 
            ### Copy our image tokens into those positions!
            special_image_mask = (input_ids == self.image_token_index).unsqueeze(-1)
            final_mask = special_image_mask.to(token_embeddings.device)

            ### Flatten Input Embeddings to become (Batch*SeqLen x Embed Dim)
            token_embeddings = token_embeddings.view(-1, token_embeddings.size(-1))

            ### Flatten Final Mask to Also be Batch * Seq Len ###
            final_mask_1d = final_mask.squeeze(-1).reshape(-1)

            ### Compute Num Tokens to fill ###
            num_tokens_to_fill = final_mask.sum()

            ### Make sure number of tokens we are filling is the same as the number of image embedding tokens we have ###
            assert num_tokens_to_fill == proj_image_flat.shape[0]

            ### Expand mask to include embed dim so we can use masked_scatter_
            expanded_mask = final_mask_1d.unsqueeze(-1).expand(-1, token_embeddings.shape[-1])

            ### Copy in our Image Embeddings into our Placeholder Embeddings in our Text !!! ###
            token_embeddings.masked_scatter_(expanded_mask, proj_image_flat)

            ### Restore Orignal Shape of (B x Seq Len x Embed Dim)
            token_embeddings = token_embeddings.reshape(text_embed_shape)

        outputs, past_key_values = self.language_model(
            input_embeds=token_embeddings,
            attention_mask=attention_mask, 
            position_ids=position_ids, 
            past_key_values=past_key_values, 
        )

        return outputs, past_key_values
    
if __name__ == "__main__":


    text_config = Llama4TextConfig(hidden_size=768, 
                                   intermediate_size=768*4, 
                                   intermediate_size_mlp=768*4, 
                                   num_hidden_layers=2)
                                   
    vision_config = Llama4VisionConfig(image_size=448,
                                       patch_size=14, 
                                       num_hidden_layers=2)
    
    model = Llama4ForConditionalGeneration(vision_config, text_config)

    input_ids = torch.randint(0,200000, size=(4,2048))   
    prepend_image_tokens = torch.tensor([200080] + [200092 for _ in range(256)] + [200081], dtype=torch.long).unsqueeze(0).expand(4,-1)
    input_ids = torch.cat([prepend_image_tokens, input_ids], dim=-1)

    pixel_values = torch.randn(4,3,448,448)

    ### Pass Initial input and Get Cached Key/Value ##
    outputs, cache = model(input_ids, pixel_values)

    ### Get your Next token Prediction ###
    print("First Token Prediction")
    pred_next_token = outputs[:, -1].argmax(axis=-1, keepdims=True)
    print(pred_next_token)

    ### Pass in New Tokens (and Cache) to predict again the next token ###
    print("Second Token Prediction")
    outputs, cache = model(pred_next_token, past_key_values=cache)
    pred_next_token = outputs[:, -1].argmax(axis=-1, keepdims=True)
    print(pred_next_token)
    