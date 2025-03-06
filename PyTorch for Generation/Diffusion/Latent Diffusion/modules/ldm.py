import torch
import torch.nn as nn
from transformers import CLIPTextModel

from .vae import VAE, VQVAE
from .unet import UNet2DModel
from .embeddings import PositionalEncoding, ClassConditionalEmbeddings

class LDM(nn.Module):
    
    def __init__(self, config):
    
        super(LDM, self).__init__()

        self.config = config

        ### Load VAE Model ###
        if self.config.quantize:
            self.vae = VQVAE(config=config)
        else:
            self.vae = VAE(config=config)   

        ### VAE is PreTrained, Disable Gradients ###
        for param in self.vae.parameters():
            param.requires_grad = False

        ### Load Diffusion Time Embeddings ###
        self.sinusoidal_time_embeddings = PositionalEncoding(
            max_len=config.num_diffusion_timesteps, 
            time_embed_start_dim=config.time_embed_start_dim,
            time_embed_proj_dim=config.time_embed_proj_dim
        )

        ### Load Text Conditioning Model (if we have text and its not preencoded) ###
        if config.text_conditioning and not config.pre_encoded_text:
            self.text_conditioning = CLIPTextModel.from_pretrained(
                   config.text_conditioning_hf_model, 
            )

            self.text_conditioning.eval()

            for param in self.text_conditioning.parameters():
                param.requires_grad = False

        ### Load Class Conditioning Module ###
        if config.class_conditioning:
            self.class_conditioning = ClassConditionalEmbeddings(
                num_classes=config.num_classes, 
                embed_dim=config.class_embed_dim
            )

        ### Load UNET Model ###
        self.unet = UNet2DModel(config)

    def _load_vae_state_dict(self, state_dict):
        
        self.vae.load_state_dict(state_dict)

    def forward(self, 
                noisy_input, 
                timesteps, 
                text_conditioning=None, 
                text_attention_mask=None,
                class_conditioning=None,
                cfg_weight=0):
        
        

if __name__ == "__main__":

    from .config import LDMConfig

    config = LDMConfig()

    model = LDM(config=config)
    print(model)