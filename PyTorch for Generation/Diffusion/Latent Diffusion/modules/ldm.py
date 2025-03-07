import torch
import torch.nn as nn
from transformers import CLIPTextModel

from .vae import VAE, VQVAE
from .unet import UNet2DModel
from .embeddings import PositionalEncoding, ClassConditionalEmbeddings
from .scheduler import Sampler

loss_functions = {"mse": nn.MSELoss(), 
                  "mae": nn.L1Loss(), 
                  "huber": nn.HuberLoss()}

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
            self.text_encoder = CLIPTextModel.from_pretrained(
                   config.text_conditioning_hf_model, 
            )

            self.text_encoder.eval()

            for param in self.text_encoder.parameters():
                param.requires_grad = False

        ### Load Class Conditioning Module ###
        if config.class_conditioning:
            self.class_encoder = ClassConditionalEmbeddings(
                num_classes=config.num_classes, 
                embed_dim=config.class_embed_dim
            )

        ### Load UNET Model ###
        self.unet = UNet2DModel(config)

        ### Get DDPM Sampler ###
        self.ddpm_sampler = Sampler(total_timesteps=config.num_diffusion_timesteps,
                                    beta_start=config.beta_start, 
                                    beta_end=config.beta_end)

    def _load_vae_state_dict(self, state_dict):
        self.vae.load_state_dict(state_dict)

    @torch.no_grad()
    def _vae_encode_images(self, x, scale_factor=None):
        return self.vae.encode(x, return_stats=False, scale_factor=scale_factor)["posterior"]
    
    @torch.no_grad()
    def _vae_decode_images(self, x, scale_factor=None):
        return self.vae.decode(x, scale_factor=scale_factor)
    
    @torch.no_grad()
    def _encode_text(self, input_ids, attention_mask=None):
        return self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    def forward(self, 
                images, 
                text_conditioning=None, 
                text_attention_mask=None,
                class_conditioning=None,
                scale_factor=None,
                cfg_dropout_prob=0):
        
        ### Get Text Conditioning ###
        if not self.config.pre_encoded_text and self.config.text_conditioning:
            assert (text_conditioning.dtype == torch.long), "CLIP expects text tokens in Long Tensor"
            if text_conditioning is not None:
                text_conditioning = self._encode_text(text_conditioning, text_attention_mask)

                ### Randomly Drop Conditioning Signal ###
                if cfg_dropout_prob > 0:
                    dropout_mask = torch.rand(text_conditioning.shape[0], device=text_conditioning.device) < cfg_dropout_prob
                    text_conditioning[dropout_mask] = 0
            else:
                ### If No Context is Passed we are doing Unconditional Generation ###
                text_conditioning = torch.zeros((images.shape[0], 1, self.config.text_embed_dim), device=images.device)

        ### Get Class Conditioning ###
        if self.config.class_conditioning:
            if class_conditioning is not None:
                class_conditioning = self.class_encoder(class_conditioning)

                ### Randomly Drop Conditioning Signal ###
                if cfg_dropout_prob > 0:
                    dropout_mask = torch.rand(class_conditioning.shape[0], device=class_conditioning.device) < cfg_dropout_prob
                    class_conditioning[dropout_mask] = self.class_encoder.unconditional_embedding
            else:
                # If No Context is Passed, we are doing Unconditional Generation (0 Embeddings) #
                class_conditioning = self.class_encoder.unconditional_embedding.unsqueeze(0).repeat(images.shape[0], 1)

        ### Compress Images using AutoEncoder ###
        compressed_images = self._vae_encode_images(images, scale_factor=scale_factor)

        ### Sample Random Timesteps for Noise ###
        timesteps = torch.randint(0, self.config.num_diffusion_timesteps, (images.shape[0], ))

        ### Add Noise to Images ###
        noisy_images, noise = self.ddpm_sampler.add_noise(compressed_images, timesteps)
        
        ### Get Timestep Embeddings ###
        timestep_embeddings = self.sinusoidal_time_embeddings(timesteps)

        ### Predict Noise with UNet Model ###
        noise_pred = self.unet(noisy_images, 
                               timestep_embeddings, 
                               text_conditioning=text_conditioning, 
                               text_attention_mask=text_attention_mask,
                               class_conditioning=class_conditioning)

        ### Compute Loss ###
        loss = loss_functions[self.config.loss_fn](noise_pred, noise.to(noise_pred.device))

        return loss

    
    @torch.no_grad()
    def inference(self,
                  noise, 
                  text_conditioning=None, 
                  text_attention_mask=None,
                  class_conditioning=None, 
                  cfg_weight=1.0):
        
        pass

if __name__ == "__main__":

    from .config import LDMConfig

    config = LDMConfig(text_conditioning=False, 
                       class_conditioning=True,
                       pre_encoded_text=True)

    model = LDM(config=config)

    rand_images = torch.randn(2,3,256,256)
    rand_text = torch.randint(0,1000, size=(2,))
    out = model(rand_images, class_conditioning=None)
    print(out)