import os
from accelerate import Accelerator
from torch import optim
import lpips
from modules import LPIPS as mylpips
from modules import PatchGAN, VAE, VQVAE
from modules import LDMConfig
from cli_parser import vae_trainer_cli_parser
from dataset import get_dataset

### Load Arguments ###
parser = vae_trainer_cli_parser()
args = parser.parse_args()

### Initialize Accelerate ###
path_to_experiment = os.path.join(args.working_directory, args.experiment_name)
accelerator = Accelerator(project_dir=path_to_experiment,
                          gradient_accumulation_steps=args.gradient_accumulation_steps,
                          log_with="wandb" if args.log_wandb else None)

### Load Dataset ###
mini_batchsize = args.per_gpu_batch_size // args.gradient_accumulation_steps 
dataloader = get_dataset(dataset=args.dataset,
                         path_to_data=args.path_to_data,
                         batch_size=mini_batchsize,
                         num_channels=args.in_channels, 
                         img_size=args.img_size, 
                         random_resize=args.random_resize, 
                         interpolation=args.interpolation, 
                         random_flip_p=args.random_flip_p,
                         return_caption=False,
                         num_workers=args.num_workers,
                         pin_memory=args.pin_memory)

### Load Config ###
config = LDMConfig(img_size=args.img_size, 
                   in_channels=args.in_channels, 
                   out_channels=args.out_channels, 
                   latent_channels=args.latent_channels, 
                   residual_layers_per_block=args.residual_layers_per_block,
                   attention_layers=args.attention_layers,
                   attention_residual_connections=args.attention_residual_connections, 
                   vae_channels_per_block=args.vae_channels_per_block,
                   vae_up_down_factor=args.vae_up_down_factor, 
                   vae_up_down_kernel_size=args.vae_up_down_kernel_size, 
                   codebook_size=args.codebook_size, 
                   vq_embed_dim=args.vq_embed_dim, 
                   beta=args.beta)   

### Load VAE/VQ-VAE Model ###
if args.quantize:
    accelerator.print("Training VQ-VAE Model")
    model = VQVAE(config)
else:
    accelerator.print("Training VAE Model")
    model = VAE(config)

total_parameters = 0
for name, param in model.named_parameters():
    if param.requires_grad_:
        total_parameters += param.numel()
print("Number of Training Parameters:", total_parameters)

### LOAD LPIPS ###
use_lpips = not args.disable_lpips
if use_lpips:
    lpips_model = lpips.LPIPS().eval()
else:
    lpips_model = mylpips()
    lpips_model.load_checkpoint(args.lpips_checkpoint)

### Load PatchGAN ###
use_disc = not args.disable_discriminator
if use_disc:
    discriminator = PatchGAN(input_channels=args.input_channels,
                             start_dim=args.disc_start_dim, 
                             depth=args.disc_depth, 
                             kernel_size=args.disc_kernel_size, 
                             leaky_relu_slope=args.disc_leaky_relu_slope)

### Load Optimizers ###
optimizer = optim.Adam(model.parameters(), 
                       lr=args.learning_rate,
                       betas=(args.beta1, args.beta2), 
                       weight_decay=args.weight_decay)

if use_disc:
    disc_optimizer = optim.Adam(discriminator.parameters(), 
                                lr=args.disc_learning_rate, 
                                betas=(args.beta1, args.beta2),
                                weight_decay=args.weight_decay)
    
### Load Schedulers ###

