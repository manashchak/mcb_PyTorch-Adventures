import os
from accelerate import Accelerator
import torch
from torch import optim
import torch.nn.functional as F
import lpips
from transformers import get_scheduler
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

    if args.use_lpips_package:
        lpips_model = lpips.LPIPS(net="vgg").eval()
    else:
        lpips_model = mylpips()
        lpips_model.load_checkpoint(args.lpips_checkpoint)

### Load PatchGAN ###
use_disc = not args.disable_discriminator
if use_disc:
    discriminator = PatchGAN(input_channels=args.in_channels,
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
scheduler = get_scheduler(name=args.lr_scheduler,
                          optimizer=optimizer, 
                          num_training_steps=args.total_train_iterations * accelerator.num_processes, 
                          num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes)

if use_disc:
    num_disc_steps = args.total_train_iterations - args.disc_start
    disc_scheduler = get_scheduler(name=args.disc_lr_scheduler,
                                   optimizer=disc_optimizer, 
                                   num_training_steps=num_disc_steps * accelerator.num_processes, 
                                   num_warmup_steps=args.disc_lr_warmup_steps * accelerator.num_processes)
    
### Prepare Everything ###
model, lpips_model, discriminator, optimizer, disc_optimizer, scheduler, disc_scheduler, dataloader = accelerator.prepare(
    model, lpips_model, discriminator, optimizer, disc_optimizer, scheduler, disc_optimizer, dataloader
)

### Start Training ###
global_step = 0
train = True

while train:

    ### Set Everything to Training Mode ###
    model.train()
    if use_disc:
        discriminator.train()

    for batch in dataloader:

        ### Inference Model ###        
        images = batch["images"]
        model_output = model(images)

        ### Parse VAE Output ###
        posterior = model_output["posterior"]
        reconstruction = model_output["reconstruction"]
        kl_loss = model_output["kl_loss"]

        ### Train only VAE until disc_start, and then alternate vae and patchgan updates ###
        if (global_step % 2 == 0) or (global_step < args.disc_start):

            with accelerator.accumulate(model):
                
                ### Calculate Reconstruction Loss ###
                reconst_loss = F.mse_loss(reconstruction, images, reduction="none")

                ### Comute Perceptual Loss ###
                with torch.no_grad():
                    lpips_loss = lpips_model(reconstruction, images)
                
                print(lpips_loss)
    
        else:
            continue
        
        global_step += 1

        if global_step >= args.total_train_iterations:
            train = False
            break
            

