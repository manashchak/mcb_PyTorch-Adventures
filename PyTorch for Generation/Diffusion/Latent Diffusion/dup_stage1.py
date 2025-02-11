import os
from accelerate import Accelerator
import torch
from torch import optim
import torch.nn.functional as F
from transformers import get_scheduler

from modules import VAE, VQVAE, LDMConfig, VAELpipsDiscriminatorLoss
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

### Load Loss Function ###
loss_fn = VAELpipsDiscriminatorLoss(disc_start=args.disc_start, 
                                    use_disc=not args.disable_discriminator,
                                    disc_in_channels=args.in_channels, 
                                    disc_start_dim=args.disc_start_dim,
                                    disc_depth=args.disc_depth, 
                                    disc_kernel_size=args.disc_kernel_size, 
                                    disc_leaky_relu_slope=args.disc_leaky_relu_slope,
                                    disc_loss=args.disc_loss, 
                                    disc_weight=0, #args.disc_weight,
                                    use_lpips=not args.disable_lpips, 
                                    use_lpips_package=True, 
                                    path_to_lpips_checkpoint=args.lpips_checkpoint, 
                                    lpips_weight=args.lpips_weight, 
                                    reconstruction_loss=args.reconstruction_loss_fn, 
                                    use_logvar_scaling=args.scale_perceptual_by_var)

### Load Optimizers (model params and output_logvar) and Schedulers ###
params = list(model.parameters()) + [loss_fn.output_logvar]
optimizer = optim.Adam(params, 
                       lr=args.learning_rate, 
                       betas=(args.beta1, args.beta2),
                       weight_decay=args.weight_decay)

scheduler = get_scheduler(name=args.lr_scheduler,
                          optimizer=optimizer, 
                          num_training_steps=args.total_train_iterations * accelerator.num_processes, 
                          num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes)

if not args.disable_discriminator:
    use_disc = not args.disable_discriminator
    disc_optimizer = optim.Adam(loss_fn.discriminator.parameters(), 
                                lr=args.learning_rate, 
                                betas=(args.beta1, args.beta2),
                                weight_decay=args.weight_decay)

    disc_scheduler = get_scheduler(name=args.disc_lr_scheduler,
                                optimizer=disc_optimizer, 
                                num_training_steps=args.total_train_iterations * accelerator.num_processes, 
                                num_warmup_steps=args.disc_lr_warmup_steps * accelerator.num_processes)


### Prepare Everything ###
model, loss_fn, optimizer, disc_optimizer, scheduler, disc_scheduler, dataloader = accelerator.prepare(
    model, loss_fn, optimizer, disc_optimizer, scheduler, disc_scheduler, dataloader
)

### Start Training ###
global_step = 0
train = True
print_disc_start = True

### Initialize Variables to Accumulate ###
vae_log = {"loss": 0,
           "perceptual_loss": 0,
           "reconstruction_loss": 0, 
           "lpips_loss": 0,
           "kl_loss": 0,
           "generator_loss": 0}

disc_log = {"disc_loss": 0, 
            "logits_real": 0,
            "logits_fake": 0}

### Quick Helper to Rest Logs ###
def reset_log(log):
    return {key: 0 for (key, _) in log.items()}

while train:

    ### Set Everything to Training Mode ###
    model.train()
    if use_disc:
        accelerator.unwrap_model(loss_fn).discriminator.train()

    for batch in dataloader:

        ### Inference Model ###        
        images = batch["images"]
        model_output = model(images)

        ### Parse VAE Output ###
        posterior = model_output["posterior"]
        reconstruction = model_output["reconstruction"]
        kl_loss = model_output["kl_loss"]
        
        
        ### Toggles for Updating Discriminator vs Generator ###
        gd_toggle = (global_step % 2 == 0)
        train_disc = (global_step >= args.disc_start)

        if train_disc and print_disc_start:
            print_disc_start = False
            accelerator.print("STARTING GAN TRAINING")

        ### If train_disc is false then we will enter this condition and train only the VAE ###
        ### If trian disc is true, then we will enter this condition only on even steps ##
        ### on odd steps we will update the discriminator ###
        if gd_toggle or not train_disc: 
            
            with accelerator.accumulate(model):
                
                ### Compute Perceptual Loss (Reconstruction + LPIPS) ###
                perceptual_loss, reconstruction_loss, lpips_loss = \
                    accelerator.unwrap_model(loss_fn).forward_perceptual_loss(images, reconstruction)

                ### Average KL Loss ###
                kl_loss = kl_loss.mean()

                ### Compute Generator Loss (for GAN Task) if train_disc is True ###
                generator_loss = torch.zeros(size=(), device=images.device)
                adaptive_weight = torch.zeros(size=(), device=images.device)

                ### Check, if train_disc is now True, and an Even Step (previous condition) ###
                if train_disc: 
                    
                    last_layer = accelerator.unwrap_model(model).decoder.conv_out.weight
                    generator_loss, adaptive_weight = \
                        accelerator.unwrap_model(loss_fn).forward_generator_loss(reconstruction, 
                                                                                 perceptual_loss,
                                                                                 last_layer)
            
                ### Compute Total Weighted Loss ###
                loss = perceptual_loss + args.kl_weight * kl_loss + args.disc_weight * adaptive_weight * generator_loss

                ### Update Model ###
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                ### Create Log of Everything ###
                log = {"loss": loss,
                       "perceptual_loss": perceptual_loss,
                       "reconstruction_loss": reconstruction_loss, 
                       "lpips_loss": lpips_loss,
                       "kl_loss": kl_loss,
                       "generator_loss": generator_loss}
                
                ### Accumulate Log ###
                for key, value in log.items():
                    vae_log[key] += value.mean() / args.gradient_accumulation_steps

        ### On Odd Steps (when train_disc is True) we will update the discriminator ###
        else:

            discriminator = accelerator.unwrap_model(loss_fn).discriminator
            
            with accelerator.accumulate(discriminator):

                loss, logits_real, logits_fake = \
                    accelerator.unwrap_model(loss_fn).forward_discriminator_loss(images, reconstruction)
                
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(discriminator.parameters(), args.max_grad_norm)

                disc_optimizer.step()
                disc_scheduler.step()
                disc_optimizer.zero_grad(set_to_none=True)

                log = {"disc_loss": loss.item(), 
                       "logits_real": logits_real.item(),
                       "logits_fake": logits_fake.item()}
                
        ### Every Gradient Sync Marks the End of a Full Accumulation Step ###
        if accelerator.sync_gradients:
            
            ### If we updated the VAE ###
            if gd_toggle or not train_disc:

                ## Gather Across GPUs ###
                vae_log = {key: accelerator.gather_for_metrics(value).mean().item() for key, value in vae_log.items()}
                vae_log["lr"] = scheduler.get_last_lr()[0]
                accelerator.log(vae_log, step=global_step)

                ### Reset Log for Next Accumulation ###
                vae_log = reset_log(vae_log)

            ### If we updated the Discriminator ###
            else:

                ## Gather Across GPUs ###
                disc_log = {key: accelerator.gather_for_metrics(value).mean().item() for key, value in disc_log.items()}
                disc_log["lr"] = disc_scheduler.get_last_lr()[0]
                accelerator.log(disc_log, step=global_step)

                ### Reset Log for Next Accumulation ###
                disc_log = reset_log(disc_log)
                
            global_step += 1

        if global_step >= args.total_train_iterations:
            train = False
            accelerator.print("COMPLETED TRAINING!!")
            break