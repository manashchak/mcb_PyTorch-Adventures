import os
import random
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms
from accelerate import Accelerator

from model import CycleGAN, ModelConfig
from dataset import Src2TgtDataset
from utils import load_testing_samples

#######################
### TRAINING CONFIG ###
#######################
experiment_name = "landscape2monet"
working_directory = "work_dir"
training_iterations = 25000
save_gen_iterations = 250
logging_iterations = 10
batch_size = 16
learning_rate = 3e-5
beta1 = 0.5
beta2 = 0.999
num_workers = 32
num_testing_samples = 5
cycle_loss_weight = 10.0

######################
### DATASET CONFIG ###
######################
path_to_src = "/mnt/datadrive/data/Pic2Monet/trainB"
path_to_tgt = "/mnt/datadrive/data/Pic2Monet/trainA"
path_to_testing_src = "/mnt/datadrive/data/Pic2Monet/testB"

########################
### INIT ACCELERATOR ###
########################
path_to_experiment = os.path.join(working_directory, experiment_name)
if not os.path.isdir(path_to_experiment):
    os.mkdir(path_to_experiment)

accelerator = Accelerator(project_dir=path_to_experiment)

###########################
### PREPARE DATALOADERS ###
###########################
transform = transforms.Compose(
    [
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5,], [0.5, 0.5, 0.5,])
    ]
) 
trainset = Src2TgtDataset(path_to_src, path_to_tgt, transform)
trainloader = DataLoader(trainset, 
                         batch_size=batch_size, 
                         shuffle=True, 
                         num_workers=num_workers)

#######################
### TESTING SAMPLES ###
#######################
path_to_testing_samples = [os.path.join(path_to_testing_src, file) \
                            for file in os.listdir(path_to_testing_src)]

######################
### LOSS FUNCTIONS ###
######################
reconstruction_loss = torch.nn.L1Loss()
discriminator_loss = torch.nn.MSELoss() # Not BCE! USES LSGANs MSE Loss for Discriminator

####################
### DEFINE MODEL ###
####################
config = ModelConfig()
model = CycleGAN(config=config)

#########################
### DEFINE OPTIMIZERS ###
#########################
optimizer_G = optim.Adam(
    params=list(model.generator_src2tgt.parameters()) + list(model.generator_tgt2src.parameters()), 
    lr=learning_rate, 
    betas=(beta1, beta2)
)

optimizer_D = optim.Adam(
    params=list(model.discriminator_src.parameters()) + list(model.discriminator_tgt.parameters()), 
    lr=learning_rate, 
    betas=(beta1, beta2)
)

##########################
### PREPARE EVERYTHING ###
##########################
model, trainloader, optimizer_G, optimizer_D = accelerator.prepare(
    model, trainloader, optimizer_G, optimizer_D
)

#####################
### TRAINING LOOP ###
#####################
train = True
pbar = tqdm(range(training_iterations), disable=not accelerator.is_main_process)
completed_steps = 0

while train:

    for src_imgs, tgt_imgs in trainloader:

        ### Get Real Images ###
        src_imgs = src_imgs.to(accelerator.device)
        tgt_imgs = tgt_imgs.to(accelerator.device)

        ### Pass Through Generators ###
        fake_tgt_images = accelerator.unwrap_model(model).generator_src2tgt(src_imgs)
        fake_src_images = accelerator.unwrap_model(model).generator_tgt2src(tgt_imgs)

        ##########################
        ### TRAIN DISCIMINATOR ###
        ##########################
        
        ### Compute tgt discriminator loss (stop gradient so no update on generator) ###
        disc_real_tgt = accelerator.unwrap_model(model).discriminator_tgt(tgt_imgs)
        disc_fake_tgt = accelerator.unwrap_model(model).discriminator_tgt(fake_tgt_images.detach())

        disc_real_tgt_loss = discriminator_loss(disc_real_tgt, torch.ones_like(disc_real_tgt))
        disc_fake_tgt_loss = discriminator_loss(disc_fake_tgt, torch.zeros_like(disc_fake_tgt))
        disc_tgt_loss = disc_real_tgt_loss + disc_fake_tgt_loss

        ### Update src Discriminator ###
        disc_real_src = accelerator.unwrap_model(model).discriminator_src(src_imgs)
        disc_fake_src = accelerator.unwrap_model(model).discriminator_src(fake_src_images.detach())

        disc_real_src_loss = discriminator_loss(disc_real_src, torch.ones_like(disc_real_src))
        disc_fake_src_loss = discriminator_loss(disc_fake_src, torch.zeros_like(disc_real_src))
        disc_src_loss = disc_real_src_loss + disc_fake_src_loss

        ### Total Discriminator Loss ###
        disc_loss = (disc_tgt_loss + disc_src_loss) / 2

        ### Update Discriminator ###
        optimizer_D.zero_grad()
        accelerator.backward(disc_loss)
        optimizer_D.step()

        #######################
        ### TRAIN GENERATOR ###
        #######################

        ### Pass fake images (without detach) to discriminators to update generators ###
        gen_discriminator_src_pred = accelerator.unwrap_model(model).discriminator_src(fake_src_images)
        gen_discriminator_tgt_pred = accelerator.unwrap_model(model).discriminator_tgt(fake_tgt_images)


        ### Generator Loss (we want to predict our generated images as real (1)) ###
        tgt2src_generator_loss = discriminator_loss(gen_discriminator_src_pred, torch.ones_like(gen_discriminator_src_pred))
        src2tgt_generator_loss = discriminator_loss(gen_discriminator_tgt_pred, torch.ones_like(gen_discriminator_tgt_pred))

        ### Cycle Consistency Loss ###
        ### If we take src, pass to src2tgt, and then pass to tgt2src, we should be back to src and vice-versa ###
        cycle_src = accelerator.unwrap_model(model).generator_tgt2src(fake_tgt_images)
        cycle_tgt = accelerator.unwrap_model(model).generator_src2tgt(fake_src_images)

        cycle_src_loss = reconstruction_loss(cycle_src, src_imgs)
        cycle_tgt_loss = reconstruction_loss(cycle_tgt, tgt_imgs)

        ### Total Generator Loss ###
        gen_loss = tgt2src_generator_loss + src2tgt_generator_loss + \
                    cycle_loss_weight * (cycle_src_loss + cycle_tgt_loss)

        ### Update Generator ###
        optimizer_G.zero_grad()
        accelerator.backward(gen_loss)
        optimizer_G.step()

        if accelerator.is_main_process and (completed_steps % logging_iterations) == 0:
            loss_string = (
                f"D_Total: {disc_loss.item():.4f}, "
                f"G_Total: {gen_loss.item():.4f}"
            )
            accelerator.print(loss_string)

        ### Generate Example Style Transfers ###
        if completed_steps % save_gen_iterations == 0 and accelerator.is_main_process:
        
            test_samples = load_testing_samples(path_to_testing_samples, 
                                                k=num_testing_samples, 
                                                transforms=transform)
            with torch.no_grad():
                style_transfers = accelerator.unwrap_model(model).generator_src2tgt(test_samples.to(accelerator.device))
            
            ### Rescale from -1 to 1 to 0 to 255 ###
            test_samples = torch.clamp(test_samples, -1., 1.)
            test_samples_vis = (test_samples + 1) / 2
            test_samples_vis = test_samples_vis.cpu().permute(0,2,3,1).numpy()
            test_samples_vis = (255 * test_samples_vis).astype(np.uint8)
            test_samples_vis = [Image.fromarray(img).convert("RGB") for img in test_samples_vis]

            style_transfers = torch.clamp(style_transfers, -1., 1.)
            style_transfers_vis = (style_transfers + 1) / 2
            style_transfers_vis = style_transfers_vis.cpu().permute(0,2,3,1).numpy()
            style_transfers_vis = (255 * style_transfers_vis).astype(np.uint8)
            style_transfers_vis = [Image.fromarray(img).convert("RGB") for img in style_transfers_vis]

            ### Plot original and generated images ###
            fig, axes = plt.subplots(2, num_testing_samples, figsize=(15, 6))
            for i in range(num_testing_samples):
                
                axes[0, i].imshow(test_samples_vis[i])
                axes[0, i].axis('off')

                if i == 0:
                    axes[0, i].set_title('Original')

                axes[1, i].imshow(style_transfers_vis[i])
                axes[1, i].axis('off')

                if i == 0:
                    axes[1, i].set_title('Monet')

            plt.tight_layout()
            plt.savefig(os.path.join("gens", f"iteration_{completed_steps}.png"))

        if completed_steps  >= training_iterations:
            train = False
            accelerator.print("Completed Training")
            break 

        accelerator.wait_for_everyone()
        completed_steps += 1
        pbar.update(1)

accelerator.save_state(output_dir=os.path.join(path_to_experiment, "final_checkpoint"))

accelerator.end_training()

