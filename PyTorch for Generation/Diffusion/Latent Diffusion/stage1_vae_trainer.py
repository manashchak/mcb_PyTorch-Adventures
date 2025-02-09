import os
import argparse
from accelerate import Accelerator
import torch
from datasets import load_dataset
import lpips
from modules import LPIPS as mylpips
from modules import PatchGAN
from modules import VAE, VQVAE


def parse_arguments():

    parser = argparse.ArgumentParser(description="AutoEncoder Training Configuration")

    ### EXPERIMENT CONFIG ###
    parser.add_argument("--experiment_name", 
                        help="Name of Experiment being Launched", 
                        required=True, 
                        type=str)
    
    parser.add_argument("--wandb_run_name",
                        required=True, 
                        type=str)
    
    parser.add_argument("--path_to_data", 
                        help="Path to ImageNet root folder which should contain \train and \validation folders", 
                        required=True, 
                        type=str)
    
    parser.add_argument("--working_directory", 
                        help="Working Directory where checkpoints and logs are stored, inside a \
                        folder labeled by the experiment name", 
                        required=True, 
                        type=str)
    
    ### MODEL CONFIGURATION ###
    parser.add_argument("--img_size",
                        help="Input image resolution for VAE",
                        default=256,
                        type=int)

    parser.add_argument("--in_channels",
                        help="Number of input channels for images",
                        default=3,
                        type=int)

    parser.add_argument("--out_channels",
                        help="Number of output channels for VAE",
                        default=3,
                        type=int)

    parser.add_argument("--latent_channels",
                        help="Number of latent channels in compressed space",
                        default=4,
                        type=int)

    parser.add_argument("--residual_layers_per_block",
                        help="Number of residual layers per block in the encoder",
                        default=2,
                        type=int)

    parser.add_argument("--attention_layers",
                        help="Number of attention layers per block in the encoder",
                        default=1,
                        type=int)

    parser.add_argument("--attention_residual_connections",
                        help="Use residual connections in attention layers",
                        action='store_true')

    parser.add_argument("--vae_channels_per_block",
                        help="Number of channels per block for VAE",
                        nargs="+",
                        default=(128,256,512,512),
                        type=int)

    parser.add_argument("--vae_up_down_factor",
                        help="Scaling factor for up/downsampling in the VAE",
                        default=2,
                        type=int)

    parser.add_argument("--vae_up_down_kernel_size",
                        help="Kernel size for up/downsampling operations in the VAE",
                        default=3,
                        type=int)

    parser.add_argument("--quantize",
                        action=argparse.BooleanOptionalAction,
                        help="Enable quantization for VAE")

    parser.add_argument("--codebook_size",
                        help="Number of embeddings in the codebook for vector quantization",
                        default=16384,
                        type=int)

    parser.add_argument("--vq_embed_dim",
                        help="Embedding dimension for vector quantization",
                        default=4,
                        type=float)

    parser.add_argument("--beta",
                        help="Beta parameter for quantization commitment loss",
                        default=0.25,
                        type=float)
    
    ### DISCRIMINATOR CONFIG ###
    parser.add_argument('--disable_discriminator',
                        help="Flag to turn off GAN Loss",
                        action=argparse.BooleanOptionalAction)

    parser.add_argument("--disc_start_dim",
                        help="Starting channels projection for PatchGAN",
                        default=64, 
                        type=int)
    
    parser.add_argument("--disc_depth",
                        help="Number of Convolution Blocks in PatchGAN",
                        default=3, 
                        type=int)
    
    parser.add_argument("--disc_kernel_size",
                        help="Kernel size for convolutions in PatchGAN",
                        default=4, 
                        type=int)
    
    parser.add_argument("--disc_leaky_relu_slope",
                        help="Negative Slope for Leaky Relu",
                        default=0.2, 
                        type=float)
    
    parser.add_argument("--disc_learning_rate", 
                        help="max discriminator learning rate in cosine schedule",
                        default=4.5e-6,
                        type=float)
    
    parser.add_argument("--disc_scheduler", 
                        help="What LR Scheduler do you want for Discriminator?",
                        default="constant",
                        choices=("constant", "linear", "cosine"),
                        type=str)
    
    parser.add_argument("--disc_start", 
                        help="Whats step do you want the disciminator loss to begin?",
                        default=50001,
                        type=int)
    
    parser.add_argument("--disc_weight", 
                        help="Multiplicative factor for discriminator",
                        default=1.0,
                        type=float)
    
    parser.add_argument("--disc_loss", 
                        help="What loss function for the discriminator?",
                        default="hinge",
                        type=str)
    
    ### LPIPS ###
    parser.add_argument('--disable_lpips',
                        help="Flag to turn off LPIPS Loss",
                        action=argparse.BooleanOptionalAction)

    parser.add_argument("--use_lpips_package", 
                        help="Flag to use the original LPIPS package, otherwise own implementation",
                        action=argparse.BooleanOptionalAction)
    
    parser.add_argument("--lpips_checkpoint",
                        help="Checkpoint to our own LPIPS implementation",
                        default="lpips_vgg.pt", 
                        type=str)
    
    parser.add_argument("--lpips_weight",
                        help="Multiplicative factor for lpips loss",
                        default=0.5, 
                        type=float)
    
    ### TRAINING CONFIG ###
    parser.add_argument("--learning_rate", 
                        help="max learning rate in cosine schedule",
                        default=4.5e-6,
                        type=float)
    
    parser.add_argument("--lr_warmup_steps",
                        help="How many steps to warmup Learning Rate", 
                        default=2000,
                        type=int)
    
    parser.add_argument("--total_train_iterations", 
                        help="Number of training iterations",
                        default=100000,
                        type=int)
    
    parser.add_argument("--checkpoint_iterations",
                        help="After every how many iterations to save checkpoint",
                        default=2500,
                        type=int)
    
    parser.add_argument("--")
    
    ### LOGGING CONFIG ###


    ### DATASET CONFIG ###
    

