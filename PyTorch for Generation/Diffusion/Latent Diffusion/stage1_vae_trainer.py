import os
from accelerate import Accelerator
import torch
from datasets import load_dataset
import lpips
from modules import LPIPS as mylpips
from modules import PatchGAN, VAE, VQVAE
from cli_parser import vae_trainer_cli_parser

### Load Arguments ###
parser = vae_trainer_cli_parser()
args = parser.parse_args_()
print(args)