import argparse

def experiment_config_parser(parser):

    parser = parser.add_argument_group("Experiment Configuration")

    parser.add_argument("--experiment_name", 
                        help="Name of Experiment being Launched", 
                        required=True, 
                        type=str,
                        metavar="experiment_name")
    
    parser.add_argument("--wandb_run_name",
                        required=True, 
                        type=str,
                        metavar="wandb_run_name")
    
    parser.add_argument("--path_to_data", 
                        help="Path to ImageNet root folder which should contain \\train and \\validation folders", 
                        required=True, 
                        type=str,
                        metavar="path_to_data")
    
    parser.add_argument("--working_directory", 
                        help="Working Directory where checkpoints and logs are stored, inside a \
                        folder labeled by the experiment name", 
                        required=True, 
                        type=str,
                        metavar="working_directory")

    parser.add_argument("--log_wandb",
                        action=argparse.BooleanOptionalAction, 
                        help="Do you want to log to WandB?")
    
def vae_config(parser):

    parser = parser.add_argument_group("VAE Configuration")

    parser.add_argument("--img_size",
                        help="Input image resolution for VAE",
                        default=256,
                        type=int,
                        metavar="img_size")

    parser.add_argument("--in_channels",
                        help="Number of input channels for images",
                        default=3,
                        type=int,
                        metavar="in_channels")

    parser.add_argument("--out_channels",
                        help="Number of output channels for VAE",
                        default=3,
                        type=int,
                        metavar="out_channels")

    parser.add_argument("--latent_channels",
                        help="Number of latent channels in compressed space",
                        default=4,
                        type=int,
                        metavar="latent_channels")

    parser.add_argument("--residual_layers_per_block",
                        help="Number of residual layers per block in the encoder",
                        default=2,
                        type=int,
                        metavar="residual_layers")

    parser.add_argument("--attention_layers",
                        help="Number of attention layers per block in the encoder",
                        default=1,
                        type=int,
                        metavar="attention_layers")

    parser.add_argument("--attention_residual_connections",
                        help="Use residual connections in attention layers",
                        action=argparse.BooleanOptionalAction)

    parser.add_argument("--vae_channels_per_block",
                        help="Number of channels per block for VAE",
                        nargs="+",
                        default=(128, 256, 512, 512),
                        type=int,
                        metavar="channels_per_block")

    parser.add_argument("--vae_up_down_factor",
                        help="Scaling factor for up/downsampling in the VAE",
                        default=2,
                        type=int,
                        metavar="up_down_factor")

    parser.add_argument("--vae_up_down_kernel_size",
                        help="Kernel size for up/downsampling operations in the VAE",
                        default=3,
                        type=int,
                        metavar="kernel_size")

    parser.add_argument("--quantize",
                        action=argparse.BooleanOptionalAction,
                        help="Enable quantization for VAE")

    parser.add_argument("--codebook_size",
                        help="Number of embeddings in the codebook for vector quantization",
                        default=16384,
                        type=int,
                        metavar="codebook_size")

    parser.add_argument("--vq_embed_dim",
                        help="Embedding dimension for vector quantization",
                        default=4,
                        type=float,
                        metavar="vq_embed_dim")

    parser.add_argument("--beta",
                        help="Beta parameter for quantization commitment loss",
                        default=0.25,
                        type=float,
                        metavar="beta_value")

def discriminator_config(parser):

    parser = parser.add_argument_group("Discriminator Configuration")

    parser.add_argument('--disable_discriminator',
                        help="Flag to turn off GAN Loss",
                        action=argparse.BooleanOptionalAction)

    parser.add_argument("--disc_start_dim",
                        help="Starting channels projection for PatchGAN",
                        default=64, 
                        type=int,
                        metavar="disc_start_dim")
    
    parser.add_argument("--disc_depth",
                        help="Number of Convolution Blocks in PatchGAN",
                        default=3, 
                        type=int,
                        metavar="disc_depth")
    
    parser.add_argument("--disc_kernel_size",
                        help="Kernel size for convolutions in PatchGAN",
                        default=4, 
                        type=int,
                        metavar="disc_kernel_size")
    
    parser.add_argument("--disc_leaky_relu_slope",
                        help="Negative Slope for Leaky Relu",
                        default=0.2, 
                        type=float,
                        metavar="disc_leaky_relu_slope")
    
    parser.add_argument("--disc_learning_rate", 
                        help="max discriminator learning rate in cosine schedule",
                        default=4.5e-6,
                        type=float,
                        metavar="disc_learning_rate")
    
    parser.add_argument("--disc_scheduler", 
                        help="What LR Scheduler do you want for Discriminator?",
                        default="constant",
                        choices=("constant", "linear", "cosine"),
                        type=str,
                        metavar="disc_scheduler")
    
    parser.add_argument("--disc_lr_warmup_steps", 
                        help="How many warmup steps do you want in your discriminator scheduler?",
                        default=2000, 
                        type=int,
                        metavar="disc_lr_warmup_steps")

    parser.add_argument("--disc_start", 
                        help="Whats step do you want the disciminator loss to begin?",
                        default=50001,
                        type=int,
                        metavar="disc_start")
    
    parser.add_argument("--disc_weight", 
                        help="Multiplicative factor for discriminator",
                        default=1.0,
                        type=float,
                        metavar="disc_weight")
    
    parser.add_argument("--disc_loss", 
                        help="What loss function for the discriminator?",
                        default="hinge",
                        type=str,
                        metavar="disc_loss")
    
def lpips_config(parser):

    parser = parser.add_argument_group("LPIPS Configuration")

    parser.add_argument('--disable_lpips',
                        help="Flag to turn off LPIPS Loss",
                        action=argparse.BooleanOptionalAction)

    parser.add_argument("--use_lpips_package", 
                        help="Flag to use the original LPIPS package, otherwise own implementation",
                        action=argparse.BooleanOptionalAction)
    
    parser.add_argument("--lpips_checkpoint",
                        help="Checkpoint to our own LPIPS implementation",
                        default="lpips_vgg.pt", 
                        type=str,
                        metavar="lpips_checkpoint")
    
    parser.add_argument("--lpips_weight",
                        help="Multiplicative factor for lpips loss",
                        default=0.5, 
                        type=float,
                        metavar="lpips_weight")
    
def training_config(parser):

    parser = parser.add_argument_group("Generic Training Configuration")

    parser.add_argument("--learning_rate", 
                        help="max learning rate in cosine schedule",
                        default=4.5e-6,
                        type=float,
                        metavar="learning_rate")
    
    parser.add_argument("--lr_warmup_steps",
                        help="How many steps to warmup Learning Rate", 
                        default=2000,
                        type=int,
                        metavar="warmup_steps")
    
    parser.add_argument("--total_train_iterations", 
                        help="Number of training iterations",
                        default=100000,
                        type=int,
                        metavar="total_training_iterations")
    
    parser.add_argument("--checkpoint_iterations",
                        help="After every how many iterations to save checkpoint",
                        default=2500,
                        type=int,
                        metavar="checkpoint_iterations")
    
    parser.add_argument("--per_gpu_batch_size", 
                        help="How many sampels do you want per gpu (multipled by n_gpus)", 
                        default=64, 
                        type=int,
                        metavar="per_gpu_batch_size")
    
    parser.add_argument("--gradient_accumulation_steps",
                        help="Number of gradient steps (splitting batch_size)",
                        default=1, 
                        type=int,
                        metavar="gradient_accumulation_steps")
    
    parser.add_argument("--num_workers",
                        help="How many workers for DataLoader?",
                        default=8, 
                        type=int,
                        metavar="num_workers")
    
    parser.add_argument("--max_grad_norm", 
                        help="Maximum norm for gradient clipping", 
                        default=1.0, 
                        type=float,
                        metavar="max_grad_norm")
    

def vae_training_configuration(parser):

    parser = parser.add_argument_group("VAE Training Configurations")

    parser.add_argument("--train_decoder_only",
                        help="Do you only want to train the decoder (good for finetuning)",
                        action=argparse.BooleanOptionalAction)
    
def optimizer_config(parser):
    
    parser = parser.add_argument_group("Optimizer Configurations")

    parser.add_argument("--beta1", 
                        default=0.9, 
                        help="Beta 1 parameter for momentum calculation",
                        type=float,
                        metavar="beta1")
    
    parser.add_argument("--beta2",
                        default=0.999,
                        help="Beta 2 parameter for momentum calculation",
                        type=float,
                        metavar="beta2")
    
    parser.add_argument("--weight_decay", 
                        help="Weight decay for optimizer", 
                        default=0.05, 
                        type=float,
                        metavar="weight_decay")
    
    parser.add_argument("--bias_weight_decay",
                        help="Apply weight decay to bias",
                        default=False, 
                        action=argparse.BooleanOptionalAction)
    
    parser.add_argument("--norm_weight_decay",
                        help="Apply weight decay to normalization weight and bias",
                        default=False, 
                        action=argparse.BooleanOptionalAction)
    
    

def vae_trainer_cli_parser():

    parser = argparse.ArgumentParser(description="CLI Parser for AutoEncoder Training")

    experiment_config_parser(parser)
    vae_config(parser)
    discriminator_config(parser)
    lpips_config(parser)
    training_config(parser)
    vae_training_configuration(parser)
    optimizer_config(parser)

    return parser



if __name__ == "__main__":
    parser = vae_trainer_cli_parser()
    args = parser.parse_args()
    print(args)