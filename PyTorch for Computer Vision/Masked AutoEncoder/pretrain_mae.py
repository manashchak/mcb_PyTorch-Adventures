import os
import numpy as np
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import Compose, Normalize, RandomHorizontalFlip, RandomResizedCrop, ToTensor, Resize
from torchvision.transforms.functional import InterpolationMode
from accelerate import Accelerator
from transformers import get_cosine_schedule_with_warmup

from MaskedAutoEncoder import MAEConfig, ViTMAEForPreTraining

import warnings 
warnings.filterwarnings("ignore")

def parse_args():
    ### Parse Training Arguments ###
    parser = argparse.ArgumentParser(description="Arguments for Image Classification Training")

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


    ### TRAINING CONFIG ###
    parser.add_argument("--epochs",
                        help="Number of Epochs to Train",
                        default=800, 
                        type=int)
    parser.add_argument("--warmup_epochs",
                        help="Number of Epochs to Train",
                        default=40, 
                        type=int)
    parser.add_argument("--save_checkpoint_interval", 
                        help="After how many epochs to save model checkpoints",
                        default=10,
                        type=int)
    parser.add_argument("--per_gpu_batch_size", 
                        help="Effective batch size. If split_batches is false, batch size is \
                            multiplied by number of GPUs utilized ", 
                        default=256, 
                        type=int)
    parser.add_argument("--gradient_accumulation_steps", 
                        help="Number of Gradient Accumulation Steps for Training", 
                        default=1, 
                        type=int)
    parser.add_argument("--learning_rate", 
                        help="Starting Learning Rate for Cosine Scheduler Learning Rate", 
                        default=1.5e-4,
                        type=float)
    parser.add_argument("--weight_decay", 
                        help="Weight decay for optimizer", 
                        default=0.05, 
                        type=float)
    parser.add_argument("--bias_weight_decay",
                        help="Apply weight decay to bias",
                        default=False, 
                        action=argparse.BooleanOptionalAction)
    parser.add_argument("--norm_weight_decay",
                        help="Apply weight decay to normalization weight and bias",
                        default=False, 
                        action=argparse.BooleanOptionalAction)
    parser.add_argument("--max_grad_norm", 
                        help="Maximum norm for gradient clipping", 
                        default=1.0, 
                        type=float)

    ### DATALOADER CONFIG ###
    parser.add_argument("--img_size", 
                        help="Width and Height of Images passed to model", 
                        default=224, 
                        type=int)
    parser.add_argument("--input_channels", 
                        help="Number of channels in image", 
                        default=3, 
                        type=int)
    parser.add_argument("--num_workers", 
                        help="Number of workers for DataLoader", 
                        default=32, 
                        type=int)
    
    ### MODEL CONFIG ###
    parser.add_argument("--encoder_embed_dim", 
                        help="Encoder Embedding Dimension", 
                        default=768, 
                        type=int)
    parser.add_argument("--encoder_depth", 
                        help="Number of Transformer Blocks in Encoder", 
                        default=12, 
                        type=int)
    parser.add_argument("--encoder_num_heads", 
                        help="Number of Attention Heads in Encoder Transformers", 
                        default=12, 
                        type=int)
    parser.add_argument("--encoder_mlp_ratio", 
                        help="MLP Projection Ratio in Encoder Transformers", 
                        default=4, 
                        type=int)
    parser.add_argument("--decoder_embed_dim", 
                        help="Decoder Embedding Dimension", 
                        default=512, 
                        type=int)
    parser.add_argument("--decoder_depth", 
                        help="Number of Transformer Blocks in Decoder", 
                        default=8, 
                        type=int)
    parser.add_argument("--decoder_num_heads", 
                        help="Number of Attention Heads in Decoder Transformers", 
                        default=16, 
                        type=int)
    parser.add_argument("--decoder_mlp_ratio", 
                        help="MLP Projection Ratio in Decoder Transformers", 
                        default=4, 
                        type=int)
    parser.add_argument("--custom_weight_init", 
                        help="Do you want to initialize the model with truncated normal layers?", 
                        default=False, 
                        action=argparse.BooleanOptionalAction)
    
    ### EXTRA CONFIGS ###
    parser.add_argument("--log_wandb",
                        action=argparse.BooleanOptionalAction, 
                        default=False)

    parser.add_argument("--resume_from_checkpoint", 
                        help="Checkpoint folder for model to resume training from, inside the experiment folder", 
                        default=None, 
                        type=str)

    args = parser.parse_args()

    return args

### Grab Arguments ###
args = parse_args()

### Init Accelerator ###
path_to_experiment = os.path.join(args.working_directory, args.experiment_name)
accelerator = Accelerator(project_dir=path_to_experiment,
                          gradient_accumulation_steps=args.gradient_accumulation_steps,
                          log_with="wandb" if args.log_wandb else None)

### Weights and Biases Logger ###
if args.log_wandb:
    experiment_config = {"epochs": args.epochs,
                        "effective_batch_size": args.per_gpu_batch_size*accelerator.num_processes, 
                        "learning_rate": args.learning_rate,
                        "warmup_epochs": args.warmup_epochs,
                        "custom_weight_init": args.custom_weight_init}
    
    accelerator.init_trackers(args.experiment_name, config=experiment_config, init_kwargs={"wandb": {"name": args.wandb_run_name}})

### Load Model ###
mae_config = MAEConfig(img_size=args.img_size,
                       in_channels=args.input_channels,
                       encoder_embed_dim=args.encoder_embed_dim, 
                       encoder_depth=args.encoder_depth, 
                       encoder_num_heads=args.encoder_num_heads, 
                       encoder_mlp_ratio=args.encoder_mlp_ratio, 
                       decoder_embed_dim=args.decoder_embed_dim,
                       decoder_depth=args.decoder_depth, 
                       decoder_num_heads=args.decoder_num_heads, 
                       decoder_mlp_ratio=args.decoder_mlp_ratio)

model = ViTMAEForPreTraining(config=mae_config)

### Count Number of Parameters ###
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
accelerator.print("Number of Parameters:", params)

### Set Transforms for Training and Testing ###
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transforms = Compose([
    RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=InterpolationMode.BICUBIC),  
    RandomHorizontalFlip(),   
    ToTensor(),  
    Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)           
])

test_transforms = Compose([ 
    Resize((args.img_size, args.img_size)),
    ToTensor(),  
    Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)     
])

### Load Dataset ###
path_to_train_data = os.path.join(args.path_to_data, "train")
path_to_valid_data = os.path.join(args.path_to_data, "validation")
trainset = datasets.ImageFolder(path_to_train_data, transform=train_transforms)
testset = datasets.ImageFolder(path_to_valid_data, transform=test_transforms)


### Prep DataLoader with Custom Collate Function (No need on Validation only for Training) ###
mini_batchsize = args.per_gpu_batch_size // args.gradient_accumulation_steps 

trainloader = DataLoader(trainset, 
                         batch_size=mini_batchsize, 
                         shuffle=True, 
                         num_workers=args.num_workers, 
                         pin_memory=True)

testloader = DataLoader(testset, 
                        batch_size=mini_batchsize, 
                        shuffle=True, 
                        num_workers=args.num_workers, 
                        pin_memory=True)

### Define Optimizer (And seperate out weight decay and no weight decay parameters) ###
if (not args.bias_weight_decay) or (not args.norm_weight_decay):
    accelerator.print("Disabling Weight Decay on Some Parameters")
    weight_decay_params = []
    no_weight_decay_params = []
    for name, param in model.named_parameters():

        if param.requires_grad:
            
            ### Dont have Weight decay on any bias parameter (including norm) ###
            if "bias" in name and not args.bias_weight_decay:
                no_weight_decay_params.append(param)

            ### Dont have Weight Decay on any Norm scales params (weights) ###
            elif "bn" in name and not args.norm_weight_decay:
                no_weight_decay_params.append(param)

            else:
                weight_decay_params.append(param)

    optimizer_group = [
        {"params": weight_decay_params, "weight_decay": args.weight_decay},
        {"params": no_weight_decay_params, "weight_decay": 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_group, lr=args.learning_rate, betas=(0.9,0.95))

else:
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9,0.95))

### Define Scheduler (Compute number of training steps from epochs and adjust for num_gpus) ###
num_training_steps = len(trainloader) * args.epochs // args.gradient_accumulation_steps
num_warmup_steps = len(trainloader) * args.warmup_epochs // args.gradient_accumulation_steps
scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, 
                                            num_warmup_steps=num_warmup_steps, 
                                            num_training_steps=num_training_steps)

### Prepare Everything ###
model, optimizer, trainloader, testloader, scheduler = accelerator.prepare(
    model, optimizer, trainloader, testloader, scheduler
)
accelerator.register_for_checkpointing(scheduler)

### Check if we are resuming from checkpoint ###
if args.resume_from_checkpoint is not None:
    accelerator.print(f"Resuming from Checkpoint: {args.resume_from_checkpoint}")
    path_to_checkpoint = os.path.join(path_to_experiment, args.resume_from_checkpoint)
    accelerator.load_state(path_to_checkpoint)
    starting_checkpoint = int(args.resume_from_checkpoint.split("_")[-1])
else:
    starting_checkpoint = 0

for epoch in range(starting_checkpoint, args.epochs):
    
    accelerator.print(f"Training Epoch {epoch}")

    ### Storage for Everything ###
    train_loss = []
    test_loss = []
    accumulated_loss = 0 

    ### Training Progress Bar ###
    progress_bar = tqdm(range(len(trainloader)//args.gradient_accumulation_steps), 
                        disable=not accelerator.is_local_main_process)

    model.train()
    for images, _ in trainloader:

        ### Move Data to Correct GPU ###
        images = images.to(accelerator.device)
        
        with accelerator.accumulate(model):
            
            ### Pass Through Model ###
            encoded, decoded, logits, loss = model(images)

            ### Compute and Store Loss ##
            accumulated_loss += loss / args.gradient_accumulation_steps

            ### Compute Gradients ###
            accelerator.backward(loss)

            ### Clip Gradients ###
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            ### Update Model ###
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()


        ### Only when GPUs are being synchronized (When all the grad accumulation is done) store metrics ###
        if accelerator.sync_gradients:
            
            ### Gather Metrics Across GPUs ###
            loss_gathered = accelerator.gather_for_metrics(accumulated_loss)

            ### Store Current Iteration Error ###
            train_loss.append(torch.mean(loss_gathered).item())

            ### Reset Accumulated for next Accumulation ###
            accumulated_loss = 0

            ### Iterate Progress Bar ###
            progress_bar.update(1)


    model.eval()
    for images, _ in tqdm(testloader, disable=not accelerator.is_local_main_process):

        images = images.to(accelerator.device)

        with torch.no_grad():
            encoded, decoded, logits, loss = model(images)

        ### Gather across GPUs ###
        loss_gathered = accelerator.gather_for_metrics(loss)

        ### Store Current Iteration Error ###
        test_loss.append(torch.mean(loss_gathered).item())

    epoch_train_loss = np.mean(train_loss)
    epoch_test_loss = np.mean(test_loss)


    accelerator.print("Training Loss:", round(epoch_train_loss,3))
    
    accelerator.print("Testing Loss:", round(epoch_test_loss,3))

    ### Log with Weights and Biases ###
    if args.log_wandb:
        accelerator.log({"training_loss": epoch_train_loss,
                        "testing_loss": epoch_test_loss, 
                        "learning_rate": scheduler.get_last_lr()[0]}, step=epoch)
    
    ### Checkpoint Model ###
    if epoch % args.save_checkpoint_interval == 0:
        path_to_checkpoint = os.path.join(path_to_experiment, f"checkpoint_{epoch}")
        accelerator.save_state(output_dir=path_to_checkpoint)

### End Training for Trackers to Exit ###
accelerator.end_training()