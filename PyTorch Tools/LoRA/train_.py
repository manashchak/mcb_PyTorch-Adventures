import argparse
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from accelerate import Accelerator
from datasets import load_dataset
from tqdm import tqdm

from lora import LoRAModel
from transformers import AutoModelForImageClassification, AutoImageProcessor
from transformers import DefaultDataCollator


def main(args):

    accelerator = Accelerator()

    ### Load Dataset ###
    dataset = load_dataset(args.hf_dataset)
    labels = dataset["train"].features["label"].names

    ### Load Model and Convert to LoRA ###
    model = AutoModelForImageClassification.from_pretrained(args.hf_model_name, 
                                                            num_labels=len(labels), 
                                                            ignore_mismatched_sizes=True)
    lora_model = LoRAModel(model=model, 
                        rank=args.rank, 
                        lora_alpha=args.lora_alpha, 
                        use_rslora=args.use_rslora, 
                        target_modules=args.target_modules,
                        exclude_modules=args.exclude_modules,
                        lora_dropout=args.lora_dropout)


    ### Load Image Processor ###
    processor = AutoImageProcessor.from_pretrained(args.hf_model_name)

    ### Apply Processor to Dataset ###
    def transforms(examples):
        examples["pixel_values"] = [processor(img.convert("RGB"))["pixel_values"][0] for img in examples["image"]]
        del examples["image"]
        return examples
    
    dataset = dataset.with_transform(transforms)

    ### Create DataLoader ###
    collate_fn = DefaultDataCollator()
    trainloader = DataLoader(dataset["train"], batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True, num_workers=4)
    testloader = DataLoader(dataset["validation"], batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False, num_workers=4)

    ### Define Optimizer ###
    params_to_train = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(params_to_train, lr=args.lr)

    ### Define Loss Function ###
    loss_fn = nn.CrossEntropyLoss()

    ### Start Training ###
    for epoch in range(args.epochs):

        for batch in tqdm(trainloader):
            
            ### Compute Logits from Model ###
            logits = lora_model(batch["pixel_values"])["logits"]
            
            break
        break





if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="LoRA ViT FineTuning Script")

    parser.add_argument("--hf_dataset",
                        help="Path to dataset (we use ImageFolder)",
                        required=True, 
                        type=str)

    parser.add_argument("--hf_model_name",
                        help="What Image model do you want to use from Huggingface?",
                        default="google/vit-base-patch16-224",
                        type=str)

    parser.add_argument("--target_modules",
                        nargs="+",
                        help="What modules do you want to apply LoRA to? if None then all layers get LoRA",
                        default=None, 
                        type=str)

    parser.add_argument("--exclude_modules",
                        nargs="+",
                        help="What modules do you not want to apply LoRA to",
                        default="classifier",
                        type=str)

    parser.add_argument("--rank",
                        help="What rank do you want to use for LoRA?",
                        default=16,  
                        type=int)

    parser.add_argument("--lora_alpha",
                        help="What LoRA Alpha do you want to use?",
                        default=16,
                        type=int)

    parser.add_argument("--use_rslora",
                        help="Use Rank Stabilized LoRA Scaling as described in https://arxiv.org/abs/2312.03732",
                        action=argparse.BooleanOptionalAction)

    parser.add_argument("--lora_dropout",
                        help="What dropout do you want on LoRA layers",
                        default=0,
                        type=float)

    parser.add_argument("--batch_size",
                        default=2,
                        type=int)

    parser.add_argument("--lr",
                        default=0.001, 
                        type=float)
    
    parser.add_argument("--epochs",
                        default=10, 
                        type=int)


    args = parser.parse_args()

    main(args)
# def train(model, device, epochs, optimizer,
#           scheduler, loss_fn, trainloader,
#           valloader, savepath="ViTDogsvCatsSmall.pt"):
#     log_training = {"epoch": [],
#                     "training_loss": [],
#                     "training_acc": [],
#                     "validation_loss": [],
#                     "validation_acc": []}

#     best_val_loss = np.inf
#     for epoch in range(1, epochs + 1):
#         print(f"Starting Epoch {epoch}")
#         training_losses, training_accuracies = [], []
#         validation_losses, validation_accuracies = [], []

#         for image, label in tqdm(trainloader):
#             image, label = image.to(device), label.to(device)
#             optimizer.zero_grad(set_to_none=True)
#             out = model.forward(image)["logits"]
         
#             ### CALCULATE LOSS ##
#             loss = loss_fn(out, label)
#             training_losses.append(loss.item())

#             ### CALCULATE ACCURACY ###
#             predictions = torch.argmax(out, axis=1)
#             accuracy = (predictions == label).sum() / len(predictions)
#             training_accuracies.append(accuracy.item())

#             loss.backward()
#             optimizer.step()
#             scheduler.step()

#         for image, label in tqdm(valloader):
#             image, label = image.to(device), label.to(device)
#             with torch.no_grad():
#                 out = model.forward(image)["logits"]

#                 ### CALCULATE LOSS ##
#                 loss = loss_fn(out, label)
#                 validation_losses.append(loss.item())

#                 ### CALCULATE ACCURACY ###
#                 predictions = torch.argmax(out, axis=1)
#                 accuracy = (predictions == label).sum() / len(predictions)
#                 validation_accuracies.append(accuracy.item())

#         training_loss_mean, training_acc_mean = np.mean(training_losses), np.mean(training_accuracies)
#         valid_loss_mean, valid_acc_mean = np.mean(validation_losses), np.mean(validation_accuracies)

#         ### Save Model If Val Loss Decreases ###
#         if valid_loss_mean < best_val_loss:
#             print("---Saving Model---")
#             torch.save(model.state_dict(), savepath)
#             best_val_loss = valid_loss_mean

#         log_training["epoch"].append(epoch)
#         log_training["training_loss"].append(training_loss_mean)
#         log_training["training_acc"].append(training_acc_mean)
#         log_training["validation_loss"].append(valid_loss_mean)
#         log_training["validation_acc"].append(valid_acc_mean)


#         print("Training Loss:", training_loss_mean)
#         print("Training Acc:", training_acc_mean)
#         print("Validation Loss:", valid_loss_mean)
#         print("Validation Acc:", valid_acc_mean)

#     return log_training, model

# ### SETUP DATASET ###
# PATH_TO_DATA = "../../data/dogsvscats"
# dataset = ImageFolder(PATH_TO_DATA)

# normalizer = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
# train_transforms = Compose([
#     Resize((224, 224)),
#     RandomHorizontalFlip(),
#     ToTensor(),
#     normalizer])

# val_transforms = Compose([
#     Resize((224, 224)),
#     ToTensor(),
#     normalizer])

# train_samples, test_samples = int(0.9 * len(dataset)), len(dataset) - int(0.9 * len(dataset))
# train_dataset, val_dataset = torch.utils.data.random_split(dataset, lengths=[train_samples, test_samples])

# train_dataset.dataset.transform = train_transforms
# val_dataset.dataset.transform = val_transforms

# ### SETUP TRAINING LOOP ###
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# print("Training on Device {}".format(DEVICE))

# ### Define Optimizer ###
# params_to_train = filter(lambda p: p.requires_grad, model.parameters())
# optimizer = optim.AdamW(params=params_to_train, lr=1e-3)

# ### Define Loss ###
# loss_fn = nn.CrossEntropyLoss()

# ### Build DataLoaders ###
# batch_size = 128
# trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
# valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# ### Define Scheduler ###
# EPOCHS = 2
# scheduler = get_cosine_schedule_with_warmup(optimizer, 
#                                             num_warmup_steps=250, 
#                                             num_training_steps=EPOCHS*len(trainloader))

# logs, model = train(model=model.to(DEVICE),
#                     device=DEVICE,
#                     epochs=EPOCHS,
#                     optimizer=optimizer,
#                     scheduler=scheduler,
#                     loss_fn=loss_fn,
#                     trainloader=trainloader,
#                     valloader=valloader)