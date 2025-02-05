import os
import io
from PIL import Image
from datasets import load_dataset
from torchvision.transforms import Compose, RandomResizedCrop, RandomHorizontalFlip, ToTensor

parquet_files = [file for file in os.listdir("/mnt/datadrive/data/ConceptualCaptions/val") if ".parquet" in file]
path_to_parquet_files = [os.path.join("/mnt/datadrive/data/ConceptualCaptions/val", file) for file in parquet_files]


dataset = load_dataset("parquet", data_files={'train': path_to_parquet_files}, num_proc=32)
dataset = dataset.filter(lambda example: example["status"] == "success", num_proc=32)

cols_to_rm = [col for col in dataset.column_names["train"] if col not in ["caption", "jpg"]]
dataset = dataset.remove_columns(cols_to_rm)

def bin2pil(examples):

    images = []
    for image in examples["jpg"]:
        image = Image.open(io.BytesIO(image))
        
        images.append(image)

    examples["image"] = images

    return examples

dataset = dataset.map(bin2pil, batched=True, batch_size=100, num_proc=32)
dataset = dataset.remove_columns("jpg")

train_transforms = Compose([
    RandomResizedCrop(384, scale=(0.2, 1.0)),  
    RandomHorizontalFlip(),   
    ToTensor(),  
])

def transforms(examples):
    examples["pixel_values"] = [train_transforms(image.convert("RGB")) for image in examples["image"]]
    del examples["image"]
    return examples

dataset.set_transform(transforms)

from torch.utils.data import DataLoader
from tqdm import tqdm

loader = DataLoader(dataset["train"], batch_size=128, num_workers=32, shuffle=True)

for batch in tqdm(loader):
    print(batch["pixel_values"].shape)