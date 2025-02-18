import os
import json
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import AutoTokenizer
from datasets import load_from_disk

from datasets import disable_caching
disable_caching()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def image_transforms(num_channels=3, 
                     img_size=256, 
                     random_resize=True, 
                     interpolation="bilinear",
                     random_flip_p=0,
                     train=True):

    interpolation_dict = {"nearest": InterpolationMode.NEAREST, 
                          "bilinear": InterpolationMode.BILINEAR,
                          "bicubic": InterpolationMode.BICUBIC}
    

    if random_resize and train: 
        resize = transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0), interpolation=interpolation_dict[interpolation])
    else:
        resize = transforms.Resize((img_size, img_size))

    if not train:
        random_flip_p = 0
    
    image2tensor = transforms.Compose([
                        transforms.Lambda(lambda img: img.convert("RGB") if num_channels == 3 else img),
                        resize,
                        transforms.RandomHorizontalFlip(p=random_flip_p),
                        transforms.ToTensor(), 
                        transforms.Normalize([0.5 for _ in range(num_channels)], 
                                            [0.5 for _ in range(num_channels)]),
                        
                    ])
    
    return image2tensor

class GenericImageDataset(Dataset):

    """
    Generic Image Dataset

    Args:
        - path_to_data: Points to a folder full of images of faces 
        - nested: Does that path_to_data contain folders, in which there are images
    
    """

    def __init__(self, path_to_data, nested=False, transform=None):
        self.transforms = transform

        if not nested:
            self.path_to_files = [os.path.join(path_to_data, file) for file in os.listdir(path_to_data)]
        else:
            self.path_to_files = []
            for dir in os.listdir(path_to_data):
                path_to_dir = os.path.join(path_to_data, dir)
                self.path_to_files.extend([os.path.join(path_to_dir, file) for file in os.listdir(path_to_dir)])
    
    def __len__(self):
        return len(self.path_to_files)
    
    def __getitem__(self, idx):
        img_path = self.path_to_files[idx]
        img = Image.open(img_path)
        img = self.transforms(img)

        return {"images": img}    

def conceptual_captions_collate_fn(tokenizer_model="openai/clip-vit-large-patch14", return_transcript=True):
    pass


def conceptual_captions(path_to_data, transforms, return_caption):

    dataset = load_from_disk(path_to_data)

    def img_transforms(batch):

        transformed_images = [
            transforms(image) for image in batch["image"]
        ]

        batch["images"] = transformed_images

        batch.pop("image")

        return batch

    dataset.set_transform(img_transforms)

    if not return_caption:
        if "encoded_text" in dataset.column_names["train"]:
            dataset = dataset.remove_columns("encoded_text")
        if "caption" in dataset.column_names["train"]:
            dataset = dataset.remove_columns("caption")

    return dataset["train"]

def get_dataset(dataset,
                path_to_data, 
                num_channels=3, 
                img_size=256, 
                random_resize=True, 
                interpolation="bilinear",
                random_flip_p=0.5,
                train=True,
                return_caption=True):
    
    img_transform = image_transforms(num_channels=num_channels,
                                     img_size=img_size, 
                                     random_resize=random_resize, 
                                     interpolation=interpolation, 
                                     random_flip_p=random_flip_p, 
                                     train=train)
    if dataset == "celeba":

        if return_caption:
            raise Exception("CelebA Has No Captions!")
        
        trainset = GenericImageDataset(path_to_data=path_to_data, 
                                       transform=img_transform)
         
    elif dataset == "celebahq":

        if return_caption:
            raise Exception("CelebAHQ Has No Captions!")
        
        trainset = GenericImageDataset(path_to_data=path_to_data, 
                                       transform=img_transform)
        
    elif dataset == "ffhd":
    
        if return_caption:
            raise Exception("FFHQ Has No Caption")
        
        trainset = GenericImageDataset(path_to_data=path_to_data, 
                                       transform=img_transform,
                                       nested=True)
        
    elif dataset == "imagenet": 
        if return_caption:
            raise Exception("Imagenet Has No Captions!")

        trainset = GenericImageDataset(path_to_data=path_to_data, 
                                       transform=img_transform, 
                                       nested=True)
        
    elif dataset == "birds":
        if return_caption:
            raise Exception("CUB Birds has no captions!")

        trainset = GenericImageDataset(path_to_data=path_to_data, 
                                       transform=img_transform, 
                                       nested=True)
        
    elif dataset == "conceptual_captions":

        trainset = conceptual_captions(path_to_data, 
                                       img_transform, 
                                       return_caption=return_caption)

    return trainset

if __name__ == "__main__":

    path_to_celeb = "/mnt/datadrive/data/CelebA/img_align_celeba/img_align_celeba/"
    path_to_celebhq = "/mnt/datadrive/data/CelebAMask-HQ/CelebA-HQ-img/"
    path_to_imagenet = "/mnt/datadrive/data/ImageNet/train/"
    path_to_conceptual = "/mnt/datadrive/data/ConceptualCaptions/hf_train"
    path_to_coco = "/mnt/datadrive/data/coco2017/"
    path_to_birds = "/mnt/datadrive/data/birds/bird_images/images"
    path_to_ffhq = "/mnt/datadrive/data/ffhd/images1024x1024"

    loader = get_dataset(dataset="conceptual_caption", 
                         path_to_data=path_to_conceptual,
                         return_caption=False)
