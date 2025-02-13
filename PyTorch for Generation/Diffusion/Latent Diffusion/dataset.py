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
                     random_flip_p=0.5,
                     train=True):

    interpolation_dict = {"nearest": InterpolationMode.NEAREST, 
                          "bilinear": InterpolationMode.BILINEAR,
                          "bicubic": InterpolationMode.BICUBIC}
    

    if random_resize and train: 
        resize = transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0), interpolation=interpolation_dict[interpolation])
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

class COCODataset(Dataset):
    def __init__(self, path_to_data, train=True, transforms=None):

        self.path_to_root = path_to_data
        
        if train:
            path_to_annotations = os.path.join(self.path_to_root, "annotations", "captions_train2017.json")
            self.path_to_images = os.path.join(self.path_to_root, "train2017")

        else:
            path_to_annotations = os.path.join(self.path_to_root, "annotations", "captions_val2017.json")
            self.path_to_images = os.path.join(self.path_to_root, "val2017")


        self._prepare_annotations(path_to_annotations)

        self.image2tensor = transforms
            
    def _prepare_annotations(self, path_to_annotations):

        ### Load Annotation Json ###
        with open(path_to_annotations, "r") as f:
            annotation_json = json.load(f)
            
        ### For Each Image ID Get the Corresponding Annotations ###
        id_annotations = {}
    
        for annot in annotation_json["annotations"]:
            image_id = annot["image_id"]
            caption = annot["caption"]
    
            if image_id not in id_annotations:
                id_annotations[image_id] = [caption]
            else:
                id_annotations[image_id].append(caption)
    
        ### Coorespond Image Id to Filename ###
        path_id_coorespondance = {}
    
        for image in annotation_json["images"]:
            file_name = image["file_name"]
            image_id = image["id"]
        
            path_id_coorespondance[file_name] = id_annotations[image_id]

        self.filenames = list(path_id_coorespondance.keys())
        self.annotations = path_id_coorespondance


    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):

        ### Grab Filename and Cooresponding Annotations ###
        filename = self.filenames[idx]
        annotation = self.annotations[filename]

        ### If more than 1 annotation, randomly select ###
        annotation = random.choice(annotation)

        ### Remove Any Whitespace from Text ###
        annotation = annotation.strip()

        ### Error Handling on any Broken Images ###
        try:
            ### Load Image ###
            path_to_img = os.path.join(self.path_to_images, filename)
            img = Image.open(path_to_img).convert("RGB")

            ### Apply Image Transforms ###
            img = self.image2tensor(img)
            
            return img, annotation

        except Exception as e:
            print("Exception:", e)
            return None, None

def coco_collate_fn(tokenizer_model, return_transcript):

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)

    def collate_fn(examples):
        
        ### Remove any potential Nones ###
        examples = [i for i in examples if i[0] is not None]

        if len(examples) > 0:

            ### Grab Images and add batch dimension ###
            images = [i[0].unsqueeze(0) for i in examples]

            ### Grab Text Annotations ###
            annot = [i[1] for i in examples]

            ### Stick All Images Together along Batch ###
            images = torch.concatenate(images)

            ### Store Batch as Dictionary ###
            batch = {"images": images}

            if return_transcript:

                ### Tokenize Annotations with Padding ###
                annotation = tokenizer(annot, padding=True, return_tensors="pt")
                batch["context"] = annotation["input_ids"]
                batch["attention_mask"] = annotation["attention_mask"].bool()

            return batch
        
        else:
            print("Broken Batch!")
            return None
        
    return collate_fn

def imagenet_dataset(path_to_data, transforms=None):

    path_to_train_data = os.path.join(path_to_data, "train")

    dataset = dataset.ImageFolder(path_to_train_data, transform=transforms)

    return dataset

def conceptual_captions(path_to_data, transforms=None):

    dataset = load_from_disk(path_to_data)

    return dataset

def conceptual_captions_collate_fn(img_transforms, return_captions=True):

    def collate_fn(batch):

        images, encoded_texts, attn_mask = [], [], []
        for sample in batch:
            images.append(img_transforms(sample["image"]))

            encoded_text = torch.tensor(sample["encoded_text"])
            encoded_texts.append(encoded_text)
            attn_mask.append(torch.ones(encoded_text.shape[0]))

        images = torch.stack(images)
        
        batch = {"images": images}

        if return_captions:
            encoded_texts = torch.nn.utils.rnn.pad_sequence(encoded_texts, batch_first=True)
            attn_mask = torch.nn.utils.rnn.pad_sequence(attn_mask, batch_first=True).bool()
            
            batch['context'] = encoded_text
            batch["attention_mask"] = attn_mask

        return batch

    return collate_fn

def get_dataset(dataset,
                batch_size, 
                path_to_data, 
                num_channels=3, 
                img_size=256, 
                random_resize=True, 
                interpolation="bilinear",
                random_flip_p=0.5,
                train=True,
                return_caption=True,
                num_workers=8,
                pin_memory=True,
                tokenizer_model="openai/clip-vit-large-patch14"):
    
    img_transform = image_transforms(num_channels=num_channels,
                                     img_size=img_size, 
                                     random_resize=random_resize, 
                                     interpolation=interpolation, 
                                     random_flip_p=random_flip_p, 
                                     train=train)
    if dataset == "celeba":

        if return_caption:
            raise Exception("CelebA Has No Captions!")
        
        dataset = GenericImageDataset(path_to_data=path_to_data, 
                                      transform=img_transform)
        
        loader = DataLoader(dataset, 
                            batch_size=batch_size,
                            num_workers=num_workers,
                            pin_memory=pin_memory,
                            shuffle=True)
        
    if dataset == "celebahq":

        if return_caption:
            raise Exception("CelebAHQ Has No Captions!")
        
        dataset = GenericImageDataset(path_to_data=path_to_data, 
                                      transform=img_transform)
        
        loader = DataLoader(dataset, 
                            batch_size=batch_size,
                            num_workers=num_workers,
                            pin_memory=pin_memory,
                            shuffle=True)
        
    elif dataset == "imagenet": 
        if return_caption:
            raise Exception("Imagenet Has No Captions!")

        dataset = GenericImageDataset(path_to_data=path_to_data, 
                                      transform=img_transform, 
                                      nested=True)
        
        loader = DataLoader(dataset, 
                            batch_size=batch_size,
                            num_workers=num_workers,
                            pin_memory=pin_memory,
                            shuffle=True)
    
    elif dataset == "coco":

        dataset = COCODataset(path_to_data=path_to_data, 
                              transforms=img_transform)
        
        collate_fn = coco_collate_fn(tokenizer_model, 
                                     return_transcript=return_caption)
        
        loader = DataLoader(dataset, 
                            batch_size=batch_size,
                            num_workers=num_workers,
                            pin_memory=pin_memory,
                            collate_fn=collate_fn,
                            shuffle=True)
        
    elif dataset == "conceptual_caption":

        dataset = conceptual_captions(path_to_data=path_to_data, 
                                      transforms=img_transform)

        collate_fn = conceptual_captions_collate_fn(img_transforms=img_transform)

        loader = DataLoader(dataset["train"],
                            batch_size=batch_size, 
                            num_workers=num_workers,
                            collate_fn=collate_fn, 
                            pin_memory=pin_memory,
                            shuffle=True)
        
    return loader

if __name__ == "__main__":

    path_to_celeb = "/mnt/datadrive/data/CelebA/img_align_celeba/img_align_celeba/"
    path_to_celebhq = "/mnt/datadrive/data/CelebAMask-HQ/CelebA-HQ-img/"
    path_to_imagenet = "/mnt/datadrive/data/ImageNet/train/"
    path_to_conceptual = "/mnt/datadrive/data/ConceptualCaptions/hf_train"
    path_to_coco = "/mnt/datadrive/data/coco2017/"

    loader = get_dataset(dataset="conceptual_caption", 
                         batch_size=64, 
                         num_workers=32,
                         path_to_data=path_to_conceptual)
