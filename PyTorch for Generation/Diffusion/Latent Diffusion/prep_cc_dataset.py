import os
from datasets import load_dataset
from io import BytesIO
from PIL import Image
import requests
import argparse


def download_image(example):

    images = []

    for img_link in example["img_link"]:
        try:

            response = requests.get(img_link, timeout=10)
            response.raise_for_status()

            image = Image.open(BytesIO(response.content))

        except:
            
            image = None

        images.append(image)

    example["image"] = images

    return example 

def build_hf_dataset(path_to_tsv, path_to_store, num_workers=32):

    dataset = load_dataset("csv", 
                           data_files=path_to_tsv, 
                           delimiter='\t',
                           column_names=["text", "img_link"], 
                           header=None,
                           cache_dir=path_to_store)
    
    dataset = dataset.map(download_image, batched=True, batch_size=100, num_proc=num_workers)
    dataset = dataset.filter(lambda x: x["image"] is not None, num_proc=num_workers)
    
    dataset.save_to_disk(path_to_store)

    
    


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Download and Prepare ConceptualCaption-3M")
    
    parser.add_argument("--path_to_root", 
                        help="Path to Train and Validation tsv downloads",
                        required=True,
                        type=str)
    
    parser.add_argument("--path_to_store",
                        help="Will create a train/validation folder at this directory",
                        required=True,
                        type=str)
    
    parser.add_argument("--sample_pct",
                        help="If you don't want all 3 Million images, random sample some proportion",
                        default=1.0, 
                        required=False, 
                        type=float)
                        
    
    parser.add_argument("--train_tsv_name",
                        help="Name of the training tsv file",
                        default="train.tsv", 
                        required=False, 
                        type=str)
    
    parser.add_argument("--val_tsv_name",
                        help="Name of the validation tsv file",
                        default="validation.tsv", 
                        required=False, 
                        type=str)
    
    parser.add_argument("--num_workers", 
                        help="Path to Train and Validation tsv downloads",
                        default=32,
                        type=int)
    
    args = parser.parse_args()

    build_hf_dataset(path_to_tsv=os.path.join(args.path_to_root, args.val_tsv_name),
                     path_to_store=args.path_to_store,
                     num_workers=args.num_workers)

    from datasets import load_from_disk
    dataset = load_from_disk(args.path_to_store)
    
    idx = 125
    dataset["train"][idx]["image"].show()
    print(dataset["train"][idx]["text"])
        
    
