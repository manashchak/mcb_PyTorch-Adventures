import os
from datasets import load_dataset
from io import BytesIO
from PIL import Image
import requests
import argparse
import warnings
warnings.simplefilter("error")


def download_image(example):

    images = []

    for img_link in example["img_link"]:
        try:

            response = requests.get(img_link, timeout=2)
            response.raise_for_status()

            image = Image.open(BytesIO(response.content))

        except:
     
            image = None

        images.append(image)

    example["image"] = images

    return example 

def build_hf_dataset(path_to_train_tsv, path_to_val_tsv, path_to_store, sample_pct=1.0, num_workers=32):

    dataset = load_dataset("csv", 
                           data_files={"train": path_to_train_tsv, "test": path_to_val_tsv}, 
                           delimiter='\t',
                           column_names=["text", "img_link"], 
                           header=None,
                           cache_dir=path_to_store)

    if sample_pct != 1.0:
        num_train_samples = len(dataset["train"])
        samples_to_keep = int(num_train_samples * sample_pct)
        
        print(f"Keeping {sample_pct*100}% of the Training Data: {samples_to_keep}/{num_train_samples}")
        dataset["train"] = dataset["train"].shuffle().select(range(samples_to_keep))
    
    print(dataset)

    dataset = dataset.map(download_image, batched=True, batch_size=100, num_proc=num_workers)
    dataset = dataset.filter(lambda x: x["image"] is not None, num_proc=num_workers)
    
    dataset.save_to_disk(path_to_store)
    dataset.cleanup_cache_files()

    
    


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
    
    parser.add_argument("--batch_size",
                        help="Number of samples to download per thread",
                        default=100, 
                        required=False, 
                        type=int)
    
    parser.add_argument("--num_workers", 
                        help="Path to Train and Validation tsv downloads",
                        default=32,
                        required=False,
                        type=int)
    
    args = parser.parse_args()

    build_hf_dataset(path_to_train_tsv=os.path.join(args.path_to_root, args.train_tsv_name),
                     path_to_val_tsv=os.path.join(args.path_to_root, args.val_tsv_name),
                     path_to_store=args.path_to_store,
                     sample_pct=args.sample_pct,
                     num_workers=args.num_workers)

    from datasets import load_from_disk
    dataset = load_from_disk(args.path_to_store)
    
    print(dataset)
        
    
