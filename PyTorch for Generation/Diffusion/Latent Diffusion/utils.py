import os
import torch
import numpy as np
from PIL import Image
from dataset import image_transforms

def load_val_images(path_to_image_folder, 
                    img_size,
                    device, 
                    dtype):

    image_files = os.listdir(path_to_image_folder)
    path_to_imgs = [os.path.join(path_to_image_folder, file) for file in image_files]
    
    val_img_transforms = image_transforms(img_size=img_size, 
                                          train=False)
    
    val_images = [Image.open(path).convert("RGB") for path in path_to_imgs]
    val_images = torch.stack([val_img_transforms(img) for img in val_images])

    val_images = val_images.to(device)

    # Cast images to the correct precision type
    weight_dtype = torch.float32 
    if dtype == "fp16":
        weight_dtype = torch.float16
    elif dtype == "bf16":
        weight_dtype = torch.bfloat16
    
    val_images = val_images.to(weight_dtype)

    return val_images, image_files

def save_generated_images(image_tensors, 
                          path_to_save_folder, 
                          step,
                          image_files):
    

    ### Clamp Output Between [-1 to 1] ###
    image_tensors = torch.clamp(image_tensors, -1., 1.)

    ### Denormalize Images ([-1,1] -> [0,1]) ###
    image_tensors = (image_tensors + 1) / 2

    ### Transpose ###
    image_tensors = image_tensors.cpu().permute(0,2,3,1).numpy()

    ### Convert to [0, 255] ###
    image_tensors = (255 * image_tensors).astype(np.uint8)

    ### Convert to PIL Image ###
    images = [Image.fromarray(img).convert("RGB") for img in image_tensors]

    ### Store Images in New Directory ###
    path_to_newdir = os.path.join(path_to_save_folder, f"iteration_{step}")
    os.makedirs(path_to_newdir, exist_ok=True)

    ### Loop through images ###
    for image, file_name in zip(images, image_files):
        path_to_image = os.path.join(path_to_newdir, file_name)
        image.save(path_to_image)
        

        
