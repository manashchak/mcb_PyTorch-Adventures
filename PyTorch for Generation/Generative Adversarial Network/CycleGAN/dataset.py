import os
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class Src2TgtDataset(Dataset):
    """
    Dataste class for style transfer between a folder of images
    from domain A (src) and another folder of images from doman B (tgt)

    These datasets are not a parallel corpus, we will train a model
    between unpaired images!
    """

    def __init__(self, 
                 path_to_src_dir, 
                 path_to_tgt_dir, 
                 transforms=None):
        
        self.path_to_src_imgs = [os.path.join(path_to_src_dir, file) \
                                    for file in os.listdir(path_to_src_dir)]
        self.path_to_tgt_imgs = [os.path.join(path_to_tgt_dir, file) \
                                    for file in os.listdir(path_to_tgt_dir)]

        self.transforms = transforms

    def __len__(self):

        ### We need to know which dataset (src vs tgt) as the most images which 
        ### will act as our total number of samples in the dataset

        return max(len(self.path_to_src_imgs), len(self.path_to_tgt_imgs))
    
    def __getitem__(self, idx):

        path_to_src_img = random.sample(self.path_to_src_imgs, k=1)[0]
        path_to_tgt_img = random.sample(self.path_to_tgt_imgs, k=1)[0]

        src = self.transforms(Image.open(path_to_src_img).convert("RGB"))
        tgt = self.transforms(Image.open(path_to_tgt_img).convert("RGB"))

        return src, tgt