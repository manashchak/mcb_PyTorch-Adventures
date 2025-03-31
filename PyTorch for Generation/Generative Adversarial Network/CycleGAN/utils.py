import torch 
import random 
from PIL import Image

def load_testing_samples(path_to_samples, 
                         transforms,
                         k=5):

    samples = random.sample(path_to_samples, k=k)

    samples = [
        transforms(Image.open(s)).unsqueeze(0) for s in samples
    ]

    samples = torch.cat(samples)

    return samples


    