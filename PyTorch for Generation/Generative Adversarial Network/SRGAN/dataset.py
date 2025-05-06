from torch.utils.data import Dataset
from torchvision import datasets, transforms

class SRImageDataset(Dataset):
    def __init__(self, root_dir, hr_size=96, scale_factor=4):
        self.dataset = datasets.ImageFolder(root_dir)
        lr_size = hr_size // scale_factor

        self.hr_transform = transforms.Compose([
            transforms.Resize((hr_size, hr_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.lr_transform = transforms.Compose([
            transforms.Resize((lr_size,lr_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        hr_img = self.hr_transform(img)
        lr_img = self.lr_transform(img)

        return hr_img, lr_img