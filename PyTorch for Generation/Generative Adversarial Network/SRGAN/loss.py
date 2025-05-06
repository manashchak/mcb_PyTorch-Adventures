import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vgg import vgg16, VGG16_Weights

class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        
        ### Load VGG Model ###
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

        ### Dont Update VGG Parameters ###
        for param in vgg.parameters():
            param.requires_grad = False

        ### Keep Only Feature Extractor Layers ###
        self.vgg_features = vgg.features.eval()

        ### ImageNet Mean/Std ###
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1))

    def forward(self, real_imgs, fake_imgs):
        
        ### Convert real/fake imgs from [-1 to 1] to [0,1] ###
        real_imgs = (real_imgs * 0.5) + 0.5
        fake_imgs = (fake_imgs * 0.5) + 0.5 

        ### Normalize w/ ImageNet Mean/Std ###
        real_imgs = (real_imgs - self.mean) / self.std
        fake_imgs = (fake_imgs - self.mean) / self.std

        ### Resize to VGG Resolution of (224 x 224) ###
        real_imgs = F.interpolate(real_imgs, mode="bilinear", size=(224,224), align_corners=False)
        fake_imgs = F.interpolate(fake_imgs, mode="bilinear", size=(224,224), align_corners=False)

        ### Get VGG Features from Real and Fake Images ###
        with torch.no_grad():
            real_feautures = self.vgg_features(real_imgs)

        fake_features = self.vgg_features(fake_imgs)

        ### Minimize MSE Loss between Features ###
        return torch.mean((real_feautures - fake_features)**2)
    