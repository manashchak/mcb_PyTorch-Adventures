import torch
import torch.nn as nn
from torchvision.models.vgg import vgg16, VGG16_Weights

class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        
        ### Load VGG Model ###
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        
        ### Keep Only Feature Extractor Layers ###
        self.vgg_features = vgg.features.eval()

    def forward(self, real_imgs, fake_imgs):

        ### Get VGG Features from Real and Fake Images ###
        real_feautures = self.vgg_features(real_imgs)
        fake_features = self.vgg_features(fake_imgs)

        ### Minimize MSE Loss between Features ###
        return torch.mean((real_feautures - fake_features)**2)
    