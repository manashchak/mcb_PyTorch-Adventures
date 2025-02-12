"""
UperNet Segmentation

Code heavily inspired by:
https://github.com/yassouali/pytorch-segmentation/tree/master

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PSPModule(nn.Module):

    """
    Pyramid Pooling Module

    To the last layer of features extracted from a model, 
    pool the feature at different resolutions, set by the
    bin_sizes. By doing this, we learn different granularities
    of features that we can aggregate together. We can take
    smaller resolution stages, interpolate up to the largest
    resolution, and concatenate to our original features
    as a block of multilevel information. 

    Args:
        in_channels: Number of input channels to our PSPModule 
                     and also will be the number of output channels
        bin_sizes: List of Pooling resolutions at every layer

    """

    def __init__(self, in_channels, bin_sizes=[1,2,4,6]):
        super(PSPModule, self).__init__()

        self.in_channels = in_channels
        self.bin_sizes = bin_sizes

        ### Compute Output Channels for Each Stage ###
        ### so when we concat the stages we will have the ###
        ### same number of channels again ###
        out_channels = in_channels // len(bin_sizes)

        ### Build Block for every bin_size ###
        self.stages = nn.ModuleList()
        for s in bin_sizes:
            self.stages.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool1d(s),
                    nn.Conv2d(in_channels, 
                              out_channels, 
                              kernel_size=1, 
                              bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )

        ### Create our BottleNeck that processes the tensor of ###
        ### our stages concatenated to the original features ###

        ### original_features: in_channels
        ### stages: out_channels
        ### total_channels: in_channels + len(bin_size) * out_channels

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + len(bin_sizes) * out_channels, 
                      in_channels, 
                      kernel_size=3, 
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(in_channels), 
            nn.ReLU(inplace=True)
        )

    def forward(self, features):

        batch, channels, height, width = features.shape

        ### Create Pyramids (starting with our original features) ###
        pyramid = [features]

        ### Loop Through the Stages ###
        for stage in self.stages:

            ### Use Stage to Pool to Bin Size ###
            pooled_features = stage(features)

            ### Upsample Back To Image Size ###
            upsampled_features = F.interpolate(pooled_features, 
                                               size=(height, width),
                                               mode="bilinear",
                                               align_corners=True)
            
            ### Append to Pyramid ###
            pyramid.append(upsampled_features)

        ### Concatenate together all tensors in the Pyramid ###
        pyramid = torch.cat(pyramid, dim=1)

        ### Convolution to Merge Feature Information Together ###
        output = self.bottleneck(pyramid)

        return output
    
class FPN(nn.Module):

    """
    Feature Pyramid Network 

    Given a stack of features (from different parts of a vision model) 
    we map all the channels of the feature to that of the earliest layer. 
    We then add consecutive layers together, upsampling the low resolution
    features to match the higher level if necessary. To these accumulated
    features we concatenate back on our lowest level, most granular features 
    again onto our stack. We can interpolate all of these features up to 
    the higest resolution output (the first layer). Lastly concatenate 
    and use a convolution to fuse together all the information.

    Caveat: Because we are using a ViT/MAE as our encoder, all the features
            at every level will be the same resolution. We will have to do 
            some extra work to construct a feature pyramid from this using 
            some Transpose Convolutions 
    
    Args:
         - How many channels are there in every feature we input
    """

    def __init__(self, feature_channels):

        super(FPN, self).__init__()

        ### Store the Channels per Feature we Input ###
        self.feature_channels = feature_channels

        ### our model will output the number of channels of the ###
        ### highest resolution output (the first feature in the list) ###
        self.out_channels = self.feature_channels[0]

        ### Lets say we have 4 layers like the following (from resnet) ###
        ### (B x 64 x 64 x 64)
        ### (B x 128 x 32 x 32)
        ### (B x 256 x 16 x 16)
        ### (B x 512 x 16 x 16)

        ### We will first conver the last three layer to the number of the ###
        ### channels of the input layer. To do this we will use a convolution ###
        ### (B x 64 x 64 x 64)
        ### (B x 128 x 32 x 32) -> (B x 64 x 32 x 32)
        ### (B x 256 x 16 x 16) -> (B x 64 x 16 x 16)
        ### (B x 512 x 16 x 16) -> (B x 64 x 16 x 16)
        self.conv_proj = nn.ModuleList()
        for channels in feature_channels[1:]:
            self.conv_proj.append(nn.Conv2d(channels, self.out_channels, kernel_size=1))

        ### After we add together consecutive layers (from the lowest resolution up) ###
        ### We want to pass that through a convolution to learn the features of the ###
        ### Merged information. 
        ### Lets give our layers some names
        ### 0: (B x 64 x 64 x 64)
        ### 1: (B x 64 x 32 x 32)
        ### 2: (B x 64 x 16 x 16)
        ### 3: (B x 64 x 16 x 16)

        ### Our first layer will be to add (3) to (2)
        ### Our second layer will be to add (2) to (1), but by upsampling (2) to the dimension of (1)
        ### Our third layer will be to add (1) to (0), but by upsampling (1) to the dimension of (0)
        ### On these three layers we will perform a convolution
        self.smooth_conv = nn.ModuleList()
        for _ in range(len(feature_channels)-1):
            self.smooth_conv.append(nn.Conv2d(self.out_channels, 
                                              self.out_channels, 
                                              kernel_size=3, 
                                              stride=1, padding="same"))
            
        ### Notice that originally, our stack when (0, 1, 2, 3), but now we have (3+2, 2+1, 1+0). We need to flip
        ### this around so we have it from early parts of the model to deeper parts again: (1+0, 2+1, 3+2). After
        ### we do that we will concatenate on our final features again to preserve the coarsest feature extracted
        ### by our image encoder. Therefore we will have a tuple like: (3+2, 2+1, 1+0, features[-1])
        ### with these features, we can concatenate them together and perform a convolution to merge it together
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(len(feature_channels)*self.out_channels, 
                      self.out_channels, 
                      kernel_size=3, 
                      padding="same", 
                      bias=False), 
            nn.BatchNorm2d(self.out_channels), 
            nn.ReLU()
        )

    def up_and_add(self, x, y):
        
        """
        x is smaller than or equal to y, so interpolate x upto the size of y 
        and then sum together to merge features 
        """

        return F.interpolate(x, size=(y.shape[2], y.shape[3]), align_corners=True) + y

    def forward(self, features):

        ### Map everything to the lowest channels features (the first layer) ###
        for idx in range(len(features)-1):
        
            ### Grab one of the 3 layers ###
            conv_layer = self.conv1x1[idx]

            ### Grab a feature (starting from the second one) ###
            feature = features[idx+1]
        
            ### Replace the feature in the features list with our convolved feature ### 
            features[idx+1] = conv_layer(feature)


        ### Take low res features (later layers) and interpolate/add the the previous layer ##
        pyramid = []
        for i in reversed(range(1, len(features))):
            pyramid.append(self.up_and_add(features[i], features[i-1]))

        ### We now have 3 layers in P (the sum of consecutive layers) ###
        pyramid = [conv(x) for conv, x in zip(self.smooth_conv, pyramid)]

        ### Because we summed backwards, we have our early outputs at the end now, so flip back ###
        pyramid = list(reversed(pyramid))

        ### our last layer of the features hav the most granular features, lets add them back! ###
        pyramid.append(features[-1])

        ### Upsample all Images to Highest resolution (first block) ###
        H, W = pyramid[0].shape[2], pyramid[0].shape[3]
        
        pyramid[1:] = [F.interpolate(feature, size=(H,W), mode="bilinear") for feature in pyramid[1:]]

        ### Append the Feature ###
        output = self.conv_fusion(torch.cat(pyramid, dim=1))
        
        return output

class Feature2Pyramid(nn.Module):

    """
    The problem with ViT/MAE is that each feature we output will be of 
    shape (B x 768 x 14 x 14). This means we can't really have a feature
    pyramid (more like a feature cube!) So we need to do some work to construct
    this pyramid. 

    To do this, we can provide a rescaled parameter, in our case [4,2,1,0.5]. This
    will use upsampling or downsampling of each output layer (assuming we have 4)
    layers to construct our pyramid

    For example:
    rescales: [4,2,1,0.5]
    Layer1: (B x 768 x 14 x 14) -> (B x 768 x 56 x 56)
    Layer2: (B x 768 x 14 x 14) -> (B x 768 x 28 x 28)
    Layer3: (B x 768 x 14 x 14) -> (B x 768 x 14 x 14)
    Layer4: (B x 768 x 14 x 14) -> (B x 768 x 7 x 7)

    Code very close to MMSegmentation Implementaiton:
    https://github.com/open-mmlab/mmsegmentation/blob/main/mmseg/models/necks/featurepyramid.py    
    
    """

    def __init__(self, embed_dim, rescales=[4,2,1,0.5]):

        super(Feature2Pyramid, self).__init__()

        self.rescales = rescales
        self.embed_dim = embed_dim

        self.scales = nn.ModuleList()

        for k in self.rescales:

            if k == 4:
                self.up4 = nn.Sequential(
                    nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                    nn.BatchNorm2d(embed_dim), 
                    nn.GELU(),
                    nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2)
                )

                self.scales.append(self.up4)

            elif k == 2:

                self.up2 = nn.Sequential(
                    nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                    nn.BatchNorm2d(embed_dim), 
                    nn.GELU()
                )
                
                self.scales.append(self.up2)
            
            elif k == 1:

                self.identity = nn.Sequential(
                    nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding="same"),
                    nn.BatchNorm2d(embed_dim), 
                    nn.GELU()
                )
                
                self.scales.append(self.identity)
            
            elif k == 0.5:

                self.down2 = nn.Sequential(
                    nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(embed_dim), 
                    nn.GELU()
                )

                self.scales.append(self.down2)

            elif k == 0.25:

                self.down4 = nn.Sequential(
                    nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=3, stride=4, padding=1),
                    nn.BatchNorm2d(embed_dim), 
                    nn.GELU()
                )

                self.scales.append(self.down4)
            
            else:
                raise NotImplementedError("Currently support rescales of 4, 2, 1, 0.5, 0.25")
    

    def forward(self, inputs):

        pyramid = []

        for features, layer in zip(inputs, self.scales):
            pyramid.append(layer(features))

        return features

class UperNetHead(nn.Module):
    
    """
    UperNet Head

    Takes in features from an image encoder (ViT/ResNet/...)
    and creates a low resolution feature space that can be 
    projected back into a higher resolution segmentation mask 
    
    """

    def __init__(self, config):
        super(UperNetHead, self).__init__()

        self.config = config
    
        feature_channels = config.channels_per_layer
        
        ### Convert Features to Pyramid ###
        self.feat2pyr = Feature2Pyramid(embed_dim=config.embed_dim, 
                                        rescales=config.rescales)
        
        ### PSP Module Works on the Last Layer ###
        self.psp = PSPModule(in_channels=feature_channels[-1])
        
        ### FPN Modules Needs Number of Channels at Every Layer, but will
        ### Return a tensor in the shape of the first feature ###
        self.fpn = FPN(feature_channels)

        ### If FPN return a tensor in the shape of the first feature, 
        ### then the head will take that as the input 
        self.head = nn.Conv2d(in_channels=feature_channels[0])


    def forward(self, inputs):
        
        ### Build Pyramid ###
        pyramid = self.feat2pyr(inputs)

        ### Pooling Module On Last Feature ###
        pyramid[-1] = self.psp(pyramid[-1])

        ### Feature Pyramid Network ###
        output = self.fpn(pyramid)

        ### Prediction Head ###
        output = self.head(output)

        return output
            



