# Masked AutoEncoders are Scalable Vision Learners

<img src="https://github.com/priyammaz/PyTorch-Adventures/blob/main/src/visuals/mae.png?raw=true" alt="drawing" width="700"/>


### Model Structure
Just as Masked Language Modeling underpins architectures like BERT, Masked Image Modeling serves as a powerful method for large-scale image pretraining. This implementation focuses on reproducing the Masked Autoencoder (MAE) on the ImageNet dataset.

A key difference between the MAE architecture and models like BERT lies in their Encoder/Decoder structure. In BERT, text sequences are randomly masked, and the input includes the masked tokens replaced with a specific mask token. In contrast, the MAE approach masks 75% of the image patches, and only the remaining 25% of visible patches are passed to the encoder. This design significantly reduces the computational load on the encoder.The decoder then processes the full sequence of image patches, consisting of both the encoded visible patches and the masked tokens, to reconstruct the original image. The decoder is lightweight, utilizing a smaller embedding dimension and fewer transformer blocks compared to the encoder. This architectural design reduces the overall computation required for the reconstruction task while enabling efficient learning of high-quality visual representations.

The main benefit is, once pretrained, the encoder is just a normal Vision Transformer. We can pass in the full images (rather than mask) and finetune for downstream tasks like Classification and Segmentation, things we will implement here today!

