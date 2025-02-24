# Exploratory Data Adventures!

![banner](src/visuals/banner.png)

---
[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/) &nbsp; 
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/) &nbsp; 
[![](https://img.shields.io/badge/contributors-welcome-informational?style=for-the-badge)](https://github.com/priyammaz/HAL-DL-From-Scratch/graphs/contributors) &nbsp;
[![Licence](https://img.shields.io/github/license/Ileriayo/markdown-badges?style=for-the-badge)](./LICENSE)

### Open Source Learning and the Democratization of AI

The greatest aspect of Artificial Intelligence today is its open source nature. Very rarely in history have key technologies, especially those as powerful as these models developed today, have been made completely available to the public. Without the incredible online resources made by all of the contributing scientists and engineers, I never would have learned as much as I know today. This repository acts as a documentation of my own exploration as well as an attempt to teach everything I learn to others. I will try my best to cite everyone and everything I reference that helped me learn everything as well!

The main limitation for researchers in this field today is we normally don't have buckets of GPU's just sitting around to train with! Every example I do will be a proof of concept, but to the best of my ability I will attempt to reproduce any model that is feasible!

#### Contributions

I am typically more wrong than I am right! If you find any errors in my work, that means there is an error in my knowledge. Please let me know as I want this to be as accurate as possible, but also, I want to learn as much as I can! If you want to contribute anything yourself, just submit a PR and I will review it!

### Data Prep ###
We will be using a couple of datasets in our Deep Learning Adventures!!
- [Cats Vs Dogs](https://www.microsoft.com/en-us/download/details.aspx?id=54765)
- [IMBD Movie Reviews](https://ai.stanford.edu/~amaas/data/sentiment/)
- [MNIST Handwritten Digits](https://pytorch.org/vision/stable/datasets.html)
- [Harry Potter Corups](https://github.com/formcept/whiteboard/tree/master/nbviewer/notebooks/data/harrypotter)

Ensure you have a */data* folder in your root directory of the git repo and run the following to install all datasets
```
bash download_data.sh 
```
#### Extra Datasets ####
There are a few other datasets that we will use but are inconsistent to automatically download and are used in the more advanced architectures! Just download them from the link and save them in the */data* folder! These datasets may also be too large to train in Google Drive so keep that in mind!
- [CelebA](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)
- [MS-COCO](https://cocodataset.org/#download)
- [ADE20K](http://sceneparsing.csail.mit.edu/)

## Foundations
- [**Intro to PyTorch: Exploring the Mechanics**](PyTorch%20Basics/Intro%20to%20PyTorch/) &nbsp; [<img src="src/visuals/x_logo.png" alt="drawing" style="width:20px;"/>](https://x.com/data_adventurer/status/1834073826612707543)&nbsp; [<img src="src/visuals/play_button.png" alt="drawing" style="width:30px;"/>](https://youtu.be/d86lJxKInYg?feature=shared) &nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1YQanR0ME7ThsU9YwLzXhGvYGOdH2ErSa?usp=sharing)


- [**PyTorch Datasets and DataLoaders**](PyTorch%20Basics/PyTorch%20DataLoaders/) &nbsp; [<img src="src/visuals/x_logo.png" alt="drawing" style="width:20px;"/>](https://x.com/data_adventurer/status/1834084927215730801)&nbsp; [<img src="src/visuals/play_button.png" alt="drawing" style="width:30px;"/>](https://youtu.be/S8X6qcColBY?feature=shared)  &nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1nurV-kJmoPYlXP-qNAGGLsFXuS3lpNil?usp=sharing)


- [**Leveraging Pre-Trained Models for Transfer Learning**](PyTorch%20Basics/Basics%20of%20Transfer%20Learning/) &nbsp; [<img src="src/visuals/x_logo.png" alt="drawing" style="width:20px;"/>](https://x.com/data_adventurer/status/1839491569533223011)&nbsp; [<img src="src/visuals/play_button.png" alt="drawing" style="width:30px;"/>](https://www.youtube.com/watch?v=c6VTUx0EdqM)&nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KYCINwxq-y8QOMCRylsxDaP9RCUHz-bV?usp=sharing)


- [**Intro to Vision: Digging into Convolutions & AlexNet**](PyTorch%20for%20Computer%20Vision/Intro%20to%20Vision/) &nbsp; [<img src="src/visuals/x_logo.png" alt="drawing" style="width:20px;"/>](https://x.com/data_adventurer/status/1882126373633872065)&nbsp; [<img src="src/visuals/play_button.png" alt="drawing" style="width:30px;"/>](https://youtu.be/WoIxtSBYyYA)&nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1BE-y1876znMeijFu4AX4qcZdt-fs8o7a?usp=sharing)


- [**The Curse of Backprop: Residual Connections & Implementing ResNet**](PyTorch%20for%20Computer%20Vision/ResNet/) &nbsp; [<img src="src/visuals/x_logo.png" alt="drawing" style="width:20px;"/>](https://x.com/data_adventurer/status/1883133149317829104)&nbsp; [<img src="src/visuals/play_button.png" alt="drawing" style="width:30px;"/>](https://www.youtube.com/watch?v=TqIU9K8nNhs)&nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OPnOApHCcZFFWkw-zfhNvfyQeswQxgea?usp=sharing)


- [**Digging into the Sequences: Classification with Recurrence**](PyTorch%20for%20NLP/Recurrent%20Neural%20Networks/IMDB%20Classification/) &nbsp; [<img src="src/visuals/x_logo.png" alt="drawing" style="width:20px;"/>](https://x.com/data_adventurer/status/1883135474476208152)&nbsp; [<img src="src/visuals/play_button.png" alt="drawing" style="width:30px;"/>](https://www.youtube.com/watch?v=UBjmWHX8xlI)&nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1c98opjQt1w-HTp10U1myjSWU9acDsaV4?usp=sharing)


- [**Lets Write a Story: Recurrence for Text Generation**](PyTorch%20for%20NLP/Recurrent%20Neural%20Networks/Harry%20Potter%20Generation/) &nbsp; [<img src="src/visuals/x_logo.png" alt="drawing" style="width:20px;"/>](https://x.com/data_adventurer/status/1885066005275119943)&nbsp; [<img src="src/visuals/play_button.png" alt="drawing" style="width:30px;"/>](https://www.youtube.com/watch?v=f8qoaeF2kzY)&nbsp; &nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KO4JeIHRiKxiRJdK7gY-B9bZGfDSvCt_?usp=sharing)

- [**Distributed Training with Huggingface 🤗 Accelerate**](PyTorch%20Basics/Huggingface%20Accelerate/)&nbsp; [<img src="src/visuals/x_logo.png" alt="drawing" style="width:20px;"/>](https://x.com/data_adventurer/status/1886848974243483895)&nbsp; [<img src="src/visuals/play_button.png" alt="drawing" style="width:30px;"/>](https://youtu.be/cHKyhHu6WW0)&nbsp;

## Neural Networks from Scratch ##
- [**ManualGrad: Simple Neural Network Implementation**](Neural%20Networks%20from%20Scratch/ManualGrad/)
- [**MyTorch: Building a Simple AutoGrad Engine**](Neural%20Networks%20from%20Scratch/AutoGrad/)
  
## Computer Vision ##

- [**UNet for Image Segmentation**](PyTorch%20for%20Computer%20Vision/UNET%20for%20Segmentation/)
- [**Moving from Convolutions: Vision Transformer**](PyTorch%20for%20Computer%20Vision/Vision%20Transformer) &nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Mh-yaSWwfTs1UcOdRQjRIvLuj6PU6liZ?usp=sharing)
- [**Masked Image Modeling with Masked Autoencoders**](PyTorch%20for%20Computer%20Vision/Masked%20AutoEncoder/)
  
## Natural Language Processing ##
- [**Causal Language Modeling: GPT**](PyTorch%20for%20NLP/GPT%20for%20Causal%20Language%20Models)&nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1DZ406Ytb-ls1jDI1BovARwYq__ptr1Tx?usp=sharing)
- [**Masked Language Modeling: RoBERTa**](PyTorch%20for%20NLP/RoBERTa%20for%20Masked%20Language%20Models)&nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1MChQ84-1VKBbjNCmzPQL02hxl-gckEYh?usp=sharing)
  
- [**Attention is All You Need for Language Translation**](PyTorch%20for%20NLP/Seq2Seq%20for%20Neural%20Machine%20Translation/)

## Speech Processing ##
- **Intro to Audio Processing in PyTorch**
- **Connectionist Temporal Classification Loss**
- **Intro to Automatic Speech Recognition**
- [**Quantized Audio Pre-Training: Wav2Vec2**](PyTorch%20for%20Audio/Wav2Vec2/)
- **EnCodec**

## Generative AI
- ### AutoEncoders ##
  - [**Intro to AutoEncoders**](PyTorch%20for%20Generation/AutoEncoders/Intro%20to%20AutoEncoders/Intro_To_AutoEncoders.ipynb)&nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1DldfPN9q1uSA4UkZYHV-3Ms5be333EKN?usp=sharing)
  - [**Variational AutoEncoders**](PyTorch%20for%20Generation/AutoEncoders/Intro%20to%20AutoEncoders/Variational_AutoEncoders.ipynb)&nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_NLc6g5UJ-tmRUXZbF5r1FgWoEApaLmH?usp=sharing)
  - **Beta-VAE**
  - [**Vector-Quantized Variational Autoencoders**](PyTorch%20for%20Generation/AutoEncoders/Intro%20to%20AutoEncoders/Vector_Quantized_Variational_AutoEncoders.ipynb)&nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1QqdHlnfJV5BATUymrXy-wi3F8YUIQFpl?usp=sharing)
  - **Gumbel Softmax Vector Quantization**

- ### Autoregressive Generation ##
  - **PixelCNN**
  - **WaveNet**

- ### Generative Adversarial Networks ##
  - [**Intro to Generative Adversarial Networks**](PyTorch%20for%20Generation/Generative%20Adversarial%20Network/Intro%20to%20GANs/)&nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1s2zEnDKo0zr4hqzSmsDzQvWHs40KPkgz?usp=sharing)
  - [**Deep Convolutional GAN**](PyTorch%20for%20Generation/Generative%20Adversarial%20Network/DCGAN/)&nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1kB-6KvR0IVsXtWI6MElNd3m02mpX7S9u?usp=sharing)
  - [**Wasserstein GAN**](PyTorch%20for%20Generation/Generative%20Adversarial%20Network/WGAN/)&nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1HneZAIokGAZW8Eq0ussFBoaw3BmG3imm?usp=sharing)
  - **SuperResolution with SRGAN**
  - **Image2Image Translation with CycleGAN**

- ### Diffusion ##
  - [**Intro to Diffusion**](PyTorch%20for%20Generation/Diffusion/Intro%20to%20Diffusion/) &nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KBupTiAId1LO67IcM-yn3xkK81aj06sG?usp=sharing)
  - [**Text-Conditional Diffusion with Classifier Free Guidance**](PyTorch%20for%20Generation/Diffusion/Conditional%20Diffusion/)
  - [**Latent-Space Diffusion**](PyTorch%20for%20Generation/Diffusion/Latent%20Diffusion/)
  - **Diffusion Transformers**

## MultiModal AI ##
- **Building Vision/Language Representations: CLIP**
- **Automatic Image Captioning**
- **Visual Question Answering**

## Dive into Transformers ##

- **Attention Mechanisms**
  - [**Attention is All You Need**](PyTorch%20for%20Transformers/Attention%20Mechanisms/Attention/) &nbsp; [<img src="src/visuals/x_logo.png" alt="drawing" style="width:20px;"/>](https://x.com/data_adventurer/status/1893829898277671057)&nbsp; [<img src="src/visuals/play_button.png" alt="drawing" style="width:30px;"/>](https://youtu.be/JXY5CmiK3LI)&nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1vJAlxRehKG7yzAFJTTPa4Ecew0cxc1fL?usp=sharing)
  - [**Windowed Attention**](PyTorch%20for%20Transformers/Attention%20Mechanisms/Sliding%20Window%20Attention/)
  - **LinFormer**

- **Positional Embeddings**
  - **Sinusoial Encodings**
  - **Rotary Positional Encoding**
  - **ALiBi**

- **Feed Forward**
  - **Mixture of Experts**
  
- **Normalization**
  - **Layer Norm**
  - **RMS Norm**
  
- **Optimization**
  - **KV Cache**

## Reinforcement Learning
- ### Model-Based Learning 
  - [**Policy Iteration**](PyTorch%20for%20Reinforcement%20Learning/Intro%20to%20Reinforcement%20Learning/Model-Based%20Learning/intro_rl_and_policy_iter.ipynb)
  - [**Value Iteration**](PyTorch%20for%20Reinforcement%20Learning/Intro%20to%20Reinforcement%20Learning/Model-Based%20Learning/value_iteration.ipynb)
- ### Model-Free Learning 
  - **Temporal Difference Learning**
  - **SARSA**
  - **Q Learning**
- **Deep Reinforcement Learning**
  - **Deep Q Learning**
  - **Double Deep Q Learning**
  - **Dueling Deep Q Learning**
  - **Prioritized Experience Replay**

## Tools
- [**Low Rank Adaptation**](PyTorch%20Tools/LoRA/)
- **QLoRA**
- **TensorRT**
- **DeepSpeed**
