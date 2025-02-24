# Deep Convolutional GAN

<img src="https://github.com/priyammaz/PyTorch-Adventures/blob/main/src/visuals/dcgan_implementation.png?raw=true" alt="drawing" width="700"/>

## Convolutions are Just Better!
So in our [Intro to Gans](https://github.com/priyammaz/PyTorch-Adventures/tree/main/PyTorch%20for%20Generation/Generative%20Adversarial%20Network/Intro%20to%20GANs) we took a look at the GAN Formulation, but mostly just stuck to Linear Layers on MNIST. It sets the stage, but for the purposes of Images, we know Convolutions are the better method! Well, incomes the [DCGAN](https://arxiv.org/abs/1511.06434v2)!

Nothing too crazy here, all that changes is we are using a Convolutional Neural Network for both the Generator and Discriminator! MNIST is also too simple an example, so we will be applying to to the [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) Dataset