# Wasserstein GAN

<img src="https://github.com/priyammaz/PyTorch-Adventures/blob/main/src/visuals/optimal_transport.png?raw=true" alt="drawing" width="700"/>

[Image Source](https://medium.com/analytics-vidhya/introduction-to-optimal-transport-fd1816d51086)

Moving dirt from one place to another can become surprisingly complicated, although if you want to find the most efficient way to move from one pile and create a new shape altogether, this becomes a much deeper mathematical problem. This is the fundamental idea behind the field known as [Optimal Transport](https://en.wikipedia.org/wiki/Transportation_theory_(mathematics))

The Wasserstein Distance is a measure of *work* to move dirt from one pile to another, or in our case, to convert one distribution to another. In the GAN formulation, we have two distributions: $P_G$ is the distribution of the generated images and $P_R$ is the distribution of the real images. The original GAN formulation would pit these two against each other through a mixmax game, but this leads to some problems such as Mode Collapse. In the Wasserstein GAN we reformulate the problem to find the most efficent way to transport $P_G$ to $P_R$, and can lead to smoother optimization spaces. 

We will in this tutorial:

- Explore the Earth Movers Distance
- Explore the basics of Optimal Transport
- Derive a limitation of the Jensen Shannon Divergence from GANs
- Learn about Constrained Optimization
  - Lagrangian Multipliers
  - Primal/Dual Forms
- Derive the Dual Form of the Wasserstein Distance
  - Kantorovich-Rubinstein Duality
- Train WGAN with Weight Clipping
- Train WGAN with Gradient Penalties
  - How to use ```torch.autograd.grad```
