## Going Deeper with Residual Connections &nbsp; [<img src="../../src/visuals/x_logo.png" alt="drawing" style="width:20px;"/>](https://x.com/data_adventurer/status/1883133149317829104)&nbsp; [<img src="../../src/visuals/play_button.png" alt="drawing" style="width:30px;"/>](https://www.youtube.com/watch?v=TqIU9K8nNhs)&nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OPnOApHCcZFFWkw-zfhNvfyQeswQxgea?usp=sharing)


The "Deepness" of a neural network is determined by how many computational layers we include
in a model. It was quickly shown though that the deeper a model got, the harder it is to optimize
until the point gradient descent is no longer able to do anything. Residuals were proposed in the paper
[Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf) where earlier 
layers were summed to later layers of computation, offering additional paths for gradients to flow
backward! ResNet introduced a pivotal concept that is used all over modern Deep Learning today. 

![resnet](../../src/visuals/residual_block.png)

#### We will be exploring a couple of things in this notebook!
- Deep dive into backpropagation, some of its mathematical derivations and limitations
- Implementation of the Larger ResNet Models (50, 101 and 152 Layers)
- Training ResNet With and Without Residuals to compare training performance


![perf](../../src/visuals/residuals_performance.png)