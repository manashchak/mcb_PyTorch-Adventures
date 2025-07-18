# Attention is All You Need &nbsp; [<img src="../../../src/visuals/x_logo.png" alt="drawing" style="width:20px;"/>](https://x.com/data_adventurer/status/1893829898277671057)&nbsp; [<img src="../../../src/visuals/play_button.png" alt="drawing" style="width:30px;"/>](https://youtu.be/JXY5CmiK3LI)&nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1vJAlxRehKG7yzAFJTTPa4Ecew0cxc1fL?usp=sharing)

![image](https://raw.githubusercontent.com/priyammaz/PyTorch-Adventures/main/src/visuals/attention_mechanism_visual.png)

Attention networks have become crucial in state of the art architectures, namely Transformers! Today we will be delving a bit deeper into attention and how it works! Although attention was mainly intended for use in sequence modeling, it has found its way into Computer Vision, Graphs and basically every domain, demonstrating the flexibility of the architecture. Lets discuss this from a sequence modeling perspective today though just to build intuition on how this works.

Today we will learn to build the two types of Transformers, Encoders and Decoders.

### Encoder Transformers

These are equivalent to bidirectional RNNs, where at every timestep in our sequence we can look forward and backwards. We will start by exploring how to build this! These are the building blocks of architectures like BERT, RoBERTa, Wav2Vec2 and a variety of other models!

### Decoder Transformers

These are causal RNNs, where at every timestep, we are only allowed to look into the current and past! Therefore, we cannot look into the future! This is just a small change on top of the original Encoder transformer, by including a causal mask. These are the building blocks of architectures like GPT, LLama, and other autoregressive generative models.

