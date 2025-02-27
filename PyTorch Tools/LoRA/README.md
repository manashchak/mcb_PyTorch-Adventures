# Low Rank Adaptation (LoRA)

<img src="https://github.com/priyammaz/PyTorch-Adventures/blob/main/src/visuals/lora.png?raw=true" alt="drawing" width="400"/>


## Why Does LoRA Matter?
LoRA (and QLoRA) have become really great ways to finetune models. The main idea is, whenever we finetune a model, all we are doing is taking the existing weights $W$ and changing it by some perturbation. So the first idea is, why not just learn the perturbation? The second idea is, do we need to learn the entire perturbation, or can we learn some low rank (compressed) version of it. This enables two things:

- We only have to train a small set of weights and can leave the pretrained weights alone (so less compute!)
- When saving files, we can just save the weights we trained instead of all the weights in the model (so less storage!)

Less Storage and Less Compute? Sign me up!

## What Does it Save?

Lets take a common linear layers found in Transformers: The Query Projection. This takes a data matrix of the shape ```(B x S x 768)```, assuming we have a text embed dim of 768, and projects it with a linear layer with weights of the shape ```(768 x 768)``` (i.e. 768 inputs 768 outputs). That means this linear layer has a **whopping 589824 parameters**! This also means when finetuning, we will be training all of these weights. We would typically write this as:

$$O = XW^T$$

What if instead of having one big ```(768 x 768)``` linear layer, we break it up into having two linear layers. The first one called $A$ will go from ```(768 x 16)``` and the second called $B$ will be ```(16 x 768)```. Instead of learning the full linear layer $W$ from before, we learn the perturbation and do the following:

$$O = XW^T + \alpha * XAB$$

In this situation, we will not compute any gradients on $W$, as we are not changing it, we only want to update $A$ and $B$. Why bother with this? Well lets look at the weight matricies.

- $A$ is a ```(768 x 16)``` matrix with 12288 parameters
- $B$ is a ```(16 x 768)``` matrix with 12288 parameters

This means that instead of training 589824 parameters from before, we only are training 24576 parameters. That is just 4% of our original matrix! As long as this doesn't hurt our results this is a huge win!

Also why pick ```(768 x 16)```, why not ```(768 x 8)``` or ```(768 x 32)```? The bottleneck we pick is typically known as the ```rank```, you want to go as small as you can without hurting performance!

The second part is $\alpha$ which controls the strength of LoRA. It controls how much we want to apply our perturbations to our original weights. 

## What Layers Can We Apply LoRA To?

Any of them actually! We will be implementing 3 of the most common here today:

- Linear Layers ```(in_features, out_features)```
- Embedding Layers ```(vocab_size, embed_dim)```
- Convolutional Layers ```(out_channels, in_channels, kernel_height, kernel_width)```

## Weight Merging

What if you want to save the whole model? Well we can merge our LoRA weights into the original weights like the following:

$$O = XW^T + \alpha * XAB = X(W^T + \alpha AB)$$

Now you have to save the whole models weights but thats ok too once you are done training, its really whatever you want. 

## Finetuning Results

So how does this perform? Lets take a look! I do two cases here: Vision and Language Finetuning. 

### Vision Transformer Finetuning

We will be finetuning ViT on the Foods dataset with the same settings, only changing between using LoRA, training the full model, or finetuning only the classifier head. 

- Epochs: 3
- lr: 3e-5
- rank: 8
- lora_alpha: 8
- use_rslora = True

**Note**: LoRA Was applied to all QKV Layers and the Dense Layer found in the Feed Forward Module

| Model              | Learnable Params |  %Trainable | Accuracy |
| :---------------- |  ----: |----: |----: |
| ViT-Base (All Params)    | 85,876,325 | 100% | 88.5% | 
| ViT-Base + LoRA | 1,500,773 | 1.9% | 84.6% |
| ViT-Base Head Only | 77,568 | 0.1% | 77.3% |


### RoBERTa Finetuning

We will be finetuning RoBERTa on the IMDB Movie Review Dataset with the same settings, only changing between using LoRA, training the full model, or finetuning only the classifier head. 

- Epochs: 3
- lr: 3e-5
- rank: 8
- lora_alpha: 8
- use_rslora = True

**Note**: LoRA Was applied to all QKV Layers, the Dense Layer found in the Feed Forward Module and the word embedding matrix

| Model              | Learnable Params |  %Trainable | Accuracy |
| :---------------- |  ----: |----: |----: |
| RoBERTa-Base (All Params)    | 124,647,170 | 100% | 95.4% | 
| RoBERTa-Base + LoRA | 2,410,442 | 1.9% | 94.8% |
| RoBERTa-Base Head Only | 592,130 | 0.5% | 82.6% |

### Where are the Memory Savings?

You will notice that using LoRA vs not using it doesn't actually give you a ton of savings in memory. This is because, when training a model, your weights are not the only thing you have. During backpropagation, the model needs the intermediate activations from the model, and by default, the computational graph holds onto all of them. This means, the deeper the model the more of these intermediate states there are. Real memory savings come from using ```gradient_checkpointing```. For modules set for checkpointing, instead of holding onto the intermediate outputs, their forward functions will be recomputed during backprop. This will save a ton of memory, but will cost more in compute. On the other hand, when training multi-billion parameter models, this actually lets you perform it with low resources!

To further drive down the memory cost, we can use parameter quantization, which is QLoRA! We will implement that in the next one, using the ```bitsandbytes``` package

## PEFT

The best way to do this is [PEFT](https://github.com/huggingface/peft), an incredble package by Huggingface 🤗! My code is not battle tested at all and should be taken with a grain of salt, again this is all about learning!


