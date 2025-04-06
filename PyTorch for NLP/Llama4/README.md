
# Llama4

<img src="https://github.com/priyammaz/PyTorch-Adventures/blob/main/src/visuals/llama4.png?raw=true" alt="drawing" width="600"/>

Llama4 just came out! What better way to learn whats going on than implementing it. Full Credit for the code goes to the Huggingface 🤗 Team in their [modeling_llama4.py](https://github.com/huggingface/transformers/blob/d1b92369ca193da49f9f7ecd01b08ece45c2c9aa/src/transformers/models/llama4/modeling_llama4.py#L1766)

### Prereqs

There are a lot of pre-reqs here and I hope you know the general ideas behind:

- Transformers (Decoder)
- Mixture of Experts
- KV Cache
- Rotary Embeddings

### Great Resource

If you want to learn some of the details about Llama, look no further than [Umar Jamil](https://www.youtube.com/@umarjamilai) and the video on [Llama Explained](https://www.youtube.com/watch?v=Mn_9W1nCFLo). This should give you all the intuition you need to understand most of whats going on here!

To learn about Mixture of Experts, Huggingface has as great [article](https://huggingface.co/blog/moe) that you can read that should make it pretty clear about whats going on!