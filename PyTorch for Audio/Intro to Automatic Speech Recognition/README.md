# DeepSpeech2

<img src="https://github.com/priyammaz/PyTorch-Adventures/blob/main/src/visuals/deepspeech2.png?raw=true" alt="drawing" width="500"/>

We use DeepSpeech2 as a simple example of exploring how to build an ASR model. All ASR models basically have three parts:

- Convolutions to extract features (on Mel Spectrograms in this case)
- Sequence Modeling to learn the temporal relationships of the features
- CTC Loss for alignment-free transcriptions

This is a very simple implementation that will outline a lot of the principles we will be using going foward (especially for our Wav2Vec2 Implementation!)