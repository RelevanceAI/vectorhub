---
model_id: "audio/wav2vec"
model_name: "Wav2Vec" 
vector_length: "512 (default)" 
paper: "https://arxiv.org/abs/2006.11477"
repo: "https://github.com/pytorch/fairseq"
installation: "pip install vectorhub[encoders-audio-pytorch]"
release_date: "2020-06-20"
category: audio
short_description: We show for the first time that learning powerful representations from speech audio alone followed by fine-tuning on transcribed speech can outperform the best semi-supervised methods while being conceptually simpler.
---

## Description

We show for the first time that learning powerful representations from speech audio alone followed by fine-tuning on transcribed speech can outperform the best semi-supervised methods while being conceptually simpler. wav2vec 2.0 masks the speech input in the latent space and solves a contrastive task defined over a quantization of the latent representations which are jointly learned. Experiments using all labeled data of Librispeech achieve 1.8/3.3 WER on the clean/noisy test sets. When lowering the amount of labeled data to one hour, wav2vec 2.0 outperforms the previous state of the art on the 100 hour subset while using 100 times less labeled data. Using just ten minutes of labeled data and pre-training on 53k hours of unlabeled data still achieves 5.2/8.6 WER on the noisy/clean test sets of Librispeech. This demonstrates the feasibility of speech recognition with limited amounts of labeled data.

## Example

```
#pip install vectorhub[encoders-audio-pytorch]
from vectorhub.encoders.audio.pytorch import Wav2Vec
model = Wav2Vec()
sample = model.read('https://vecsearch-bucket.s3.us-east-2.amazonaws.com/voices/common_voice_en_2.wav')
model.encode(sample)
```
