---
model_name: "Trill Distilled - Triplet Loss Network" 
model_id: "audio/trill-distilled"
vector_length: 512
paper: "https://arxiv.org/abs/2002.12764"
repo: "https://tfhub.dev/google/nonsemantic-speech-benchmark/trill/3"
installation: "pip install vectorhub[encoders-audio-tfhub]"
---

## Description

The ultimate goal of transfer learning is to reduce labeled data requirements by exploiting a pre-existing embedding model trained for 
different datasets or tasks. The visual and language communities have established benchmarks to compare embeddings, but the speech 
community has yet to do so. This paper proposes a benchmark for comparing speech representations on non-semantic tasks, and proposes a 
representation based on an unsupervised triplet-loss objective. The proposed representation outperforms other representations on the 
benchmark, and even exceeds state-of-the-art performance on a number of transfer learning tasks. The embedding is trained on a publicly 
available dataset, and it is tested on a variety of low-resource downstream tasks, including personalization tasks and medical domain. 
The benchmark, models, and evaluation code are publicly released.

## Example

```
#pip install vectorhub[encoders-audio-tfhub]
from vectorhub.encoders.audio.tfhub import TrillDistilled2Vec
model = TrillDistilled2Vec()
sample = model.read('https://vecsearch-bucket.s3.us-east-2.amazonaws.com/voices/common_voice_en_2.wav')
model.encode(sample)
```