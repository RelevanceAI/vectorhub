---
model_name: "Trill - Triplet Loss Network"
model_id: "audio/trill"
vector_length: "512 (default)" 
paper: "https://arxiv.org/abs/2002.12764"
release_date: "2020-02-25"
repo: "https://tfhub.dev/google/nonsemantic-speech-benchmark/trill/3"
installation: pip install vectorhub['encoders-audio-tfhub']
category: audio
short_description: Introduces a benchmark for comparing speech representations on non-semantic tasks, and proposes a representation based on an unsupervised triplet-loss objective.
---

## Description

The ultimate goal of transfer learning is to reduce labeled data requirements by exploiting a pre-existing embedding model trained for different datasets or tasks. The visual and language communities have established benchmarks to compare embeddings, but the speech community has yet to do so. This paper proposes a benchmark for comparing speech representations on non-semantic tasks, and proposes a representation based on an unsupervised triplet-loss objective. The proposed representation outperforms other representations on the benchmark, and even exceeds state-of-the-art performance on a number of transfer learning tasks. The embedding is trained on a publicly available dataset, and it is tested on a variety of low-resource downstream tasks, including personalization tasks and medical domain. The benchmark, models, and evaluation code are publicly released.

## Working in Colab

If you are using this in colab and want to save this so you don't have to reload, use: 

```
import os 
os.environ['TFHUB_CACHE_DIR'] = "drive/MyDrive/"
os.environ["TFHUB_MODEL_LOAD_FORMAT"] = "COMPRESSED"
```


## Example

```python
#pip install vectorhub[encoders-audio-tfhub]
from vectorhub.encoders.audio.tfhub import Trill2Vec
model = Trill2Vec()
sample = model.read('https://vecsearch-bucket.s3.us-east-2.amazonaws.com/voices/common_voice_en_2.wav')
model.encode(sample)
```