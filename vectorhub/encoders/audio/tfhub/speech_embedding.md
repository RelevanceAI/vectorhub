---
model_id: "audio/speech-embedding"
model_name: "Speech Embedding" 
vector_length: "96 (default)"
release_date: "2020-01-31"
paper: "https://arxiv.org/abs/2002.01322"
repo: "https://tfhub.dev/google/speech_embedding/1"
installation: "pip install vectorhub[encoders-audio-tfhub]"
category: audio
short_description: We show that using synthesized speech data in training small spoken term detection models can be more effective than using real data.
---

## Description

With the rise of low power speech-enabled devices, there is a growing demand to quickly produce models for recognizing arbitrary  sets of keywords. As with many machine learning tasks, one of the most challenging parts in the model creation process is obtaining a sufficient amount of training data. In this paper, we explore the effectiveness of synthesized speech data in training small spoken term detection models of around 400k parameters. Instead of training such models directly on the audio or low level features such as MFCCs, we use a pre-trained speech embedding model trained to extract useful features for keyword spotting models. Using this speech embedding, we show that a model which detects 10 keywords when trained on only synthetic speech is equivalent to a model trained on over 500 real examples. We also show that a model without our speech embeddings would need to be trained on over 4000 real examples to reach the same accuracy.


## Working in Colab

If you are using this in colab and want to save this so you don't have to reload, use: 

```
import os 
os.environ['TFHUB_CACHE_DIR'] = "drive/MyDrive/"
os.environ["TFHUB_MODEL_LOAD_FORMAT"] = "COMPRESSED"
```

## Example

```
#pip install vectorhub[encoders-audio-tfhub]
from vectorhub.encoders.audio.tfhub import SpeechEmbedding2Vec
model = SpeechEmbedding2Vec()
vector = model.encode('https://vecsearch-bucket.s3.us-east-2.amazonaws.com/voices/common_voice_en_2.wav')
```
