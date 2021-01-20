---
model_id: "audio/vggish"
model_name: "VGGish" 
vector_length: "128 (default)"
release_date: "2020-03-11"
repo: "https://tfhub.dev/google/vggish/1"
installation: "pip install vectorhub[encoders-audio-tfhub]"
category: audio
short_description: VGGish is a model for audio event embedding which uses the VGG-16 network and is trained on the YouTube-8M dataset.
---

## Description

An audio event embedding model trained on the YouTube-8M dataset.
VGGish should be used:
- as a high-level feature extractor: the 128-D embedding output of VGGish can be used as the input features of another shallow model which can then be trained on a small amount of data for a particular task. This allows quickly creating specialized audio classifiers without requiring a lot of labeled data and without having to train a large model end-to-end.
- as a warm start: the VGGish model parameters can be used to initialize part of a larger model which allows faster fine-tuning and model exploration.

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
from vectorhub.encoders.audio.tfhub import Vggish2Vec
model = Vggish2Vec()
sample = model.read('https://vecsearch-bucket.s3.us-east-2.amazonaws.com/voices/common_voice_en_2.wav')
model.encode(sample)
```

## Limitations

VGGish has been trained on millions of YouTube videos and although these are very diverse, there can still be a domain 
mismatch between the average YouTube video and the audio inputs expected for any given task. You should expect to do some 
amount of fine-tuning and calibration to make VGGish usable in any system that you build.
