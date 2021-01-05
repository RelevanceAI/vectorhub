---
model_id: "text/use-lite"
model_name: "USE Lite - Universal Sentence Encoder Lite" 
vector_length: "512 (default)"
paper: "https://arxiv.org/abs/1803.11175"
repo: "https://tfhub.dev/google/universal-sentence-encoder-lite/2"
installation: pip install vectorhub[encoders-text-tfhub]
release_date: "2018-03-29"
category: text
---

## Description

The Universal Sentence Encoder Lite module is a lightweight version of Universal Sentence Encoder. This lite version is good for use cases when your computation resource is limited. For example, on-device inference. It's small and still gives good performance on various natural language understanding tasks.

## Working in Colab

If you are using this in colab and want to save this so you don't have to reload, use: 

```
import os 
os.environ['TFHUB_CACHE_DIR'] = "drive/MyDrive/"
os.environ["TFHUB_MODEL_LOAD_FORMAT"] = "COMPRESSED"
```

## Example

```python
#pip install vectorhub[encoders-text-tfhub]
from vectorhub.encoders.text.tfhub import USELite2Vec
model = USELite2Vec()
model.encode("I enjoy taking long walks along the beach with my dog.
```
