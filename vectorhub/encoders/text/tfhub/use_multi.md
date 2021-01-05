---
model_id: "text/use-multi"
model_name: "USE Multi - Universal Sentence Encoder Multilingual"
vector_length: "512 (Base model)"
paper: "https://arxiv.org/abs/1803.11175" 
repo: "https://tfhub.dev/google/collections/universal-sentence-encoder/1"
installation: "pip install vectorhub[encoders-text-tfhub]"
release_date: "2018-03-29"
category: text
---

## Description

The Universal Sentence Encoder Multilingual module is an extension of the Universal Sentence Encoder Large that includes training on multiple tasks across languages. Supports 16 languages (Arabic, Chinese-simplified, Chinese-traditional, English, French, German, Italian, Japanese, Korean, Dutch, Polish, Portuguese, Spanish, Thai, Turkish, Russian) text encoder.

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
from vectorhub.encoders.text.tfhub import USEMulti2Vec
model = USEMulti2Vec()
model.encode("I enjoy taking long walks along the beach with my dog.")
```
