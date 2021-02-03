---
model_id: "text/use-multi-mlm"
model_name: "USE with conditional MLM Multilingual"
vector_length: "1024 (Base model)"
paper: "https://arxiv.org/abs/1803.11175" 
repo: "https://tfhub.dev/google/universal-sentence-encoder-cmlm/en-large/1"
installation: "pip install vectorhub[encoders-text-tfhub]"
release_date: "2021-01-31"
category: text
---

*WARNING* This model currently has memory leaks that have yet to be patched. 

## Description

The universal sentence encoder family of models map text into high dimensional vectors that capture sentence-level semantics. Our English-large (en-large) model is trained using a conditional masked language model described in [1]. The model is intended to be used for text classification, text clustering, semantic textural similarity, etc. It can also be use used as modularized input for multimodal tasks with text as a feature. The model can be fine-tuned for all of these tasks. The large model employs a 24 layer BERT transformer architecture.


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
from vectorhub.encoders.text.tfhub import USEMultiTransformer2Vec
model = USEMultiTransformer2Vec()
model.encode("I enjoy taking long walks along the beach with my dog.")
```
