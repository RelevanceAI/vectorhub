---
model_id: "text/elmo"
model_name: "Elmo (Embeddings From Language Models)" 
vector_length: "1024 (default)"
paper: "https://arxiv.org/abs/1802.05365"
repo: "https://tfhub.dev/google/elmo/3"
installation: "pip install vectorhub[encoders-text-tfhub]"
release_date: "2020-07-03"
category: text
---

## Description

Computes contextualized word representations using character-based word representations and bidirectional LSTMs, as described in the paper "Deep contextualized word representations" [1].

ELMo (Embeddings from Language Models) representations are deep as they are a function of all of the 
internal layers of the biLM. More specifically, we learn a linear combination of the vectors stacked above each input word for each end task. 


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
#FOR WINDOWS: pip install vectorhub[encoders-text-tfhub-windows]
from vectorhub.encoders.text.tfhub import Elmo2Vec
model = Elmo2Vec()
model.encode("I enjoy taking long walks along the beach with my dog.")
```
