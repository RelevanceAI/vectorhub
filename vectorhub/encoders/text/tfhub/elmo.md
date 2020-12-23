---
model_id: "text/elmo"
model_name: "Elmo (Embeddings From Language Models)" 
vector_length: "1024 (default)"
paper: "https://arxiv.org/pdf/2007.01852v1.pdf"
repo: "https://tfhub.dev/google/LaBSE/1"
installation: "pip install vectorhub[encoders-text-tfhub]"
release_date: "2020-07-03"
category: text
---

## Description

Computes contextualized word representations using character-based word representations and bidirectional LSTMs, as described in the paper "Deep contextualized word representations" [1].

ELMo (Embeddings from Language Models) representations are deep as they are a function of all of the 
internal layers of the biLM. More specifically, we learn a linear combination of the vectors stacked above each input word for each end task. 


## Example

```python
#pip install vectorhub[encoders-text-tfhub]
#FOR WINDOWS: pip install vectorhub[encoders-text-tfhub-windows]
from vectorhub.encoders.text.tfhub import Elmo2Vec
model = Elmo2Vec()
model.encode("I enjoy taking long walks along the beach with my dog.")
```
