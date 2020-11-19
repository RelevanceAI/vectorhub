---
model_id: "text/labse"
model_name: "LaBSE - Language-agnostic BERT Sentence Embedding" 
vector_length: "768 (default)"
paper: "https://arxiv.org/pdf/2007.01852v1.pdf"
repo: "https://tfhub.dev/google/LaBSE/1"
installation: "pip install vectorhub[encoders-text-tfhub]"
release_date: "2020-07-03"
category: text
---

## Description

The language-agnostic BERT sentence embedding encodes text into high dimensional vectors. The model is trained and optimized to produce similar representations exclusively for bilingual sentence pairs that are translations of each other. So it can be used for mining for translations of a sentence in a larger corpus.

## Example

```python
#pip install vectorhub[encoders-text-tfhub]
#FOR WINDOWS: pip install vectorhub[encoders-text-tfhub-windows]
from vectorhub.encoders.text.tfhub import LaBSE2Vec
model = LaBSE2Vec()
model.encode("I enjoy taking long walks along the beach with my dog.")
```
