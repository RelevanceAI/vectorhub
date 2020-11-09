---
model_id: 'text/sentence-transformers'
model_name: "Sentence Transformer Models" 
vector_length: 'Depends on model.'
paper: https://arxiv.org/abs/1908.10084
repo: https://github.com/UKPLab/sentence-transformers
release_date: 2019-8-27
installation: pip install vectorhub[encoders-text-sentence-transformers]
---

## Description

These are Sentence Transformer models from sbert.net by UKPLab.

## Example

```
#pip install vectorhub[encoders-text-sentence-transformers]
from vectorhub.encoders.text.sentence_transformers import SentenceTransformer2Vec
model = SentenceTransformer2Vec('bert-base-uncased')
model.encode("I enjoy taking long walks along the beach with my dog.")
```
