---
model_id: "text/sentence-transformers"
model_name: "Sentence Transformer Models" 
vector_length: "Depends on model."
paper: https://arxiv.org/abs/1908.10084
repo: https://github.com/UKPLab/sentence-transformers
release_date: "2019-08-27"
installation: pip install vectorhub[encoders-text-sentence-transformers]
category: text
short_description: These are Sentence Transformer models from sbert.net by UKPLab.
---

## Description

These are Sentence Transformer models from sbert.net by UKPLab.

## Example

```python
#pip install vectorhub[encoders-text-sentence-transformers]
from vectorhub.encoders.text.sentence_transformers import SentenceTransformer2Vec
model = SentenceTransformer2Vec('distilroberta-base-paraphrase-v1')
model.encode("I enjoy taking long walks along the beach with my dog.")
```
