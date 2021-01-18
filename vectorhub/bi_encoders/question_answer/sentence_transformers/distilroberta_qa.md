---
model_id: 'qa/distilled-roberta-qa'
model_name: "Distilled Roberta QA"
vector_length: "768 (default)"
paper: "https://arxiv.org/abs/1908.10084"
repo: "https://github.com/UKPLab/sentence-transformers"
release_date: "2019-08-27"
installation: "pip install vectorhub[encoders-text-sentence-transformers]"
category: text-text
---

## Description

These are Distilled Roberta QA trained on MSMACRO dataset from sbert.net by UKPLab.

## Example


```
#pip install vectorhub[encoders-text-sentence-transformers]
from vectorhub.encoders.qa.sentence_transformers import DistilRobertaQA2Vec
model = DistilRobertaQA2Vec('bert-base-uncased')
model.encode("I enjoy taking long walks along the beach with my dog.")
```
