---
model_id: "text/legal-bert" 
model_name: "Legal Bert" 
vector_length: "768 (default)"
paper: "https://arxiv.org/abs/2010.02559"
repo: "https://huggingface.co/nlpaueb/legal-bert-base-uncased"
release_date: "2020-10-06"
installation: "pip install vectorhub[encoders-text-torch-transformers]"
category: text
short_description: We propose a systematic investigation of the available strategies when applying BERT in Legal domains.
---

## Description

BERT has achieved impressive performance in several NLP tasks. However, there has been limited investigation on its adaptation guidelines in specialised domains. Here we focus on the legal domain, where we explore several approaches for applying BERT models to downstream legal tasks, evaluating on multiple datasets. Our findings indicate that the previous guidelines for pre-training and fine-tuning, often blindly followed, do not always generalize well in the legal domain. Thus we propose a systematic investigation of the available strategies when applying BERT in specialised domains. These are: (a) use the original BERT out of the box, (b) adapt BERT by additional pre-training on domain-specific corpora, and (c) pre-train BERT from scratch on domain-specific corpora. We also propose a broader hyper-parameter search space when fine-tuning for downstream tasks and we release LEGAL-BERT, a family of BERT models intended to assist legal NLP research, computational law, and legal technology applications.

## Example

```python
#pip install vectorhub[encoders-text-torch-transformers]
from vectorhub.encoders.text.torch_transformers import LegalBert2Vec
model = LegalBert2Vec()
model.encode("I enjoy taking long walks along the beach with my dog.")
```
