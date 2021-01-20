---
model_id: "text/longformer" 
model_name: "Longformer" 
vector_length: "768 (default)"
paper: "https://arxiv.org/abs/2004.05150"
repo: "https://huggingface.co/allenai/longformer-base-4096"
release_date: "2020-04-10"
installation: "pip install vectorhub[encoders-text-torch-transformers]"
category: text
short_description: We propose a new attention-based model for long-sequence processing that is able to learn long-term dependencies and process documents of thousands of tokens or longer.
---

## Description

From the abstract of the paper:

Transformer-based models are unable to process long sequences due to their self-attention operation, which scales quadratically with the sequence length. To address this limitation, we introduce the Longformer with an attention mechanism that scales linearly with sequence length, making it easy to process documents of thousands of tokens or longer. Longformer’s attention mechanism is a drop-in replacement for the standard self-attention and combines a local windowed attention with a task motivated global attention. Following prior work on long-sequence transformers, we evaluate Longformer on character-level language modeling and achieve state-of-the-art results on text8 and enwik8. In contrast to most prior work, we also pretrain Longformer and finetune it on a variety of downstream tasks. Our pretrained Longformer consistently outperforms RoBERTa on long document tasks and sets new state-of-the-art results on WikiHop and TriviaQA.

Novelties in the LongFormer paper come from its use of attention pattern:

- Sliding Window: The attention pattern employes fixed-size window attention surrounding each token on both sides.
- Dilated Sliding Window: Similar to CNN dilation, sliding windows can be dilated (i.e. have gaps)
- Global Attention: Certain tokens attend to all tokens (e.g. for classification, global attention is used for CLS)
- Linear Projections for Global Attention: 2 sets of projections are used to compute attention scores of sliding window attention.

## Training Corpora 

- Books corpus 
- English Wikipedia 
- One third of a subset of Realnews dataset with documents longer than 1200 tokens 
- One third of stories corpus

## Example

```python
#pip install vectorhub[encoders-text-torch-transformers]
from vectorhub.encoders.text.torch_transformers import Longformer2Vec
model = Longformer2Vec()
model.encode("I enjoy taking long walks along the beach with my dog.")
```
