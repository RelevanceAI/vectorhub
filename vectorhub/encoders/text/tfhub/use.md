---
model_id: "text/use"
model_name: "USE - Universal Sentence Encoder"
vector_length: "512 (Base model)"
paper: "https://arxiv.org/abs/1803.11175"
repo: "https://tfhub.dev/google/collections/universal-sentence-encoder/1"
installation: "pip install vectorhub[encoders-text-tfhub]"
release_date: "2018-03-29"
category: text
short_description: Greater-than-word length text encoder for semantic search.
---

## Description

We present models for encoding sentences into embedding vectors that specifically target transfer learning to other NLP tasks. The models are efficient and result in accurate performance on diverse transfer tasks. Two variants of the encoding models allow for trade-offs between accuracy and compute resources. For both variants, we investigate and report the relationship between model complexity, resource consumption, the availability of transfer task training data, and task performance. Comparisons are made with baselines that use word level transfer learning via pretrained word embeddings as well as baselines do not use any transfer learning. We find that transfer learning using sentence embeddings tends to outperform word level transfer. With transfer learning via sentence embeddings, we observe surprisingly good performance with minimal amounts of supervised training data for a transfer task. We obtain encouraging results on Word Embedding Association Tests (WEAT) targeted at detecting model bias. Our pre-trained sentence encoding models are made freely available for download and on TF Hub.",

![USE Image](https://www.gstatic.com/aihub/tfhub/universal-sentence-encoder/example-similarity.png)

Image from [Google](https://tfhub.dev/google/universal-sentence-encoder/1).

## Training Corpora 

Wikipedia, Web News, web question-answering pages and discussion forums. 

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
from vectorhub.encoders.text.tfhub import USE2Vec
model = USE2Vec()
model.encode("I enjoy taking long walks along the beach with my dog.")
```
