---
model_id: "text/labse"
model_name: "LaBSE - Language-agnostic BERT Sentence Embedding" 
vector_length: "768 (default)"
paper: "https://arxiv.org/pdf/2007.01852v1.pdf"
repo: "https://tfhub.dev/google/LaBSE/1"
installation: "pip install vectorhub[encoders-text-tfhub]"
release_date: "2020-07-03"
category: text
short_description: We present a multilingual BERT embedding model, called LaBSE, that produces language-agnostic cross-lingual sentence embeddings for 109 languages.
---

## Description

The language-agnostic BERT sentence embedding encodes text into high dimensional vectors. The model is trained and optimized to produce similar representations exclusively for bilingual sentence pairs that are translations of each other. So it can be used for mining for translations of a sentence in a larger corpus.
In “Language-agnostic BERT Sentence Embedding”, we present a multilingual BERT embedding model, called LaBSE, that produces language-agnostic cross-lingual sentence embeddings for 109 languages. The model is trained on 17 billion monolingual sentences and 6 billion bilingual sentence pairs using MLM and TLM pre-training, resulting in a model that is effective even on low-resource languages for which there is no data available during training. Further, the model establishes a new state of the art on multiple parallel text (a.k.a. bitext) retrieval tasks. We have released the pre-trained model to the community through tfhub, which includes modules that can be used as-is or can be fine-tuned using domain-specific data.

## Working in Colab

If you are using this in colab and want to save this so you don't have to reload, use: 

```
import os 
os.environ['TFHUB_CACHE_DIR'] = "drive/MyDrive/"
os.environ["TFHUB_MODEL_LOAD_FORMAT"] = "COMPRESSED"
```

## Training Corpora 

LABSE has 2 types of data:
- Monolingual data (CommonCrawl and Wikipedia)
- Bilingual translation pairs (translation corpus is constructed from webpages using a bitext mining system)

The extracted sentence pairs are filtered by a pre-trained contrastive data-selection (CDS) scoring model.
Human annotators manually evaluate sentence pairs from a small sub-set of the harvested pairs and mark the pairs as either "GOOD" or "BAD" translations, from which 80% of the retrained pairs from the manual are rated as "GOOD".

## Training Setup

Short lines less than 10 characters and long lines more than 5000 characters are removed.
Wiki data was extracted from the 05-21-2020 dump using WikiExtractor.


## Example

```python
#pip install vectorhub[encoders-text-tfhub]
#FOR WINDOWS: pip install vectorhub[encoders-text-tfhub-windows]
from vectorhub.encoders.text.tfhub import LaBSE2Vec
model = LaBSE2Vec()
model.encode("I enjoy taking long walks along the beach with my dog.")
```
