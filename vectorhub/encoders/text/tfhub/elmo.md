---
model_id: "text/elmo"
model_name: "Elmo (Embeddings From Language Models)" 
vector_length: "1024 (default)"
paper: "https://arxiv.org/abs/1802.05365"
repo: "https://tfhub.dev/google/elmo/3"
installation: "pip install vectorhub[encoders-text-tfhub]"
release_date: "2020-07-03"
category: text
short_description: ELMo is a deep, character-based, bidirectional language model that learns to embed words in a way that captures their context.
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
## Training Corpora

According to the original paper, this was trained on 1 Billion Word Benchmark. The 1 Billion Word Benchmark consists of English monolingual versions
- Europarl corpus (corpus is extracted from the European parliament)
- News commentary
- News

From this, the following steps were taken to normalize the data: 
- Normalization and tokenisation was performed on using scripts from WMT11 site, slightly augmented to normalize various UTF-8 variants for common punctuation. 
- Duplicate sentences were removed
- Vocabulary was constructed by discording all words with count below 3
- Words outside of vocabulary were mapped to <UNK>
- Sentence order was randomized and data was split into 100 disjoint partitions 
- One random partition was chosen as held-out set 
- Held-out set was randomly shuffled and split into 50 disjoint partitions to be used as development/test data
- One partition partition is never predicted by the language model of the held-out data was used as test data in our experiments
- Out-of-vocabulary rate on the test set was set at 0.28%

## Example

```python
#pip install vectorhub[encoders-text-tfhub]
#FOR WINDOWS: pip install vectorhub[encoders-text-tfhub-windows]
from vectorhub.encoders.text.tfhub import Elmo2Vec
model = Elmo2Vec()
model.encode("I enjoy taking long walks along the beach with my dog.")
```
