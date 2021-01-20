---
# be sure to put quotations
model_id: 'text/albert'
model_name: 'Albert - A Lite Bert'
vector_length: 768 (albert_en_base)
paper: 'https://arxiv.org/abs/1909.11942'
repo: 'https://tfhub.dev/tensorflow/albert_en_base/1'
installation: "pip install vectorhub[encoders-text-tfhub]"
release_date: "2019-09-26"
category: text
short_description: We propose a novel method to reduce the memory consumption of BERT, and show that it improves the scalability of BERT models.
---

## Description

Increasing model size when pretraining natural language representations often results in improved performance on downstream tasks. However, at some point further model increases become harder due to GPU/TPU memory limitations and longer training times. To address these problems, we present two parameter-reduction techniques to lower memory consumption and increase the training speed of BERT. Comprehensive empirical evidence shows that our proposed methods lead to models that scale much better compared to the original BERT. We also use a self-supervised loss that focuses on modeling inter-sentence coherence, and show it consistently helps downstream tasks with multi-sentence inputs. As a result, our best model establishes new state-of-the-art results on the GLUE, RACE, and \squad benchmarks while having fewer parameters compared to BERT-large.

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
from vectorhub.encoders.text.tfhub import Albert2Vec
model = Albert2Vec()
model.encode("I enjoy taking long walks along the beach with my dog.")
```
