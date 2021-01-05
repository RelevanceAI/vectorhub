---
# be sure to put quotations
model_id: "text/bert"
model_name: "BERT - Bidirectional Encoder Representations"
vector_length: 1024 (Bert Large)
paper: "https://arxiv.org/abs/1810.04805v2"
repo: "https://tfhub.dev/google/collections/bert/1"
release_date: '2018-10-11'
installation: "pip install vectorhub[encoders-text-tfhub]"
category: text
---

## Description

We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications.

![Bert Image](https://miro.medium.com/max/619/1*iJqlhZz-g6ZQJ53-rE9VvA.png)

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
from vectorhub.encoders.text.tfhub import Bert2Vec
model = Bert2Vec()
model.encode("I enjoy taking long walks along the beach with my dog.")
```

## Model Versions

Model Table | Vector Length 
------------| ---------- 
google/bert_cased_L-12_H-768_A-12 | 768  
google/bert_cased_L-24_H-1024_A-16 | 1024
google/bert_chinese_L-12_H-768_A-12 | 768
google/bert_multi_cased_L-12_H-768_A-12 | 768
google/bert_uncased_L-12_H-768_A-12 | 768
google/bert_uncased_L-24_H-1024_A-16 | 1024


## Limitations

* NA

## Training Corpora:

* BooksCorpus (800M words)
* Wikipedia (2,500M words)

## Other Notes:

* BERT was trained for 1M steps with a batch size of 128,000 words.


