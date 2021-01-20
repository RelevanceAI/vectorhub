---
model_id: "image/bit-medium"
model_name: "BiT Medium - Big Transfer, General Visual Representation Learning (Medium)"
vector_length: "2048 (default)"
paper: "https://arxiv.org/abs/1912.11370"
repo: "https://github.com/google-research/big_transfer"
installation: "pip install vectorhub[encoders-image-tfhub]"
release_date: "2019-12-24"
category: image
short_description: BiT achieves strong transfer performance across a wide range of datasets, and does so by applying a simple recipe in a manner that allows for efficient scaling up to very large datasets.
---

## Description

Transfer of pre-trained representations improves sample efficiency and simplifies hyperparameter tuning when training 
deep neural networks for vision. We revisit the paradigm of pre-training on large supervised datasets and fine-tuning the model 
on a target task. We scale up pre-training, and propose a simple recipe that we call Big Transfer (BiT). By combining a few carefully 
selected components, and transferring using a simple heuristic, we achieve strong performance on over 20 datasets. BiT performs well across 
a surprisingly wide range of data regimes -- from 1 example per class to 1M total examples. BiT achieves 87.5% top-1 accuracy on ILSVRC-2012, 
99.4% on CIFAR-10, and 76.3% on the 19 task Visual Task Adaptation Benchmark (VTAB). On small datasets, BiT attains 76.8% on 
ILSVRC-2012 with 10 examples per class, and 97.0% on CIFAR-10 with 10 examples per class. We conduct detailed analysis 
of the main components that lead to high transfer performance.

## Working in Colab

If you are using this in colab and want to save this so you don't have to reload, use: 

```
import os 
os.environ['TFHUB_CACHE_DIR'] = "drive/MyDrive/"
os.environ["TFHUB_MODEL_LOAD_FORMAT"] = "COMPRESSED"
```

## Example

```python
#pip install vectorhub[encoders-image-tfhub]
from vectorhub.encoders.image.tfhub import BitMedium2Vec
model = BitMedium2Vec()
sample = model.read('https://getvectorai.com/assets/hub-logo-with-text.png')
model.encode(sample)
```
