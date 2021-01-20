---
model_id: "image/inception-resnet"
model_name: "Inception Resnet"
vector_length: "1536 (default)"
paper: "https://arxiv.org/abs/1602.07261"
repo: "https://tfhub.dev/google/imagenet/inception_resnet_v2/feature_vector/4"
installation: "pip install vectorhub[encoders-image-tfhub]"
release_date: "2016-02-23"
category: image
short_description: Residual connections improve the performance of deep convolutional neural networks, especially for large networks, and they improve the performance of Inception networks more than other deep network architectures.
---

## Description

Very deep convolutional networks have been central to the largest advances in image recognition performance in 
recent years. One example is the Inception architecture that has been shown to achieve very good performance at 
relatively low computational cost. Recently, the introduction of residual connections in conjunction with a more traditional 
architecture has yielded state-of-the-art performance in the 2015 ILSVRC challenge; its performance was similar to the latest 
generation Inception-v3 network. This raises the question of whether there are any benefit in combining the Inception architecture 
with residual connections. Here we give clear empirical evidence that training with residual connections accelerates the training 
of Inception networks significantly. There is also some evidence of residual Inception networks outperforming similarly expensive 
Inception networks without residual connections by a thin margin. We also present several new streamlined architectures for both 
residual and non-residual Inception networks. These variations improve the single-frame recognition performance on the ILSVRC 2012 
classification task significantly. We further demonstrate how proper activation scaling stabilizes the training of very wide residual 
Inception networks. With an ensemble of three residual and one Inception-v4, we achieve 3.08 percent top-5 error on the test set of the 
ImageNet classification (CLS) challenge.

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
from vectorhub.encoders.image.tfhub import InceptionResnet2Vec
model = InceptionResnet2Vec()
sample = model.read('https://getvectorai.com/assets/hub-logo-with-text.png')
model.encode(sample)
```
