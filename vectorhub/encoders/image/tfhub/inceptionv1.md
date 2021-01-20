---
model_id: image/inceptionv1
model_name: Inception V1
vector_length: "1024 (default)"
paper: "https://arxiv.org/abs/1409.4842"
repo: 'https://tfhub.dev/google/imagenet/inception_v1/feature_vector/4'
installation: "pip install vectorhub[encoders-image-tfhub]"
release_date: "2014-09-17"
category: image
short_description: A deeper and wider neural network architecture, codenamed Inception, which is responsible for setting the state of the art for classification and detection in the ImageNet Large-Scale Visual Recognition Challenge 2014.
---

## Description

We propose a deep convolutional neural network architecture codenamed "Inception", which was responsible for setting the new state of the art for classification and detection in the ImageNet Large-Scale Visual Recognition Challenge 2014 (ILSVRC 2014). The main hallmark of this architecture is the improved utilization of the computing resources inside the network. This was achieved by a carefully crafted design that allows for increasing the depth and width of the network while keeping the computational budget constant. To optimize quality, the architectural decisions were based on the Hebbian principle and the intuition of multi-scale processing. One particular incarnation used in our submission for ILSVRC 2014 is called GoogLeNet, a 22 layers deep network, the quality of which is assessed in the context of classification and detection.

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
from vectorhub.encoders.image.tfhub import InceptionV12Vec
model = InceptionV22Vec()
sample = model.read('https://getvectorai.com/assets/hub-logo-with-text.png')
model.encode(sample)
```
