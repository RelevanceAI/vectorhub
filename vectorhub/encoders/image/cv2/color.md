
---
model_id: "image/color"
model_name: "Color"
vector_length: "768"
paper: ""
repo: ""
installation: "pip install vectorhub"
release_date: ""
category: image
short_description: Color Encoder
---

## Description

Extracts the color from an image into vectors. It breaks down the color distribution in images.

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
from vectorhub.encoders.image.cv2 import ColorEncoder
model = ColorEncoder()
vector = model.encode('https://getvectorai.com/assets/hub-logo-with-text.png')
```
