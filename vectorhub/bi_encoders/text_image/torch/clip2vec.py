"""Add Clip2Vec
"""

from datetime import date
from typing import List
from ....doc_utils import ModelDefinition
from ....import_utils import is_all_dependency_installed
from ....base import catch_vector_errors
from ....encoders.image import BaseImage2Vec
from ....encoders.text import BaseText2Vec
if is_all_dependency_installed('encoders-text-torch-transformers'):
    import clip
    import torch
    import torch
    from PIL import Image
    import numpy as np

CLIPModelDefinition = ModelDefinition(markdown_filepath='bi_encoders/qa/torch_transformers/dpr')
__doc__ = CLIPModelDefinition.create_docs()


class Clip2Vec(BaseImage2Vec, BaseText2Vec):
    def __init__(self):
        self.device = 'cpu'
        # Note that the preprocess is a callable
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

    @property
    def urls(self):
        return {
            "ViT-B/32": {'vector_length': 512}
        }

    def encode_text(self, text: str):
        text = clip.tokenize(text).to(device)
        return self.model.encode_text(text).detach().numpy().tolist()[0]

    def encode_image(self, image_url: str):
        image = self.preprocess(self.read(image_url)).unsqueeze(0).to(device)
        return self.model.encode_image(image).detach().numpy().tolist()[0]
    
    def encode(self, data: str, data_type='image'):
        if data_type == 'image':
            return self.encode_image(data)
        elif data_type == 'text':
            return self.encode_text(data)
        raise ValueError("data_type must be either `image` or `text`")
