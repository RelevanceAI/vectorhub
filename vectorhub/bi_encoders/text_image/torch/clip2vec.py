"""Clip2Vec by OpenAI
"""
from datetime import date
from typing import List
from ....doc_utils import ModelDefinition
from ....import_utils import is_all_dependency_installed
from ....base import catch_vector_errors
from ....encoders.image import BaseImage2Vec
from ....encoders.text import BaseText2Vec
if is_all_dependency_installed('clip'):
    import clip
    import torch
    import numpy as np

CLIPModelDefinition = ModelDefinition(markdown_filepath='bi_encoders/text_image/torch/clip')
__doc__ = CLIPModelDefinition.create_docs()

class Clip2Vec(BaseImage2Vec, BaseText2Vec):
    definition = CLIPModelDefinition
    def __init__(self, url='ViT-B/32'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Note that the preprocess is a callable
        self.model, self.preprocess = clip.load(url, device=self.device)
        self.vector_length = self.urls[url]

    @property
    def urls(self):
        return {
            "ViT-B/32": {'vector_length': 512},
            "RN50": {'vector_length': {}}
        }

    def encode_text(self, text: str):
        text = clip.tokenize(text).to(device)
        return self.model.encode_text(text).detach().numpy().tolist()[0]
    
    def bulk_encode(self, texts: List[str]):
        tokenized_text = clip.tokenize(texts).to(device)
        return self.model.encode_text(tokenized_text).detach().numpy().tolist()

    def encode_image(self, image_url: str):
        image = self.preprocess(self.read(image_url)).unsqueeze(0).to(device)
        return self.model.encode_image(image).detach().numpy().tolist()[0]
    
    def bulk_encode_image(self, images: str):
        return [self.encode_image(x) for x in images]
    
    def encode(self, data: str, data_type='image'):
        if data_type == 'image':
            return self.encode_image(data)
        elif data_type == 'text':
            return self.encode_text(data)
        raise ValueError("data_type must be either `image` or `text`")

    def bulk_encode(self, data: str, data_type='image'):
        if data_type == 'image':
            return self.bulk_encode_image(data)
        elif data_type == 'text':
            return self.bulk_encode_text(data)
        raise ValueError("data_type must be either `image` or `text`")
