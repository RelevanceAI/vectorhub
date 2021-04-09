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
    import requests
    import cv2
    from PIL import Image
    from requests.exceptions import MissingSchema

CLIPModelDefinition = ModelDefinition(markdown_filepath='bi_encoders/text_image/torch/clip')
__doc__ = CLIPModelDefinition.create_docs()

class Clip2Vec(BaseImage2Vec, BaseText2Vec):
    definition = CLIPModelDefinition
    urls = {
        "ViT-B/32": {'vector_length': 512},
        "RN50": {'vector_length': 512}
    }
    def __init__(self, url='ViT-B/32', context_length:int=77):
        self.context_length = context_length
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Note that the preprocess is a callable
        self.model, self.preprocess = clip.load(url, device=self.device)
        self.vector_length = self.urls[url]["vector_length"]
        self.url = url

    def read(self, image_url):
        try:
            return Image.open(requests.get(image_url, stream=True).raw)
        except MissingSchema:
            return Image.open(image_url)

    def preprocess_black_and_white_image(self, x):
        """Pass in after the read function
        """
        x = self.preprocess.transforms[0](x)
        x = self.preprocess.transforms[1](x)
        x = self.preprocess.transforms[3](x)
        x = torch.stack((x, x, x), dim=1)
        x = self.preprocess.transforms[4](x)
        return x

    @catch_vector_errors
    def encode_text(self, text: str):
        if self.device == 'cuda':
            text = clip.tokenize(text, context_length=self.context_length).to(self.device)
            return self.model.encode_text(text).cpu().detach().numpy().tolist()[0]
        elif self.device == 'cpu':
            text = clip.tokenize(text, context_length=self.context_length).to(self.device)
            return self.model.encode_text(text).detach().numpy().tolist()[0]

    def encode_video(self, video_url: str):
        """Encode a video by the first still frame
        """
        cap = cv2.VideoCapture(video_url)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        return self.model.encode_image(image).cpu().detach().numpy().tolist()[0]

    def bulk_encode_text(self, texts: List[str]):
        if self.device == 'cuda':
            tokenized_text = clip.tokenize(texts, context_length=self.context_length).to(self.device)
            return self.model.encode_text(tokenized_text).cpu().detach().numpy().tolist()
        elif self.device == 'cpu':
            tokenized_text = clip.tokenize(texts, context_length=self.context_length).to(self.device)
            return self.model.encode_text(tokenized_text).detach().numpy().tolist()

    def preprocess_image(self, img: str):
        if self.is_greyscale(img):
            return self.preprocess_black_and_white_image(self.read(img))
        return self.preprocess(self.read(img))

    @catch_vector_errors
    def encode_image(self, image_url: str):
        if self.device == 'cpu':
            image = self.preprocess_image(image_url).unsqueeze(0).to(self.device)
            return self.model.encode_image(image).detach().numpy().tolist()[0]
        elif self.device == 'cuda':
            image = self.preprocess_image(image_url).unsqueeze(0).to(self.device)
            return self.model.encode_image(image).cpu().detach().numpy().tolist()[0]

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



class ClipText2Vec(BaseText2Vec):
    definition = CLIPModelDefinition
    urls = {
        "ViT-B/32": {'vector_length': 512},
        "RN50": {'vector_length': 512}
    }
    def __init__(self, url='ViT-B/32', context_length:int=77):
        self.context_length = context_length
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Note that the preprocess is a callable
        self.model, self.preprocess = clip.load(url, device=self.device)
        self.vector_length = self.urls[url]["vector_length"]

    @catch_vector_errors
    def encode(self, text: str):
        if self.device == 'cuda':
            text = clip.tokenize(text, context_length=self.context_length).to(self.device)
            return self.model.encode_text(text).cpu().detach().numpy().tolist()[0]
        elif self.device == 'cpu':
            text = clip.tokenize(text, context_length=self.context_length).to(self.device)
            return self.model.encode_text(text).detach().numpy().tolist()[0]

    def bulk_encode(self, texts: List[str]):
        if self.device == 'cuda':
            tokenized_text = clip.tokenize(texts, context_length=self.context_length).to(self.device)
            return self.model.encode_text(tokenized_text).cpu().detach().numpy().tolist()
        elif self.device == 'cpu':
            tokenized_text = clip.tokenize(texts, context_length=self.context_length).to(self.device)
            return self.model.encode_text(tokenized_text).detach().numpy().tolist()


class ClipImage2Vec(BaseImage2Vec):
    definition = CLIPModelDefinition
    urls = {
        "ViT-B/32": {'vector_length': 512},
        "RN50": {'vector_length': 512}
    }
    def __init__(self, url='ViT-B/32', context_length:int=77):
        self.context_length = context_length
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Note that the preprocess is a callable
        self.model, self.preprocess = clip.load(url, device=self.device)
        self.vector_length = self.urls[url]["vector_length"]

    def read(self, image_url):
        try:
            return Image.open(requests.get(image_url, stream=True).raw)
        except MissingSchema:
            return Image.open(image_url)

    def encode_video(self, video_url: str):
        """Encode a video by the first still frame
        """
        cap = cv2.VideoCapture(video_url)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        return self.model.encode_image(image).cpu().detach().numpy().tolist()[0]

    @catch_vector_errors
    def encode(self, image_url: str):
        if self.device == 'cpu':
            image = self.preprocess(self.read(image_url)).unsqueeze(0).to(self.device)
            return self.model.encode_image(image).detach().numpy().tolist()[0]
        elif self.device == 'cuda':
            image = self.preprocess(self.read(image_url)).unsqueeze(0).to(self.device)
            return self.model.encode_image(image).cpu().detach().numpy().tolist()[0]

    def bulk_encode(self, images: str):
        return [self.encode(x) for x in images]

class ClipVideo2Vec(ClipImage2Vec):
    """Encode a video using an image with CLIP
    """
    def encode(self, video_url: str):
        """Encode a video by the first still frame
        """
        cap = cv2.VideoCapture(video_url)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        return self.model.encode_image(image).cpu().detach().numpy().tolist()[0]

    def bulk_encode(self, video_urls: List[str]):
        return [self.encode(v) for v in video_urls]

