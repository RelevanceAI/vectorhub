import urllib
from typing import List, Union
from ...base import BaseImage2Vec
from ....base import catch_vector_errors
from ....doc_utils import ModelDefinition
from ....import_utils import *
from ....models_dict import MODEL_REQUIREMENTS
if is_all_dependency_installed(MODEL_REQUIREMENTS['encoders-image-cv2']):
    import cv2
    import numpy as np

CV2ModelDefinition = ModelDefinition(markdown_filepath='encoders/image/cv2/color.md')
__doc__ = CV2ModelDefinition.create_docs()

class ColorEncoder:
    definition = CV2ModelDefinition
    def __init__(self):
        pass

    @property
    def __name__(self):
        return "color"

    def encode(self, image_url, return_rgb: bool=False):
        try:
            req = urllib.request.urlopen(image_url)
            arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
            img = cv2.imdecode(arr, -1) 
        except:
            img = cv2.imread(image_url)
        color = ('b','g','r')
        if return_rgb:
            histr = {}
            for i,col in enumerate(color):
                histr[col] = cv2.calcHist([img],[i],None,[256],[0,256]).T.tolist()[0]
            return histr
        histr = []
        for i,col in enumerate(color):
            histr.extend(cv2.calcHist([img],[i],None,[256],[0,256]).T.tolist()[0])
        return histr
    
    def bulk_encode(self, image_urls: List[str]):
        return [self.encode(image_url) for image_url in image_urls]
