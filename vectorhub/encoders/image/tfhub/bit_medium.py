from datetime import date
from ....base import catch_vector_errors
from ....doc_utils import ModelDefinition
from ....import_utils import *
from ..base import BaseImage2Vec
from .bit import BitSmall2Vec

if is_all_dependency_installed('encoders-image-tfhub'):
    import tensorflow as tf
    import tensorflow_hub as hub
    import traceback

BITMediumModelDefinition = ModelDefinition(markdown_filepath="encoders/image/tfhub/bit_medium")
__doc__ = BITMediumModelDefinition.create_docs()

class BitMedium2Vec(BitSmall2Vec):
    definition = BITMediumModelDefinition
    urls = {
        'https://tfhub.dev/google/bit/m-r50x1/1': {"vector_length":2048},  # 2048 output shape
        'https://tfhub.dev/google/bit/m-r50x3/1': {"vector_length":6144}, # 6144 output shape
        'https://tfhub.dev/google/bit/m-r101x1/1': {"vector_length":2048},  # 2048 output shape
        'https://tfhub.dev/google/bit/m-r101x3/1': {"vector_length":6144},  # 6144 output shape
        'https://tfhub.dev/google/bit/m-r152x4/1': {"vector_length":8192},  # 8192 output shape
    }
    def __init__(self, model_url: str = 'https://tfhub.dev/google/bit/m-r50x1/1'):
        self.validate_model_url(model_url, list(self.urls.keys()))
        self.init(model_url)
        self.vector_length = self.urls[model_url]["vector_length"]

