from datetime import date
from ....base import catch_vector_errors
from ....doc_utils import ModelDefinition
from ....import_utils import *
from ..base import BaseImage2Vec

if is_all_dependency_installed('encoders-image-tfhub'):
    import tensorflow as tf
    import tensorflow_hub as hub

BITModelDefinition = ModelDefinition(markdown_filepath='encoders/image/tfhub/bit')

__doc__ = BITModelDefinition.create_docs()

class BitSmall2Vec(BaseImage2Vec):
    definition = BITModelDefinition
    urls = {
        'https://tfhub.dev/google/bit/s-r50x1/1': {"vector_length":2048}, # 2048 output shape
        'https://tfhub.dev/google/bit/s-r50x3/1': {"vector_length":6144},   # 6144 output shape
        'https://tfhub.dev/google/bit/s-r101x1/1': {"vector_length":2048},  # 2048 output shape
        'https://tfhub.dev/google/bit/s-r101x3/1': {"vector_length":6144},  # 6144 output shape
        'https://tfhub.dev/google/bit/s-r152x4/1': {"vector_length":8192},   # 8192 output shape
    }

    def __init__(self, model_url: str = "https://tfhub.dev/google/bit/s-r50x1/1"):
        self.validate_model_url(model_url, list(self.urls.keys()))
        self.init(model_url)
        self.vector_length = self.urls[model_url]["vector_length"]

    def init(self, model_url: str):
        self.model_url = model_url
        self.model_name = self.model_url.replace(
            'https://tfhub.dev/google/', '').replace('/', '_')
        self.model = hub.load(self.model_url)

    @catch_vector_errors
    def encode(self, image):
        if isinstance(image, str):
            image = self.read(image)
        return self.model([image]).numpy().tolist()[0]

    @catch_vector_errors
    def bulk_encode(self, images):
        """
            Bulk encode. Chunk size should be specified outside of the images.
        """
        # TODO: Change from list comprehension to properly read
        return [self.encode(x) for x in images]
