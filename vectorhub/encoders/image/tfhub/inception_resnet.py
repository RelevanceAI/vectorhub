from typing import List
from ....base import catch_vector_errors
from ....doc_utils import ModelDefinition
from ....import_utils import is_all_dependency_installed
from ..base import BaseImage2Vec
if is_all_dependency_installed('encoders-image-tfhub'):
    import tensorflow as tf
    import tensorflow_hub as hub
    import traceback

InceptionResnetModelDefinition = ModelDefinition(markdown_filepath='encoders/image/tfhub/inception_resnet')

__doc__ = InceptionResnetModelDefinition.create_docs()

class InceptionResnet2Vec(BaseImage2Vec):
    definition = InceptionResnetModelDefinition
    def __init__(self, model_url="https://tfhub.dev/google/imagenet/inception_resnet_v2/feature_vector/4"):
        self.model_url = model_url
        self.model_name = self.model_url.replace(
            'https://tfhub.dev/google/', '').replace('/', '_')
        self.model = hub.load(self.model_url)
        self.vector_length = 1536

    @property
    def urls(self):
        return {
            "https://tfhub.dev/google/imagenet/inception_resnet_v2/feature_vector/4": {"vector_length": 1536}
        }

    @catch_vector_errors
    def encode(self, image):
        """
        Encode an image using InceptionResnet.

        Example:
            >>> from vectorhub.image.encoder.tfhub import inception_resnet
            >>> model = InceptionResnet2Vec(username, api_key)
            >>> model.encode("Hey!")
        """
        if isinstance(image, str):
            image = self.read(image)
        return self.model([image]).numpy().tolist()[0]
    
    @catch_vector_errors
    def bulk_encode(self, images):
        return [self.encode(x) for x in images]
