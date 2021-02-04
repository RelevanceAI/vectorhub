from datetime import date
from ....base import catch_vector_errors
from ....doc_utils import ModelDefinition
from ....import_utils import *
from ....models_dict import MODEL_REQUIREMENTS
from ..base import BaseImage2Vec
from .resnet import ResnetV12Vec

if is_all_dependency_installed(MODEL_REQUIREMENTS['encoders-image-tfhub-resnet']):
    import tensorflow as tf
    import tensorflow_hub as hub

ResNetV2ModelDefinition = ModelDefinition(markdown_filepath='encoders/image/tfhub/resnetv2')
__doc__ = ResNetV2ModelDefinition.create_docs()

class ResnetV22Vec(ResnetV12Vec):
    definition = ResNetV2ModelDefinition
    urls = {
        'https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4': {'vector_length': 2048}, 

        # 101 layers
        'https://tfhub.dev/google/imagenet/resnet_v2_101/feature_vector/4':{'vector_length': 2048}, 

        # 152 layers
        'https://tfhub.dev/google/imagenet/resnet_v2_152/feature_vector/4':{'vector_length': 2048}, 
    }
    def __init__(self, model_url: str = 'https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4'):
        self.validate_model_url(model_url, self.urls)
        self.init(model_url)
        self.vector_length = 2048

    @property
    def urls(self):
        return {
            'https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4': {'vector_length': 2048}, 

            # 101 layers
            'https://tfhub.dev/google/imagenet/resnet_v2_101/feature_vector/4':{'vector_length': 2048}, 

            # 152 layers
            'https://tfhub.dev/google/imagenet/resnet_v2_152/feature_vector/4':{'vector_length': 2048}, 
        }
