import numpy as np
from datetime import date
from ....base import catch_vector_errors
from ....doc_utils import ModelDefinition
from ....import_utils import *
from ..base import BaseImage2Vec
from .mobilenet import MobileNetV12Vec
if is_all_dependency_installed('encoders-image-tfhub'):
    import tensorflow as tf
    import tensorflow_hub as hub

MobileNetV2ModelDefinition = ModelDefinition(markdown_filepath='encoders/image/tfhub/mobilenetv2')

__doc__ = MobileNetV2ModelDefinition.create_docs()

class MobileNetV22Vec(MobileNetV12Vec):
    definition = MobileNetV2ModelDefinition
    urls ={
            # 140 depth
            'https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/4': {"vector_length":1792, "image_dimensions":224},

            # 130 depth
            'https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/feature_vector/4': {"vector_length":1664, "image_dimensions":224},

            # 100 depth
            'https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4': {"vector_length":1280, "image_dimensions":224},
            'https://tfhub.dev/google/imagenet/mobilenet_v2_100_192/feature_vector/4': {"vector_length":1280, "image_dimensions":192},
            'https://tfhub.dev/google/imagenet/mobilenet_v2_100_160/feature_vector/4': {"vector_length":1280, "image_dimensions":160},
            'https://tfhub.dev/google/imagenet/mobilenet_v2_100_128/feature_vector/4': {"vector_length":1280, "image_dimensions":128},
            'https://tfhub.dev/google/imagenet/mobilenet_v2_100_96/feature_vector/4': {"vector_length":1280, "image_dimensions":96},

            # 75 depth
            'https://tfhub.dev/google/imagenet/mobilenet_v2_075_224/feature_vector/4': {"vector_length":1280, "image_dimensions":224},
            'https://tfhub.dev/google/imagenet/mobilenet_v2_075_192/feature_vector/4': {"vector_length":1280, "image_dimensions":192},
            'https://tfhub.dev/google/imagenet/mobilenet_v2_075_160/feature_vector/4': {"vector_length":1280, "image_dimensions":160},
            'https://tfhub.dev/google/imagenet/mobilenet_v2_075_128/feature_vector/4': {"vector_length":1280, "image_dimensions":128},
            'https://tfhub.dev/google/imagenet/mobilenet_v2_075_96/feature_vector/4': {"vector_length":1280, "image_dimensions":96},

            # 50 depth
            'https://tfhub.dev/google/imagenet/mobilenet_v2_050_224/feature_vector/4': {"vector_length":1280, "image_dimensions":224},
            'https://tfhub.dev/google/imagenet/mobilenet_v2_050_192/feature_vector/4': {"vector_length":1280, "image_dimensions":192},
            'https://tfhub.dev/google/imagenet/mobilenet_v2_050_160/feature_vector/4': {"vector_length":1280, "image_dimensions":160},
            'https://tfhub.dev/google/imagenet/mobilenet_v2_050_128/feature_vector/4': {"vector_length":1280, "image_dimensions":128},
            'https://tfhub.dev/google/imagenet/mobilenet_v2_050_96/feature_vector/4': {"vector_length":1280, "image_dimensions":96},

            # 35 depth
            'https://tfhub.dev/google/imagenet/mobilenet_v2_035_224/feature_vector/4': {"vector_length":1280, "image_dimensions":224},
            'https://tfhub.dev/google/imagenet/mobilenet_v2_035_192/feature_vector/4': {"vector_length":1280, "image_dimensions":192},
            'https://tfhub.dev/google/imagenet/mobilenet_v2_035_160/feature_vector/4': {"vector_length":1280, "image_dimensions":160},
            'https://tfhub.dev/google/imagenet/mobilenet_v2_035_128/feature_vector/4': {"vector_length":1280, "image_dimensions":128},
            'https://tfhub.dev/google/imagenet/mobilenet_v2_035_96/feature_vector/4': {"vector_length":1280, "image_dimensions":96},
        }
    def __init__(self, model_url: str = 'https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/4', resize_mode:str="symmetric"):
        self.validate_model_url(model_url, self.urls)
        self.vector_length = self.urls[model_url]["vector_length"]
        self.image_dimensions = self.urls[model_url]["image_dimensions"]
        self.init(model_url)
        self.resize_mode = resize_mode
