"""
    Abstract from: https://arxiv.org/abs/1409.4842

    We propose a deep convolutional neural network architecture codenamed "Inception", which was responsible for setting the new state of the art for classification and detection in the ImageNet Large-Scale Visual Recognition Challenge 2014 (ILSVRC 2014). The main hallmark of this architecture is the improved utilization of the computing resources inside the network. This was achieved by a carefully crafted design that allows for increasing the depth and width of the network while keeping the computational budget constant. To optimize quality, the architectural decisions were based on the Hebbian principle and the intuition of multi-scale processing. One particular incarnation used in our submission for ILSVRC 2014 is called GoogLeNet, a 22 layers deep network, the quality of which is assessed in the context of classification and detection.

"""
from datetime import date
from ....base import catch_vector_errors
from ....doc_utils import ModelDefinition
from ....import_utils import *
from ..base import BaseImage2Vec
from .inceptionv1 import InceptionV12Vec

if is_all_dependency_installed('encoders-image-tfhub'):
    import tensorflow as tf
    import tensorflow_hub as hub


InceptionV3ModelDefinition = ModelDefinition(markdown_filepath='encoders/image/tfhub/inceptionv3')


__doc__ = InceptionV3ModelDefinition.create_docs()

class InceptionV32Vec(InceptionV12Vec):
    definition = InceptionV3ModelDefinition
    def __init__(self, model_url: str = 'https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4'):
        list_of_urls = [
            # iNaturalist
            "https://tfhub.dev/google/inaturalist/inception_v3/feature_vector/4"
        ]
        self.init(model_url)
        self.vector_length = 2048
    
    @property
    def urls(self):
        return {
            "https://tfhub.dev/google/inaturalist/inception_v3/feature_vector/4": {'vector_length': 2048}
        }
