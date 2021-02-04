"""
    Abstract from: https://arxiv.org/abs/1409.4842

    We propose a deep convolutional neural network architecture codenamed "Inception", which was responsible for setting the new state of the art for classification and detection in the ImageNet Large-Scale Visual Recognition Challenge 2014 (ILSVRC 2014). The main hallmark of this architecture is the improved utilization of the computing resources inside the network. This was achieved by a carefully crafted design that allows for increasing the depth and width of the network while keeping the computational budget constant. To optimize quality, the architectural decisions were based on the Hebbian principle and the intuition of multi-scale processing. One particular incarnation used in our submission for ILSVRC 2014 is called GoogLeNet, a 22 layers deep network, the quality of which is assessed in the context of classification and detection.

"""
from datetime import date
from ....base import catch_vector_errors
from ....doc_utils import ModelDefinition
from ....import_utils import *
from ..base import BaseImage2Vec

if is_all_dependency_installed('encoders-image-tfhub'):
    import tensorflow as tf
    import tensorflow_hub as hub

InceptionModelDefinition = ModelDefinition(markdown_filepath='encoders/image/tfhub/inceptionv1')

__doc__ = InceptionModelDefinition.create_docs()

class InceptionV12Vec(BaseImage2Vec):
    definition = InceptionModelDefinition
    urls = {
           'https://tfhub.dev/google/imagenet/inception_v1/feature_vector/4': {'vector_length': 1024}
        }

    def __init__(self, model_url: str = 'https://tfhub.dev/google/imagenet/inception_v1/feature_vector/4'):
        self.init(model_url)
        self.vector_length = 1024
    
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
        return [self.encode(x) for x in images]
