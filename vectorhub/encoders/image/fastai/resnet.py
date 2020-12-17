"""
FastAI Resnet model
"""

from datetime import date
from ....base import catch_vector_errors
from ....doc_utils import ModelDefinition
from ....import_utils import is_all_dependency_installed
from ..base import BaseImage2Vec
from .base import FastAIBase

if is_all_dependency_installed('encoders-image-fastai'):
    from fastai.vision.all import (resnet18, resnet34, resnet50, resnet101, resnet152,
        squeezenet1_0, squeezenet1_1, densenet121, vgg16_bn, alexnet)

FastAIResnetModelDefinition = ModelDefinition(markdown_filepath="encoders/image/fastai/resnet_fastai")
__doc__ = FastAIResnetModelDefinition.create_docs()

class FastAIResnet2Vec(FastAIBase):
    definition = FastAIResnetModelDefinition
    def __init__(self, architecture='resnet34', databunch=None):
        """
            Refer to possible_architectures method for reference to which architectures can be instantiated. 
            Args:
                Architecture: The name of the architecture 
                Databunch: A FastAI Data collection data type that is used to instantiate a learner object.
        """
        self.databunch = databunch
        self.architecture = self.architecture_mappings[architecture]
        self._create_learner()
    
    @property
    def possible_architectures(self):
        return list(self.architecture_mappings.keys())

    @property
    def architecture_mappings(self):
        """
            Architecture mappings
        """
        return {
            'resnet18': resnet18,
            'resnet34': resnet34,
            'resnet50': resnet50,
            'resnet101': resnet101,
            'resnet152': resnet152,
            'squeezenet1_0': squeezenet1_0,
            'squeezenet1_1': squeezenet1_1,
            'densenet121': densenet121,
            'vgg16_bn': vgg16_bn,
            'alexnet': alexnet
        }

    @property
    def extraction_layer(self):
        """
            Here we selected the default to be layer_num 1 to extract the layer with the highest number of dimensions
            after it has been flattened.
        """
        return [self.learn.model[1][1]]
