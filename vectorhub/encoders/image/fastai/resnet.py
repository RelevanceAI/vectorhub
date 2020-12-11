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
    from fastai.vision.all import *
    from fastai.torch_basics import *
    from fastai.data.all import *
    from fastai.vision.core import *

FastAIResnetModelDefinition = ModelDefinition(markdown_filepath="encoders/image/fastai/resnet_fastai")
__doc__ = FastAIResnetModelDefinition.create_docs()

class FastAIResnet2Vec(FastAIBase):
    definition = FastAIResnetModelDefinition
    def __init__(self, databunch=None, architecture=resnet34):
        """
            For the FASTAI model, you should be able to use Resnet34, Resnet18, 
        """
        self.databunch = databunch
        self.architecture = architecture
        self._create_learner()
    
    @property
    def extraction_layer(self):
        """
            Here we selected the default to be layer_num 1 to extract the layer with the highest number of dimensions
            after it has been flattened.
        """
        return [self.learn.model[1][1]]
