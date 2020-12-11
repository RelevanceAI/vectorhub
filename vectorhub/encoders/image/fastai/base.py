"""
    The base class for FastAI as much of it can be replaced easily by changing the model.
"""
from abc import abstractproperty
from ..base import BaseImage2Vec

# We use wildcard imports for FastAI as this is the way it is handled in the documentation.
from ....import_utils import is_all_dependency_installed

if is_all_dependency_installed('encoders-image-fastai'):
    import torch
    import numpy as np
    from fastai.vision.all import *
    from fastai.torch_basics import *
    from fastai.data.all import *
    from fastai.vision.core import *

class FastAIBase(BaseImage2Vec):
    def __init__(self, databunch=None, architecture=None):
        self.databunch = databunch
        self.architecture = architecture
        self._create_learner(databunch, architecture)
    
    def _instantiate_empty_dataloader(self):
        """
            As it is almost impossible to instantiate an empty dataloader, we use a CIFAR as a dummy.
        """
        path = untar_data(URLs.CIFAR_100)
        files = get_image_files(path/"train")
        return ImageDataLoaders.from_lists(path, files, [0] * len(files))
    
    def _create_learner(self):
        dls = self._instantiate_empty_dataloader()
        self.learn = cnn_learner(dls, resnet34, metrics=error_rate)

    @abstractproperty
    def extraction_layer(self):
        pass

    def encode(self, image):
        with hook_outputs(self.extraction_layer) as h:
            if isinstance(image, str):
                y = self.learn.predict(self.read(image))
            elif isinstance(image, (np.array)):
                y = self.learn.predict(image)
        return h.stored[0].cpu().numpy().tolist()[0]
