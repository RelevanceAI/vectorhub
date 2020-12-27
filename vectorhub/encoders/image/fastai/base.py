"""
    The base class for FastAI as much of it can be replaced easily by changing the model.
"""
from abc import abstractproperty
from ..base import BaseImage2Vec
from ....import_utils import is_all_dependency_installed
from ....base import catch_vector_errors

if is_all_dependency_installed('encoders-image-fastai'):
    import torch
    import numpy as np
    # We use wildcard imports for FastAI as this is the way it is handled in the documentation.
    from fastai.vision.all import cnn_learner, resnet34, ImageDataLoaders, error_rate, hook_outputs
    from fastai.data.all import get_image_files, untar_data, URLs

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

    @catch_vector_errors
    def encode(self, image):
        with hook_outputs(self.extraction_layer) as h:
            if isinstance(image, str):
                y = self.learn.predict(self.read(image))
            elif isinstance(image, (np.ndarray, np.generic)):
                y = self.learn.predict(image)
        return h.stored[0].cpu().numpy().tolist()[0]
    
    @catch_vector_errors
    def bulk_encode(self, images):
        return [self.encode(x) for x in images]
