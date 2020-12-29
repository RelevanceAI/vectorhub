import warnings
import numpy as np
from datetime import date
from ....base import catch_vector_errors
from ....doc_utils import ModelDefinition
from ....import_utils import is_all_dependency_installed
from ....models_dict import MODEL_REQUIREMENTS
from ..base import BaseText2Vec
from .use import USE2Vec

if is_all_dependency_installed(MODEL_REQUIREMENTS['encoders-text-tfhub-use-multi']):
    import tensorflow as tf
    if hasattr(tf, 'executing_eagerly'):
        if not tf.executing_eagerly():
            tf.compat.v1.enable_eager_execution()
    from tensorflow.python.framework.errors_impl import NotFoundError
    try:
        import tensorflow_text
    except NotFoundError:
        print('The installed Tensorflow Text version is not aligned with tensorflow, make sure that tensorflow-text version is same version as tensorflow')

USEMultiModelDefinition = ModelDefinition(markdown_filepath='encoders/text/tfhub/use_multi')

__doc__ = USEMultiModelDefinition.create_docs()

class USEMulti2Vec(USE2Vec):
    definition = USEMultiModelDefinition
    def __init__(self, model_url: str = 'https://tfhub.dev/google/universal-sentence-encoder-multilingual/3'):
        self.validate_model_url(model_url, list(self.urls.keys()))
        self.init(model_url)
        self.vector_length = 512

    @property
    def urls(self):
        return {
            "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3": {'vector_length': 512},
            "https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3": {'vector_length': 512}
        }
