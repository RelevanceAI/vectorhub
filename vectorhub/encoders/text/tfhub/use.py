from ....base import catch_vector_errors
from ....doc_utils import ModelDefinition
from ....import_utils import *
from ....models_dict import MODEL_REQUIREMENTS
from ..base import BaseText2Vec
from datetime import date
if is_all_dependency_installed(MODEL_REQUIREMENTS['encoders-text-tfhub-use']):
    import tensorflow_hub as hub
    import tensorflow as tf
    if hasattr(tf, 'executing_eagerly'):
        if not tf.executing_eagerly():
            tf.compat.v1.enable_eager_execution()

USEModelDefinition = ModelDefinition(markdown_filepath='encoders/text/tfhub/use.md')

__doc__ = USEModelDefinition.create_docs()

class USE2Vec(BaseText2Vec):
    definition = USEModelDefinition
    # or layer19
    def __init__(self, model_url: str = 'https://tfhub.dev/google/universal-sentence-encoder/4'):
        self.validate_model_url(model_url, list(self.urls.keys()))
        self.init(model_url)
        self.vector_length = 512

    @property
    def urls(self):
        return {
            "https://tfhub.dev/google/universal-sentence-encoder/4": {'vector_length': 512},
            "https://tfhub.dev/google/universal-sentence-encoder-large/5": {'vector_length': 512}
        }

    def init(self, model_url: str):
        self.model_url = model_url
        self.model = hub.load(self.model_url)
        self.model_name = model_url.replace(
            'https://tfhub.dev/google/', '').replace('/', '_')


    @catch_vector_errors
    def encode(self, text):
        return self.model([text]).numpy().tolist()[0]

    # can consider compress in the future
    @catch_vector_errors
    def bulk_encode(self, texts, threads=10, chunks=100):
        return [i for c in self.chunk(texts, chunks) for i in self.model(c).numpy().tolist()]
