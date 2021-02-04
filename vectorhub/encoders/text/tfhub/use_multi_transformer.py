import warnings
import numpy as np
from datetime import date
from ....base import catch_vector_errors
from ....doc_utils import ModelDefinition
from ....import_utils import is_all_dependency_installed
from ....models_dict import MODEL_REQUIREMENTS
from ..base import BaseText2Vec
from .use import USE2Vec

if is_all_dependency_installed(MODEL_REQUIREMENTS['encoders-text-tfhub-use-transformer']):
    import tensorflow as tf
    import tensorflow_hub as hub
    import tensorflow_text as text

USEMultiTransformerModelDefinition = ModelDefinition(markdown_filepath='encoders/text/tfhub/use_multi_transformer')

__doc__ = USEMultiTransformerModelDefinition.create_docs()

class USEMultiTransformer2Vec(USE2Vec):
    definition = USEMultiTransformerModelDefinition
    urls = {
        "https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-base/1": {"vector_length": 512},
        "https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-base-br/1": {"vector_length": 512}
    }
    def __init__(self, model_url: str="https://tfhub.dev/google/universal-sentence-encoder-cmlm/en-large/1",
    preprocessor_url="https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"):
        self.validate_model_url(model_url, list(self.urls.keys()))
        self.init(model_url)
        self.vector_length = 512
        self.model_url = model_url
        self.preprocess_url = preprocessor_url
        self.preprocessor = hub.KerasLayer(preprocessor_url)
        self.encoder = hub.KerasLayer(model_url)

    @property
    def preprocessor_urls(self):
        return [
            "https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-preprocess/2"
        ]

    @catch_vector_errors
    def encode(self, text, pooling_strategy='defualt'):
        """
        Pooling strategy can be one of 'pooled_output' or 'default'.
        """
        return self.encoder(self.preprocessor(tf.constant([text])))['default'].numpy().tolist()[0]

    @catch_vector_errors
    def bulk_encode(self, texts, pooling_strategy='default'):
        """
        Bulk encode the texts.
        Pooling strategy can be one of 'pooled_output' or 'default'.
        """
        return self.encoder(self.preprocessor(tf.constant(texts)))['default'].numpy().tolist()

