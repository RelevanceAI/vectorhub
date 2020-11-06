import warnings
import numpy as np
from ..base import BaseText2Vec
from ....base import catch_vector_errors
from .use import USE2Vec
from ....doc_utils import ModelDefinition
from ....import_utils import is_all_dependency_installed
from ....models_dict import MODEL_REQUIREMENTS
from datetime import date
if is_all_dependency_installed(MODEL_REQUIREMENTS['encoders-text-tfhub-use-multi']):
    import tensorflow as tf
    import tensorflow_text

USEMultiModelDefinition = ModelDefinition(
    model_id = "text/use-multi",
    model_name="USE Multi - Universal Sentence Encoder Multilingual", 
    vector_length=512, 
    description="The Universal Sentence Encoder Multilingual module is an extension of the Universal Sentence Encoder Large that includes training on multiple tasks across languages. Supports 16 languages (Arabic, Chinese-simplified, Chinese-traditional, English, French, German, Italian, Japanese, Korean, Dutch, Polish, Portuguese, Spanish, Thai, Turkish, Russian) text encoder.",
    paper="https://arxiv.org/abs/1803.11175",
    repo="https://tfhub.dev/google/collections/universal-sentence-encoder/1",
    installation="pip install vectorhub[encoders-text-tfhub]",
    release_date=date(2018,3,29),
    example="""
    #pip install vectorhub[encoders-text-tfhub]
    from vectorhub.encoders.text.tfhub import USEMulti2Vec
    model = USEMulti2Vec()
    model.encode("I enjoy taking long walks along the beach with my dog.")
    """
)

__doc__ = USEMultiModelDefinition.create_docs()

class USEMulti2Vec(USE2Vec):
    definition = USEMultiModelDefinition
    def __init__(self, model_url: str = 'https://tfhub.dev/google/universal-sentence-encoder-multilingual/3'):
        list_of_urls = [
            "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3",
            "https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3",
        ]
        self.validate_model_url(model_url, list_of_urls)
        self.init(model_url)
        self.vector_length = 512
