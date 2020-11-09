from typing import List
from ..base import BaseTextText2Vec
from ....base import catch_vector_errors
from ....doc_utils import ModelDefinition
from ....import_utils import *
from ....models_dict import MODEL_REQUIREMENTS
from datetime import date

if is_all_dependency_installed(MODEL_REQUIREMENTS['text-bi-encoder-tfhub-use-qa']):
    import bert
    import numpy as np
    import tensorflow as tf
    import tensorflow_hub as hub
    import tensorflow_text

USEMultiQAModelDefinition = ModelDefinition(
    model_id='text_text/use-multi-qa',
    model_name="Universal Sentence Encoder Multilingual Question Answering",
    vector_length=512,
    description="""
        - Developed by researchers at Google, 2019, v2 [1].
        - Covers 16 languages, strong performance on cross-lingual question answer retrieval.       
        - It is trained on a variety of data sources and tasks, with the goal of learning text representations that are useful out-of-the-box to retrieve an answer given a question, as well as question and answers across different languages.
        - It can also be used in other applications, including any type of text classification, clustering, etc.
    """,
    repo="https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/3",
    installation="pip install vectorhub[encoders-text-tfhub]",
    example="""
    #pip install vectorhub[encoders-text-tfhub]
    from vectorhub.bi_encoders.text_text.tfhub import USEMultiQA2Vec
    model = USEMultiQA2Vec()
    model.encode_question('How is the weather today?')
    model.encode_answer('The weather is great today.')
    """
)

class USEMultiQA2Vec(USEQA2Vec):
    definition = USEMultiQAModelDefinition
    def __init__(self):
        self.model_url = "https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/3"
        self.model = hub.load(self.model_url)
        self.model_name = self.model_url.replace(
            'https://tfhub.dev/google/', '').replace('/', '_')
        self.vector_length = 512
