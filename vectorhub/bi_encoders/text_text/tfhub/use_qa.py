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

USEQAModelDefinition = ModelDefinition(
    model_id='text_text/use-qa',
    model_name="Universal Sentence Encoder Question Answering",
    vector_length=512,
    description="""
        - Developed by researchers at Google, 2019, v2 [1].
        - It is trained on a variety of data sources and tasks, with the goal of learning text representations that 
        are useful out-of-the-box to retrieve an answer given a question, as well as question and answers across different languages.
        - It can also be used in other applications, including any type of text classification, clustering, etc.
    """,
    release_date=date(2020,3,11),
    repo='https://tfhub.dev/google/universal-sentence-encoder-qa/3',
    installation="pip install vectorhub[encoders-text-tfhub-tftext]",
    example="""
    #pip install vectorhub[encoders-text-tfhub-tftext]
    from vectorhub.bi_encoder.text_text.tfhub import USEQA2Vec
    model = USEQA2Vec()
    model.encode_question('How is the weather today?')
    model.encode_answer('The weather is great today.')
    """
)

__doc__ = USEQAModelDefinition.create_docs()

class USEQA2Vec(BaseTextText2Vec):
    definition = USEQAModelDefinition
    def __init__(self):
        self.model_url = "https://tfhub.dev/google/universal-sentence-encoder-qa/3"
        self.model = hub.load(self.model_url)
        self.model_name = self.model_url.replace(
            'https://tfhub.dev/google/', '').replace('/', '_')
        self.vector_length = 512

    @catch_vector_errors
    def encode_question(self, question: str):
        return self.model.signatures['question_encoder'](tf.constant([question]))['outputs'].numpy().tolist()[0]
    
    @catch_vector_errors
    def bulk_encode_questions(self, questions: List[str]):
        return self.model.signatures['question_encoder'](tf.constant([question]))['outputs'].numpy().tolist()

    @catch_vector_errors
    def encode_answer(self, answer: str, context: str=None):
        if context is None:
            context = answer
        return self.model.signatures['response_encoder'](
            input=tf.constant([answer]),
            context=tf.constant([context]))['outputs'].numpy().tolist()[0]

    @catch_vector_errors
    def bulk_encode_answers(self, answers: List[str], contexts: List[str]=None):
        if contexts is None:
            contexts = answers
        return self.model.signatures['response_encoder'](
            input=tf.constant(answers),
            context=tf.constant(contexts))['outputs'].numpy().tolist()

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
