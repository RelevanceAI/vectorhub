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
    from tensorflow.python.framework.errors_impl import NotFoundError
    try:
        import tensorflow_text
    except NotFoundError:
        print('The installed Tensorflow Text version is not aligned with tensorflow, make sure that tensorflow-text version is same version as tensorflow')

USEQAModelDefinition = ModelDefinition(markdown_filepath='bi_encoders/text_text/tfhub/use_qa')
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
