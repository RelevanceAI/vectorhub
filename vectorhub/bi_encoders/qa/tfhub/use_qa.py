from typing import List
from ..base import BaseQA2Vec
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

USEQAModelDefinition = ModelDefinition(markdown_filepath='bi_encoders/qa/tfhub/use_qa')
__doc__ = USEQAModelDefinition.create_docs()

class USEQA2Vec(BaseQA2Vec):
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

    @catch_vector_errors
    def encode(self, string: str, context_string: str=None, string_type: str='answer'):
        """
            Encode question/answer using LAReQA model.
            Args:
                String: Any string 
                Context_string: The context of the string.
                string_type: question/answer. 

            Example:
            >>> from vectorhub.bi_encoders.qa.tfhub.lareqa_qa import *
            >>> model = USEQA2Vec()
            >>> model.encode_answer("Why?")
        """
        if string_type.lower() == 'answer':
            return self.encode_answer(string, context=context_string)
        elif string_type.lower() == 'question':
            return self.encode_question(string)

    @catch_vector_errors
    def bulk_encode(self, strings: List[str], context_strings: List[str]=None, string_type: str='answer'):
        """
            Bulk encode question/answer using LAReQA model.
            Args:
                String: List of strings.
                Context_string: List of context of the strings.
                string_type: question/answer.

            Example:
            >>> from vectorhub.bi_encoders.qa.tfhub.lareqa_qa import *
            >>> model = USEQA2Vec()
            >>> model.bulk_encode("Why?", string_type='answer')
        """
        if context_strings is not None:
            return [self.encode(x, context_strings[i], string_type=string_type) for i, x in enumerate(strings)]
        return [self.encode(x, string_type=string_type) for i, x in enumerate(strings)]
    
    @property
    def __name__(self):
        return "use_qa"
