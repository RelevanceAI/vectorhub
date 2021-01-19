from datetime import date
from typing import List
from ....base import catch_vector_errors
from ....doc_utils import ModelDefinition
from ....import_utils import *
from ....models_dict import MODEL_REQUIREMENTS
from ..base import BaseQA2Vec
if is_all_dependency_installed(MODEL_REQUIREMENTS['text-bi-encoder-tfhub-lareqa-qa']):
    import bert
    import numpy as np
    import tensorflow.compat.v2 as tf
    import tensorflow_hub as hub
    from tensorflow.python.framework.errors_impl import NotFoundError
    try:
        import tensorflow_text
    except NotFoundError:
        print('The installed Tensorflow Text version is not aligned with tensorflow, make sure that tensorflow-text version is same version as tensorflow')

LAReQAModelDefinition = ModelDefinition(markdown_filepath='bi_encoders/qa/tfhub/lareqa_qa')
__doc__ = LAReQAModelDefinition.create_docs()

class LAReQA2Vec(BaseQA2Vec):
    definition = LAReQAModelDefinition
    def __init__(self, model_url='https://tfhub.dev/google/LAReQA/mBERT_En_En/1', 
    vector_length=512):
        list_of_urls = [
            "https://tfhub.dev/google/LAReQA/mBERT_En_En/1",
            "https://tfhub.dev/google/LAReQA/mBERT_X_X/1",
            "https://tfhub.dev/google/LAReQA/mBERT_X_Y/1",
            "https://tfhub.dev/google/LAReQA/mBERT_X_X_mono/1",
        ]
        self.validate_model_url(model_url, list_of_urls)
        self.model_url = model_url
        self.model = hub.load(self.model_url)
        self.model_name = model_url.replace(
            'https://tfhub.dev/google/', '').replace('/', '_')
        self.vector_length = vector_length
        self.question_encoder = self.model.signatures["query_encoder"]
        self.answer_encoder = self.model.signatures['response_encoder']


    @property
    def __name__(self):
        return "lareqa_qa"

    @catch_vector_errors
    def encode_question(self, question: str):
        """
            Encode the question using LAReQA model.
            Example:
            
            >>> from vectorhub.bi_encoders.qa.tfhub.lareqa_qa import *
            >>> model = LAReQA2Vec()
            >>> model.encode_question("Why?")
        """
        return self.question_encoder(input=tf.constant(np.asarray([question])))["outputs"][0].numpy().tolist()

    @catch_vector_errors
    def bulk_encode_question(self, questions: list):
        """
            Encode questions using LAReQA model.
            Example:
            
            >>> from vectorhub.bi_encoders.qa.tfhub.lareqa_qa import *
            >>> model = LAReQA2Vec()
            >>> model.encode_question(["Why?", "Who?"])
        """
        return self.question_encoder(input=tf.constant(np.asarray(questions)))["outputs"].numpy().tolist()
    
    @catch_vector_errors
    def encode_answer(self, answer: str, context: str=None):
        """
            Encode answer using LAReQA model.
            Example:
            
            >>> from vectorhub.bi_encoders.qa.tfhub.lareqa_qa import *
            >>> model = LAReQA2Vec()
            >>> model.encode_answer("Why?")
        """
        if context is None:
            context = answer
        return self.answer_encoder(
            input=tf.constant(np.asarray([answer])),
            context=tf.constant(np.asarray([context])))["outputs"][0].numpy().tolist()

    @catch_vector_errors
    def bulk_encode_answers(self, answers: List[str], contexts: List[str]=None):
        if contexts is None:
            contexts = answers
        return self.answer_encoder(
            input=tf.constant(np.asarray(answers)),
            context=tf.constant(np.asarray(contexts)))["outputs"].numpy().tolist()

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
            >>> model = LAReQA2Vec()
            >>> model.encode_answer("Why?")
        """
        if string_type.lower() == 'answer':
            return self.encode_answer(string, context=context_string)
        elif string_type.lower() == 'question':
            return self.encode_question(string, context=context_string)

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
            >>> model = LAReQA2Vec()
            >>> model.bulk_encode("Why?", string_type='answer')
        """
        if context_strings is not None:
            return [self.encode(x, context_strings[i], string_type=string_type) for i, x in enumerate(strings)]
        return [self.encode(x, string_type=string_type) for i, x in enumerate(strings)]
    
