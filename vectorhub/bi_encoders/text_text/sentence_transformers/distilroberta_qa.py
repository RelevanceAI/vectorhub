from typing import List
from ..base import BaseTextText2Vec
from ....base import catch_vector_errors
from ....doc_utils import ModelDefinition
from ....import_utils import *
from ....models_dict import MODEL_REQUIREMENTS
from datetime import date
if is_all_dependency_installed(MODEL_REQUIREMENTS['encoders-text-sentence-transformers']):
    from sentence_transformers import SentenceTransformer

DistilRobertaQAModelDefinition = ModelDefinition(markdown_filepath='bi_encoders/text_text/sentence_transformers/distilroberta_qa.md')

__doc__ = DistilRobertaQAModelDefinition.create_docs()


class DistilRobertaQA2Vec(BaseTextText2Vec):
    definition = DistilRobertaQAModelDefinition
    def __init__(self):
        self.model = SentenceTransformer('distilroberta-base-msmarco-v1')
        self.vector_length = 768
    
    @property
    def __name__(self):
        return "distilroberta_qa"

    @catch_vector_errors
    def encode_question(self, question: str):
        return self.model.encode(["[QRY] "+ question])[0].tolist()

    @catch_vector_errors
    def bulk_encode_question(self, questions: list):
        return [self.encode(q) for q in questions]
    
    @catch_vector_errors
    def encode_answer(self, answer: str, context: str=None):
        return self.model.encode(["[DOC] "+ answer])[0].tolist()

    @catch_vector_errors
    def bulk_encode_answers(self, answers: List[str]):
        return [self.encode(a) for a in answers]
    
    @catch_vector_errors
    def encode(self, string: str, context_string: str=None, string_type: str='answer'):
        """
            Encode question/answer using LAReQA model.
            Args:
                String: Any string 
                Context_string: The context of the string.
                string_type: question/answer. 

            Example:
            >>> from vectorhub.bi_encoders.text_text.tfhub.lareqa_qa import *
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
            >>> from vectorhub.bi_encoders.text_text.tfhub.lareqa_qa import *
            >>> model = LAReQA2Vec()
            >>> model.bulk_encode("Why?", string_type='answer')
        """
        if context_strings is not None:
            return [self.encode(x, context_strings[i], string_type=string_type) for i, x in enumerate(strings)]
        return [self.encode(x, string_type=string_type) for x in enumerate(strings)]
