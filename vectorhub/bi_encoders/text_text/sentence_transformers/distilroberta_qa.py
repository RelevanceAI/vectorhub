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
