from typing import List
from ..base import BaseTextText2Vec
from ....base import catch_vector_errors
from ....doc_utils import ModelDefinition
from ....import_utils import *
from ....models_dict import MODEL_REQUIREMENTS
if is_all_dependency_installed(MODEL_REQUIREMENTS['encoders-text-sentence-transformers']):
    from sentence_transformers import SentenceTransformer


DistilRobertaModelDefinition = ModelDefinition(
    model_name="Distilled Roberta QA", 
    vector_length=768, 
    description="These are Distilled Roberta QA trained on MSMACRO dataset from sbert.net by UKPLab.",
    paper="https://arxiv.org/abs/1908.10084", 
    repo="https://github.com/UKPLab/sentence-transformers",
    installation="pip install vectorhub[encoders-text-sentence-transformers]",
    example="""
    #pip install vectorhub[encoders-text-sentence-transformers]
    from vectorhub.encoders.text_text.sentence_transformers import DistilRobertaQA2Vec
    model = DistilRobertaQA2Vec('bert-base-uncased')
    model.encode("I enjoy taking long walks along the beach with my dog.")
    """
)

__doc__ = DistilRobertaModelDefinition.create_docs()


class DistilRobertaQA2Vec(BaseTextText2Vec):
    definition = DistilRobertaModelDefinition
    def __init__(self):
        self.model = SentenceTransformer('distilroberta-base-msmarco-v1')
        self.vector_length = 768
    
    @catch_vector_errors
    def encode_question(self, question: str):
        return self.model("[QRY] "+ question).tolist()

    @catch_vector_errors
    def bulk_encode_question(self, questions: list):
        return [self.encode(q) for q in questions]
    
    @catch_vector_errors
    def encode_answer(self, answer: str, context: str=None):
        return self.model("[DOC] "+ answer).tolist()

    @catch_vector_errors
    def bulk_encode_answers(self, answers: List[str]):
        return [self.encode(a) for a in answers]