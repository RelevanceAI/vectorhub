from typing import List, Union
from ..base import BaseText2Vec
from ....base import catch_vector_errors
from ....doc_utils import ModelDefinition
from ....import_utils import *
from ....models_dict import MODEL_REQUIREMENTS
from datetime import date
if is_all_dependency_installed(MODEL_REQUIREMENTS['encoders-text-torch-transformers-auto']):
    from transformers import AutoTokenizer, AutoModel
    import torch

LegalBertModelDefinition = ModelDefinition(markdown_filepath='encoders/text/torch_transformers/legal_bert')

__doc__ = LegalBertModelDefinition.create_docs()

class LegalBert2Vec(BaseText2Vec):
    definition = LegalBertModelDefinition
    def __init__(self, model_name: str="nlpaueb/legal-bert-base-uncased"):
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    @staticmethod
    def list_possible_models():
        return {
            "nlpaueb/bert-base-uncased-contracts": "Trained on US contracts",
            "nlpaueb/bert-base-uncased-eurlex": "Trained on EU legislation", 
            "nlpaueb/bert-base-uncased-echr	": "Trained on ECHR cases",
            "nlpaueb/legal-bert-base-uncased": "Trained on all the above", 
            "nlpaueb/legal-bert-small-uncased": "Trained on all the above"
        }

    @property
    def urls(self):
        return {
            "nlpaueb/bert-base-uncased-contracts": {},
            "nlpaueb/bert-base-uncased-eurlex": {},
            "nlpaueb/bert-base-uncased-echr	": {},
            "nlpaueb/legal-bert-base-uncased": {},
            "nlpaueb/legal-bert-small-uncased": {},
        }

    @catch_vector_errors
    def encode(self, text: Union[str, List[str]]) -> List[float]:
        """
            Encode words using transformers.
            Args:
                text: str
        """
        if isinstance(text, str):
            return torch.mean(self.model(**self.tokenizer(text, return_tensors='pt'))[0], axis=1).detach().tolist()[0]
        if isinstance(text, list):
            return self.bulk_encode(text)
        raise ValueError("Not a string or a list of strings, please enter valid data type.")

    @catch_vector_errors
    def bulk_encode(self, texts: List[str]) -> List[List[float]]:
        """
            Encode multiple sentences using transformers.
            args:
                texts: List[str]
        """
        # We use pad_to_multiple_of as other arguments usually do not work.
        # TODO: FIx the older method
        # return torch.mean(self.model(**self.tokenizer(texts, return_tensors='pt', pad_to_multiple_of=self.tokenizer.model_max_length,
        #     truncation=True, padding=True))[0], axis=1).detach().tolist()
        return [self.encode(x) for x in texts]
