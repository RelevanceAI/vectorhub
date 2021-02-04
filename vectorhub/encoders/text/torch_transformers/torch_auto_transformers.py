from typing import List, Union
from ..base import BaseText2Vec
from ....base import catch_vector_errors
from ....doc_utils import ModelDefinition
from ....import_utils import *
from ....models_dict import MODEL_REQUIREMENTS
if is_all_dependency_installed(MODEL_REQUIREMENTS['encoders-text-torch-transformers-auto']):
    from transformers import AutoTokenizer, AutoModel
    import torch

TransformerModelDefinition = ModelDefinition(markdown_filepath='encoders/text/torch_transformers/torch_auto_transformers.md')

__doc__ = TransformerModelDefinition.create_docs()


def list_tested_transformer_models():
    """
        List the transformed models.
    """
    return [
        "bert-base-uncased",
        "distilbert-base-uncased",
        "facebook/bart-base"
    ]

class Transformer2Vec(BaseText2Vec):
    definition = TransformerModelDefinition
    urls =  {
            "bert-base-uncased": {'vector_length': 768},
            "distilbert-base-uncased": {'vector_length': 768},
            "facebook/bart-base": {'vector_length': 768}
        }

    def __init__(self, model_name: str):
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("Transformer2Vec uses the AutoModel to allow for easier models.")
        print("Therefore, not all models will worked but most do. " + \
            "Call the list of tested transformer models using list_tested_models.")
    
    @catch_vector_errors
    def encode(self, text: Union[str, List[str]]) -> List[float]:
        """
            Encode words using transformers.
            Args:
                word: str

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
                Sentences: List[str]
        """
        # We use pad_to_multiple_of as other arguments usually do not work.
        return torch.mean(self.model(**self.tokenizer(texts, return_tensors='pt', pad_to_multiple_of=self.tokenizer.model_max_length, 
        truncation=True, padding=True))[0], axis=1).detach().tolist()
