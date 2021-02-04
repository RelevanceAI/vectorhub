from typing import List, Union
from ..base import BaseText2Vec
from ....base import catch_vector_errors
from ....doc_utils import ModelDefinition
from ....import_utils import *
from ....models_dict import MODEL_REQUIREMENTS
if is_all_dependency_installed(MODEL_REQUIREMENTS['encoders-text-torch-transformers-auto']):
    from transformers import LongformerTokenizer, LongformerModel
    import torch

LongformerModelDefinition = ModelDefinition(markdown_filepath='encoders/text/torch_transformers/torch_longformers.md')

__doc__ = LongformerModelDefinition.create_docs()

class Longformer2Vec(BaseText2Vec):
    definition = LongformerModelDefinition
    urls = {
        'allenai/longformer-base-4096': {'vector_length': 4096},
        'allenai/longformer-large-4096': {'vector_length': 4096}
    }
    def __init__(self, model_name: str = "allenai/longformer-base-4096"):
        self.model = LongformerModel.from_pretrained(model_name)
        self.tokenizer = LongformerTokenizer.from_pretrained(model_name)

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
        raise ValueError(
            "Not a string or a list of strings, please enter valid data type.")

    @catch_vector_errors
    def bulk_encode(self, texts: List[str], pooling_method='mean') -> List[List[float]]:
        """
            Encode multiple sentences using transformers.
            args:
                texts: List[str]
        """
        # We use pad_to_multiple_of as other arguments usually do not work.
        if pooling_method == 'mean':
            return torch.mean(
                self.model(**self.tokenizer(texts, return_tensors='pt', 
                pad_to_multiple_of=self.tokenizer.model_max_length,
                truncation=True, padding=True))[0], axis=1).detach().tolist()
        else:
            raise NotImplementedError
