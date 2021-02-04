"""
CodeBert model
"""
from typing import List
from datetime import date
from ....base import catch_vector_errors
from ....doc_utils import ModelDefinition
from ....import_utils import is_all_dependency_installed
from ...text.base import BaseText2Vec

if is_all_dependency_installed('encoders-code-transformers'):
    import torch
    from transformers import RobertaTokenizer, RobertaConfig, RobertaModel

CodeBertModelDefinition = ModelDefinition(markdown_filepath="encoders/code/transformers/codebert")
__doc__ = CodeBertModelDefinition.create_docs()

class Code2Vec(BaseText2Vec):
    definition = CodeBertModelDefinition
    urls = {
        'microsoft/codebert-base': {'vector_length': 768}
    }
    def __init__(self, model_name='microsoft/codebert-base'):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaModel.from_pretrained(model_name)
        self.vector_length = self.urls[model_name]

    @catch_vector_errors
    def encode(self, description: str, code: str=None, pooling_method='mean', truncation=True):
        """
        Pooling method is either pooler_output or mean.
        Notes: if it is mean, we can take the last hidden state and add it to the
        model.
        Args:
            Description: The description of what the code is doing 
            Code: What the code is doing.
            Pooling_method: Pooling method can be either mean or pooled output.
            Truncation: Whether the the sentence should be truncated.
        """
        if pooling_method == 'pooler_output':
            return self.model.forward(**self.tokenizer.encode_plus(
                description, code, return_tensors='pt', truncation=truncation
            ))[pooling_method].detach().numpy().tolist()[0]
        elif pooling_method == 'mean':
            return self._vector_operation(self.model.forward(**self.tokenizer.encode_plus(
                description, code, return_tensors='pt', truncation=truncation
            ))['last_hidden_state'].detach().numpy().tolist(), 'mean', axis=1)[0]

    @catch_vector_errors
    def bulk_encode(self, descriptions: List[str], codes: List[str]=None, pooling_method: str='mean', truncation=True):
        """
        Pooling method is either pooler_output or mean.
        Notes: if it is mean, we can take the last hidden state and add it to the
        model.
        Args:
            Pooling_method: Pooling method can be either mean or pooled output.
            Truncation: Whether the the sentence should be truncated.
        """
        if pooling_method == 'pooler_output':
            return self.model.forward(**self.tokenizer.encode_plus(
                descriptions, codes, return_tensors='pt', truncation=truncation
            ))[pooling_method].detach().numpy().tolist()
        elif pooling_method == 'mean':
            return self._vector_operation(self.model.forward(**self.tokenizer.encode_plus(
                descriptions, codes, return_tensors='pt', truncation=truncation
            ))['last_hidden_state'].detach().numpy().tolist(), 'mean', axis=1)

    @property
    def __name__(self):
        return "codebert"
