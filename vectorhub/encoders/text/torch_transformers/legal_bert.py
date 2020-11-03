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

LegalBertModelDefinition = ModelDefinition(
    model_id="text/legal-bert", 
    model_name="Legal Bert", 
    vector_length=768, 
    description="BERT has achieved impressive performance in several NLP tasks. However, there has been limited investigation on its adaptation guidelines in specialised domains. Here we focus on the legal domain, where we explore several approaches for applying BERT models to downstream legal tasks, evaluating on multiple datasets. Our findings indicate that the previous guidelines for pre-training and fine-tuning, often blindly followed, do not always generalize well in the legal domain. Thus we propose a systematic investigation of the available strategies when applying BERT in specialised domains. These are: (a) use the original BERT out of the box, (b) adapt BERT by additional pre-training on domain-specific corpora, and (c) pre-train BERT from scratch on domain-specific corpora. We also propose a broader hyper-parameter search space when fine-tuning for downstream tasks and we release LEGAL-BERT, a family of BERT models intended to assist legal NLP research, computational law, and legal technology applications.",
    paper="https://arxiv.org/abs/2010.02559", 
    repo="https://huggingface.co/nlpaueb/legal-bert-base-uncased",
    release_date=date(2020,10,6),
    installation="pip install vectorhub[encoders-text-torch-transformers]",
    example="""
    #pip install vectorhub[encoders-text-torch-transformers]
    from vectorhub.encoders.text.torch_transformers import LegalBert2Vec
    model = LegalBert2Vec()
    model.encode("I enjoy taking long walks along the beach with my dog.")
    """
)

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
        return torch.mean(self.model(**self.tokenizer(texts, return_tensors='pt', pad_to_multiple_of=self.tokenizer.model_max_length, 
        truncation=True, padding=True))[0], axis=1).detach().tolist()
