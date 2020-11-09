from typing import List
from ..base import BaseText2Vec
from ....base import catch_vector_errors
from ....doc_utils import ModelDefinition
from ....import_utils import *
from ....models_dict import MODEL_REQUIREMENTS
if is_all_dependency_installed(MODEL_REQUIREMENTS['encoders-text-tf-transformers-auto']):
    import tensorflow as tf
    from transformers import AutoTokenizer, TFAutoModel

TransformerModelDefinition = ModelDefinition(markdown_filepath='encoders/text/tf_transformers/tf_auto_transformers.md')

__doc__ = TransformerModelDefinition.create_docs()


class TFTransformer2Vec(BaseText2Vec):
    definition = TransformerModelDefinition
    def __init__(self, model_name: str, config=None):
        if config is None:
            self.model = TFAutoModel.from_pretrained(model_name)
        else:
            self.model = TFAutoModel.from_pretrained(model_name, config=config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    @catch_vector_errors
    def encode(self, text: str) -> List[float]:
        """
            Encode word from transformers.
            This takes the beginning set of tokens and turns them into vectors
            and returns mean pooling of the tokens.
            Args:
                word: string 
        """
        return tf.reduce_mean(self.model(self.tokenizer(text, return_tensors='tf'))[0], axis=1).numpy().tolist()[0]
    
    @catch_vector_errors
    def bulk_encode(self, texts: List[str]) -> List[List[float]]:
        """
            Bulk encode words from transformers.
        """
        return tf.reduce_mean(self.model(self.tokenizer(texts, return_tensors='tf', truncation=True, padding=True))[0], axis=1).numpy().tolist()
