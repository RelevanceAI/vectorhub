from typing import List
from datetime import date
from ....base import catch_vector_errors
from ....doc_utils import ModelDefinition
from ....import_utils import *
from ....models_dict import MODEL_REQUIREMENTS
from ..base import BaseText2Vec
if is_all_dependency_installed(MODEL_REQUIREMENTS['encoders-text-sentence-transformers']):
    from sentence_transformers import SentenceTransformer

SentenceTransformerModelDefinition = ModelDefinition(markdown_filepath='encoders/text/sentence_transformers/sentence_auto_transformers.md')

LIST_OF_URLS = {
    'distilroberta-base-paraphrase-v1' : {"vector_length": 768},
    'xlm-r-distilroberta-base-paraphrase-v1' : {"vector_length": 768},

    'distilbert-base-nli-stsb-mean-tokens' : {"vector_length": 768},
    'bert-large-nli-stsb-mean-tokens' : {"vector_length": 1024},
    'roberta-base-nli-stsb-mean-tokens' : {"vector_length": 768},
    'roberta-large-nli-stsb-mean-tokens' : {"vector_length": 1024},

    'distilbert-base-nli-stsb-quora-ranking' : {"vector_length": 768},
    'distilbert-multilingual-nli-stsb-quora-ranking' : {"vector_length": 768},

    'distilroberta-base-msmarco-v1' : {"vector_length": 768},

    'distiluse-base-multilingual-cased-v2' : {"vector_length": 512},
    'xlm-r-bert-base-nli-stsb-mean-tokens' : {"vector_length": 768},

    'bert-base-wikipedia-sections-mean-tokens' : {"vector_length": 768},

    'LaBSE' : {"vector_length": 768},

    'average_word_embeddings_glove.6B.300d' : {"vector_length": 300},
    'average_word_embeddings_komninos' : {"vector_length": 300},
    'average_word_embeddings_levy_dependency' : {"vector_length": 768},
    'average_word_embeddings_glove.840B.300d' : {"vector_length": 300}
}

__doc__ = SentenceTransformerModelDefinition.create_docs()


class SentenceTransformer2Vec(BaseText2Vec):
    definition = SentenceTransformerModelDefinition
    def __init__(self, model_name: str):
        self.list_of_urls = LIST_OF_URLS
        self.validate_model_url(model_name, LIST_OF_URLS)
        self.vector_length = LIST_OF_URLS[model_name]["vector_length"]
        self.model = SentenceTransformer(model_name)

    def get_list_of_urls(self):
        """
            Return list of URLS.
        """
        return self.list_of_urls

    @catch_vector_errors
    def encode(self, text: str) -> List[float]:
        """
            Encode word from transformers.
            This takes the beginning set of tokens and turns them into vectors
            and returns mean pooling of the tokens.
            Args:
                word: string 
        """
        return self.model.encode([text])[0].tolist()
    
    @catch_vector_errors
    def bulk_encode(self, texts: List[str]) -> List[List[float]]:
        """
            Bulk encode words from transformers.
        """
        return self.model.encode(texts).tolist()
