import warnings
from typing import List
from datetime import date
from ....base import catch_vector_errors
from ....doc_utils import ModelDefinition
from ....import_utils import *
from ....models_dict import MODEL_REQUIREMENTS
from ..base import BaseText2Vec
if is_all_dependency_installed(MODEL_REQUIREMENTS['encoders-text-sentence-transformers']):
    from sentence_transformers import SentenceTransformer
    from sentence_transformers import SentenceTransformer
    from sentence_transformers import models, datasets, losses
    import gzip
    from torch.utils.data import DataLoader
    import numpy as np
    import pandas as pd
    from tqdm.auto import tqdm
    import nltk

SentenceTransformerModelDefinition = ModelDefinition(markdown_filepath='encoders/text/sentence_transformers/sentence_auto_transformers.md')

LIST_OF_URLS = {
    'distilroberta-base-paraphrase-v1' : {"vector_length": 768},
    'xlm-r-distilroberta-base-paraphrase-v1' : {"vector_length": 768},
    "paraphrase-xlm-r-multilingual-v1": {"vector_length": 768},

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
    'average_word_embeddings_glove.840B.300d' : {"vector_length": 300},
    'paraphrase-xlm-r-multilingual-v1': {"vector_length": 768},
}

__doc__ = SentenceTransformerModelDefinition.create_docs()


class SentenceTransformer2Vec(BaseText2Vec):
    definition = SentenceTransformerModelDefinition
    urls = LIST_OF_URLS
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.urls = LIST_OF_URLS
        self.validate_model_url(model_name, LIST_OF_URLS)
        if model_name in LIST_OF_URLS:
            self.vector_length = LIST_OF_URLS[model_name]["vector_length"]
        else:
            self.vector_length = None
            warnings.warn("Not included in the official model repository. Please specify set the vector length attribute.")
        self.model = SentenceTransformer(model_name)

    def get_list_of_urls(self):
        """
            Return list of URLS.
        """
        return self.urls

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

    def run_tsdae_on_documents(self, fields, documents, batch_size=32, 
        learning_rate: float=3e-5, num_epochs: int=1, 
        model_output_path: str='.', weight_decay: int=0,
        use_amp: bool=True, scheduler: str='constantlr', temp_filepath = "./_temp.txt"):
        """
Set use_amp to True if your GPU supports FP16 cores
        """
        text = ""
        for c in self.chunk(documents):
            text += self.get_fields_across_document(fields, c)
        with open(temp_filepath, "w") as f:
            f.write(text)
        self.run_tsdae(temp_filepath, batch_size=32, 
            learning_rate=3e-5, num_epochs=1, 
            model_output_path='.', weight_decay=0,
            use_amp=True, scheduler='constantlr')

    def run_tsdae(self, filepath: str, batch_size=32, 
        learning_rate: float=3e-5, num_epochs: int=1, 
        model_output_path: str='.', weight_decay: int=0,
        use_amp: bool=True, scheduler: str='constantlr'):
        """
Set use_amp to True if your GPU supports FP16 cores
        """
        self._create_sentence_transformer()
        train_sentences = self._read_sentences_from_text(filepath)
        train_dataset = datasets.DenoisingAutoEncoderDataset(train_sentences)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        train_loss = losses.DenoisingAutoEncoderLoss(self.model, decoder_name_or_path=self.model_name, tie_encoder_decoder=True)
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=num_epochs,
            weight_decay=weight_decay,
            scheduler=scheduler,
            optimizer_params={'lr': learning_rate},
            show_progress_bar=True,
            checkpoint_path=model_output_path,
            use_amp=use_amp 
        )
        print("Finished training. You can now encode.")

    def _create_sentence_transformer(self):
        word_embedding_model = models.Transformer(self.model_name)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), 'cls')
        self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    def _read_sentences_from_text(self, filepath: str,
        minimum_line_length: int=10):
        train_sentences = []
        with gzip.open(filepath, 'rt', encoding='utf8') if filepath.endswith('.gz') else open(filepath, encoding='utf8') as fIn:
            for line in tqdm(fIn, desc='Read file'):
                line = line.strip()
                if len(line) >= minimum_line_length:
                    train_sentences.append(line)
        return train_sentences
