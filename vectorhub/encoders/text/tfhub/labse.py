from datetime import date
from ....doc_utils import ModelDefinition
from ....import_utils import is_all_dependency_installed
from ....models_dict import MODEL_REQUIREMENTS

if is_all_dependency_installed(MODEL_REQUIREMENTS['encoders-text-tfhub-labse']):
    import tensorflow as tf

LABSEModelDefinition = ModelDefinition(markdown_filepath='encoders/text/tfhub/labse.md')
__doc__ = LABSEModelDefinition.create_docs()


from .bert import Bert2Vec
class LaBSE2Vec(Bert2Vec):
    definition = LABSEModelDefinition
    def __init__(self, model_url: str = 'https://tfhub.dev/google/LaBSE/1', max_seq_length: int = 128, normalize: bool = True):
        self.max_seq_length = max_seq_length
        self.normalize = normalize
        self.model = self.init(model_url)
        self.tokenizer = self.init_tokenizer()
        self.vector_length = 768
