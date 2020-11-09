from datetime import date
from ....import_utils import is_all_dependency_installed
from ....models_dict import MODEL_REQUIREMENTS
from ....doc_utils import ModelDefinition
from .bert import Bert2Vec

if is_all_dependency_installed(MODEL_REQUIREMENTS['encoders-text-tfhub-albert']):
    from official.nlp.bert import tokenization

AlbertModelDefinition = ModelDefinition(markdown_filepath='encoders/text/tfhub/albert')
__doc__ = AlbertModelDefinition.create_docs()

class Albert2Vec(Bert2Vec):
    definition = AlbertModelDefinition
    def __init__(self, model_url: str = 'https://tfhub.dev/tensorflow/albert_en_base/1', max_seq_length: int = 128, normalize: bool = True):
        list_of_urls = [
            'https://tfhub.dev/tensorflow/albert_en_base/1',
            'https://tfhub.dev/tensorflow/albert_en_xxlarge/1',
            'https://tfhub.dev/tensorflow/albert_en_large/1',
            'https://tfhub.dev/tensorflow/albert_en_xlarge/1',
        ]
        self.validate_model_url(model_url, list_of_urls)
        self.max_seq_length = max_seq_length
        self.normalize = normalize
        self.model = self.init(model_url)
        self.tokenizer = self.init_tokenizer()

    def init_tokenizer(self):
        sp_model_file = self.model_layer.resolved_object.sp_model_file.asset_path.numpy()
        return tokenization.FullSentencePieceTokenizer(sp_model_file)
