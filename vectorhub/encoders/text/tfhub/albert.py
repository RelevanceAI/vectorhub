from datetime import date
import traceback
from ..base import BaseText2Vec
from ....base import catch_vector_errors
from ....import_utils import is_all_dependency_installed
from ....models_dict import MODEL_REQUIREMENTS
from ....doc_utils import ModelDefinition

if is_all_dependency_installed(MODEL_REQUIREMENTS['encoders-text-tfhub-albert']):
    from tensorflow.python.framework.errors_impl import NotFoundError
    import tensorflow_hub as hub
    try:
        import tensorflow_text
    except NotFoundError:
        print('The installed Tensorflow Text version is not aligned with tensorflow, make sure that tensorflow-text version is same version as tensorflow')

AlbertModelDefinition = ModelDefinition(markdown_filepath='encoders/text/tfhub/albert')
__doc__ = AlbertModelDefinition.create_docs()

class Albert2Vec(BaseText2Vec):
    definition = AlbertModelDefinition
    def __init__(self, model_url: str = 'https://tfhub.dev/tensorflow/albert_en_base/2', max_seq_length: int = 228, normalize: bool = True, 
        preprocessor_url:str ='http://tfhub.dev/tensorflow/albert_en_preprocess/1'):
        list_of_urls = [
            'https://tfhub.dev/tensorflow/albert_en_base/1',
            'https://tfhub.dev/tensorflow/albert_en_xxlarge/1',
            'https://tfhub.dev/tensorflow/albert_en_large/1',
            'https://tfhub.dev/tensorflow/albert_en_xlarge/1',

            'https://tfhub.dev/tensorflow/albert_en_base/2',
            'https://tfhub.dev/tensorflow/albert_en_xxlarge/2',
            'https://tfhub.dev/tensorflow/albert_en_large/2',
            'https://tfhub.dev/tensorflow/albert_en_xlarge/2',
        ]
        self.validate_model_url(model_url, list_of_urls)
        self.max_seq_length = max_seq_length
        self.normalize = normalize
        self.init(model_url)
        self.init_tokenizer(preprocessor_url)

    def init_tokenizer(self, preprocessor_url):
        self.preprocessor = hub.KerasLayer(preprocessor_url)

    def init(self, model_url):
        self.model = hub.KerasLayer(model_url)

    @catch_vector_errors
    def encode(self, text: str):
        return self.model(self.preprocessor([text]))['pooled_output'].numpy().tolist()[0]

    @catch_vector_errors
    def bulk_encode(self, texts: list):
        return self.model(self.preprocessor(texts))['pooled_output'].numpy().tolist()
